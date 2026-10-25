"""
PyTorch Lightning module — two-stage training for RadiologyReportModel.

Stage 1 (contrastive): trains encoder + projection heads with InfoNCE loss.
  Visual and text embeddings are aligned in a shared latent space (CLIP-style).
  This warm-starts the encoder before the heavier Stage 2.

Stage 2 (generation): trains the full encoder-decoder with language modelling loss.
  Cross-attention layers learn to ground each generated token in image regions.
  This is the stage that enables XAI heatmaps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.optim as optim
from omegaconf import DictConfig
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from src.models.model import RadiologyReportModel
from src.utils.metrics import compute_nlp_metrics


class RadiologyLightningModule(pl.LightningModule):
    """
    Args:
        cfg: OmegaConf config (base_config.yaml)
        tokenizer: HuggingFace tokenizer matching the decoder
        stage: 1 (contrastive alignment) or 2 (generation fine-tuning)
    """

    def __init__(self, cfg: DictConfig, tokenizer, stage: int = 1):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.stage = stage
        self.save_hyperparameters(ignore=["tokenizer"])

        stage1_cfg = cfg.model.get("stage1", {})
        self.text_tower = stage1_cfg.get("text_tower", "baseline")
        self.kd_weight = float(stage1_cfg.get("kd_weight", 1.0))
        self.image_kd = bool(stage1_cfg.get("image_kd", False))
        self.img_kd_weight = float(stage1_cfg.get("img_kd_weight", 1.0))
        self.encoder_type = cfg.model.encoder.get("type", "vit")
        self.freeze_gpt2 = bool(cfg.model.encoder.get("freeze_gpt2", False))

        self.model = RadiologyReportModel(
            encoder_name=cfg.model.encoder.name,
            decoder_name=cfg.model.decoder.name,
            visual_dim=cfg.model.encoder.out_dim,
            encoder_pretrained=cfg.model.encoder.pretrained,
            decoder_pretrained=cfg.model.decoder.pretrained,
            freeze_encoder_layers=cfg.model.encoder.freeze_layers,
            num_cross_attn_heads=cfg.model.cross_attention.num_heads,
            dropout=cfg.model.encoder.dropout,
            text_tower=self.text_tower,
            cxrbert_dim=int(stage1_cfg.get("cxrbert_dim", 128)),
            encoder_type=self.encoder_type,
            freeze_gpt2=self.freeze_gpt2,
        )

        # Stage 2: load Stage 1 encoder weights if provided.
        # RAD-DINO ya viene preentrenado y congelado → no se carga encoder de Stage 1.
        if stage == 2 and cfg.get("stage1_checkpoint") and self.encoder_type != "rad_dino":
            self._load_encoder_from_stage1(cfg.stage1_checkpoint)

        self._val_predictions: list[str] = []
        self._val_references: list[str] = []

    # ─── FORWARD ─────────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> torch.Tensor:
        if self.stage == 1:
            out = self.model.compute_contrastive_loss(
                pixel_values=batch["image"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                temperature=self.cfg.training.stage1.temperature,
                cxrbert_emb=batch.get("cxrbert_emb"),
                kd_weight=self.kd_weight,
                raddino_emb=batch.get("raddino_emb"),
                img_kd_weight=self.img_kd_weight,
            )
            # Guarda componentes (infonce/kd/img_kd) para logging en el step correspondiente
            self._stage1_comp = {
                "infonce": out["infonce"], "kd": out["kd"], "img_kd": out["img_kd"],
            }
            return out["loss"]
        else:
            # Shift labels left by 1 for next-token prediction
            labels = batch["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            output = self.model(
                pixel_values=batch["image"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=labels,
            )
            return output.loss

    # ─── TRAINING ────────────────────────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # Stage 1: registra InfoNCE y los KD activos por separado
        if self.stage == 1 and (self.text_tower == "cxrbert_kd" or self.image_kd):
            comp = self._stage1_comp
            self.log("train/infonce", comp["infonce"], on_step=True, on_epoch=True)
            if self.text_tower == "cxrbert_kd":
                self.log("train/kd", comp["kd"], on_step=True, on_epoch=True, prog_bar=True)
            if self.image_kd:
                self.log("train/img_kd", comp["img_kd"], on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ─── VALIDATION ──────────────────────────────────────────────────────────

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss = self(batch)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        # Generate reports and accumulate for NLP metrics (Stage 2 only)
        if self.stage == 2 and batch_idx < 20:    # limit to first 20 batches for speed
            generated = self.model.generate(
                pixel_values=batch["image"],
                tokenizer=self.tokenizer,
                max_new_tokens=self.cfg.model.decoder.max_length,
                num_beams=4,
            )
            self._val_predictions.extend(generated)
            self._val_references.extend(batch["report_text"])

    def on_validation_epoch_end(self) -> None:
        if self.stage == 2 and self._val_predictions:
            metrics = compute_nlp_metrics(
                predictions=self._val_predictions,
                references=self._val_references,
            )
            for k, v in metrics.items():
                self.log(f"val/{k}", v, prog_bar=(k in ("rouge_l", "bleu_4")))
        self._val_predictions.clear()
        self._val_references.clear()

    # ─── OPTIMIZERS ──────────────────────────────────────────────────────────

    def configure_optimizers(self):
        stage_cfg = self.cfg.training.stage1 if self.stage == 1 else self.cfg.training.stage2

        # Stage 1: only train encoder + projection heads (según el brazo de texto)
        if self.stage == 1:
            params = (
                list(self.model.encoder.parameters())
                + list(self.model.image_proj.parameters())
                + [self.model.logit_scale]
            )
            if self.text_tower == "cxrbert_tower":
                # Brazo B: teacher congelado; se entrena la proyección del teacher
                params += list(self.model.cxrbert_proj.parameters())
            else:
                # baseline y cxrbert_kd usan la torre student (text_proj)
                params += list(self.model.text_proj.parameters())
                if self.text_tower == "cxrbert_kd":
                    params += list(self.model.kd_head.parameters())
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]

        # Stage 2: LR separado (mayor) para la vía visual (cross-attn + gate + visual_proj).
        # Fuerza que los gates se abran y el modelo aprenda a usar la imagen en vez de
        # colapsar en la plantilla textual. cross_attn_lr=null → comportamiento por defecto.
        cross_attn_lr = stage_cfg.get("cross_attn_lr", None) if self.stage == 2 else None
        if cross_attn_lr:
            ca, base = [], []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                (ca if ("cross_attn" in n or "visual_proj" in n) else base).append(p)
            optimizer = optim.AdamW(
                [{"params": base, "lr": stage_cfg.lr},
                 {"params": ca, "lr": float(cross_attn_lr)}],
                weight_decay=stage_cfg.weight_decay,
            )
            print(f"  Optimizer: base lr={stage_cfg.lr}, cross-attn lr={cross_attn_lr} "
                  f"({len(ca)} tensores en la vía visual)")
        else:
            optimizer = optim.AdamW(
                params,
                lr=stage_cfg.lr,
                weight_decay=stage_cfg.weight_decay,
            )

        # Estimate total steps for the cosine scheduler
        steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs \
            if self.trainer else 100
        total_steps = self.trainer.max_epochs * steps_per_epoch if self.trainer else 1000
        warmup_steps = getattr(stage_cfg, "warmup_steps", max(total_steps // 10, 100))

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # ─── HELPERS ─────────────────────────────────────────────────────────────

    def _load_encoder_from_stage1(self, checkpoint_path: str) -> None:
        """Load encoder weights from a Stage 1 checkpoint for Stage 2 warm start."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = {
            k.replace("model.encoder.", ""): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.encoder.")
        }
        missing, unexpected = self.model.encoder.load_state_dict(state, strict=False)
        if missing:
            print(f"[Stage2] Missing encoder keys from Stage1 ckpt: {missing[:5]}")
