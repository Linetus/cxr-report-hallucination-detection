"""
RadiologyReportModel — full encoder-decoder assembly.

Combines VisualEncoder + ReportDecoder and exposes the interface expected by:
  - ReportHeatmapAligner (XAI): encode_image(), decoder.cross_attention_layers
  - demo.py:                    generate(), forward()
  - LightningModule:            forward(), compute_contrastive_loss()
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import VisualEncoder, RadDinoEncoder
from .decoder import ReportDecoder


class RadiologyReportModel(nn.Module):
    """
    Encoder-CrossAttention-Decoder model for chest X-ray report generation.

    Training has two stages (controlled by LightningModule):

    Stage 1 — Visual-Text Alignment (CLIP-style contrastive)
        Only the encoder and a projection head are trained.
        Objective: CLS embeddings of paired images and report tokens
        should be closer than unpaired ones (InfoNCE loss).

    Stage 2 — Report Generation Fine-tuning
        The full model (encoder + decoder with cross-attention) is trained
        with a language modelling loss (cross-entropy on next token prediction).
    """

    def __init__(
        self,
        encoder_name: str = "vit_base_patch16_224",
        decoder_name: str = "gpt2",
        visual_dim: int = 768,
        encoder_pretrained: bool = True,
        decoder_pretrained: bool = True,
        freeze_encoder_layers: int = 0,
        num_cross_attn_heads: int = 8,
        dropout: float = 0.1,
        contrastive_proj_dim: int = 256,    # projection head for Stage 1
        text_tower: str = "baseline",       # baseline | cxrbert_tower | cxrbert_kd
        cxrbert_dim: int = 128,             # CXR-BERT projected embedding dim
        encoder_type: str = "vit",          # vit (timm) | rad_dino (frozen, transfer)
        freeze_gpt2: bool = False,          # PEFT: congela GPT-2 base, entrena solo cross-attn
    ):
        super().__init__()
        self.text_tower = text_tower
        self.encoder_type = encoder_type

        if encoder_type == "rad_dino":
            # Encoder de dominio congelado (M2/M3): 518px, 37×37 patches, 768D
            self.encoder = RadDinoEncoder(freeze=True, dropout=0.0)
        else:
            self.encoder = VisualEncoder(
                model_name=encoder_name,
                pretrained=encoder_pretrained,
                output_dim=visual_dim,
                freeze_layers=freeze_encoder_layers,
                dropout=dropout,
            )

        self.decoder = ReportDecoder(
            model_name=decoder_name,
            visual_dim=visual_dim,
            num_cross_attn_heads=num_cross_attn_heads,
            pretrained=decoder_pretrained,
            dropout=dropout,
        )

        # Stage 1 projection heads (CLIP-style)
        self.image_proj = nn.Sequential(
            nn.Linear(visual_dim, contrastive_proj_dim),
            nn.ReLU(),
            nn.Linear(contrastive_proj_dim, contrastive_proj_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.decoder.hidden_dim, contrastive_proj_dim),
            nn.ReLU(),
            nn.Linear(contrastive_proj_dim, contrastive_proj_dim),
        )

        # ── Cabezas específicas de brazo (solo se construyen si el brazo las usa,
        #    así el modelo baseline queda idéntico y los checkpoints previos cargan) ──
        if text_tower == "cxrbert_tower":
            # Brazo B: proyecta el embedding congelado del teacher al espacio contrastivo
            self.cxrbert_proj = nn.Sequential(
                nn.Linear(cxrbert_dim, contrastive_proj_dim),
                nn.ReLU(),
                nn.Linear(contrastive_proj_dim, contrastive_proj_dim),
            )
        elif text_tower == "cxrbert_kd":
            # Brazo C: cabeza de destilación — el student predice el embedding del teacher
            self.kd_head = nn.Linear(contrastive_proj_dim, cxrbert_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # log(1/0.07)

        # PEFT opcional (M2 agresivo): congela GPT-2 base, deja entrenables SOLO las
        # cross-attention (+gate) y visual_proj. Reduce params entrenables ~200M→~5M.
        if freeze_gpt2:
            for p in self.decoder.gpt2.parameters():
                p.requires_grad = False
            for block in self.decoder.gpt2.transformer.h:
                for p in block.cross_attn.parameters():
                    p.requires_grad = True

    # ─── CORE INTERFACES ─────────────────────────────────────────────────────

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Returns patch features (B, num_patches, D) for cross-attention.
        Used by XAI module (ReportHeatmapAligner).
        """
        return self.encoder(pixel_values)["patch_features"]

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Stage 2 forward: image → patch features → decoder → logits + loss.

        Returns HuggingFace CausalLMOutput with .loss and .logits.
        """
        visual_features = self.encode_image(pixel_values)
        return self.decoder.decode(
            input_ids=input_ids,
            visual_features=visual_features,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 200,
        num_beams: int = 4,
    ) -> list[str]:
        """Autoregressive report generation at inference."""
        visual_features = self.encode_image(pixel_values)
        return self.decoder.generate(
            visual_features=visual_features,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

    # ─── STAGE 1: CONTRASTIVE ALIGNMENT ──────────────────────────────────────

    def _student_text_embedding(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Text embedding del student: GPT-2 wte (congelado) → mean-pool → text_proj.

        Compartido por los brazos `baseline` y `cxrbert_kd`; devuelve el embedding
        contrastivo SIN normalizar (256D) para que la KD pueda operar sobre él."""
        with torch.no_grad():
            tok_emb = self.decoder.gpt2.transformer.wte(input_ids)   # (B, T, D)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (tok_emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = tok_emb.mean(1)
        return self.text_proj(pooled)                                # (B, P) sin normalizar

    def compute_contrastive_loss(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 0.07,
        cxrbert_emb: Optional[torch.Tensor] = None,
        kd_weight: float = 1.0,
        raddino_emb: Optional[torch.Tensor] = None,
        img_kd_weight: float = 1.0,
    ) -> dict:
        """
        Pérdida InfoNCE imagen↔texto para el Stage 1, con tres torres de texto:

          - baseline       : texto = GPT-2 wte mean-pool (bag-of-words débil)
          - cxrbert_tower  : texto = [CLS] proyectado de CXR-BERT (teacher congelado)
          - cxrbert_kd     : texto = student entrenable + destilación hacia el teacher

        KD de imagen (M3, independiente de la torre de texto): si se pasa `raddino_emb`,
        destila el CLS del ViT student hacia el CLS de RAD-DINO (teacher, 768→768 coseno).

        Devuelve dict {loss, infonce, kd, img_kd}.
        """
        # ── Lado imagen (idéntico en los tres brazos → comparación justa) ──
        enc_out = self.encoder(pixel_values)
        img_emb = F.normalize(self.image_proj(enc_out["cls_embedding"]), dim=-1)

        # ── KD de imagen (M3): ViT CLS → RAD-DINO CLS ──
        img_kd = torch.zeros((), device=img_emb.device)
        if raddino_emb is not None:
            vit_cls = F.normalize(enc_out["cls_embedding"], dim=-1)   # (B, 768)
            teacher_img = F.normalize(raddino_emb, dim=-1)            # (B, 768)
            img_kd = (1.0 - (vit_cls * teacher_img).sum(-1)).mean()

        # ── Lado texto según el brazo ──
        kd = torch.zeros((), device=img_emb.device)
        if self.text_tower == "cxrbert_tower":
            if cxrbert_emb is None:
                raise ValueError("cxrbert_tower requiere cxrbert_emb en el batch.")
            txt_emb = F.normalize(self.cxrbert_proj(cxrbert_emb), dim=-1)
        else:
            # baseline y cxrbert_kd comparten el student text path
            txt_raw = self._student_text_embedding(input_ids, attention_mask)  # (B, P)
            txt_emb = F.normalize(txt_raw, dim=-1)
            if self.text_tower == "cxrbert_kd":
                if cxrbert_emb is None:
                    raise ValueError("cxrbert_kd requiere cxrbert_emb en el batch.")
                teacher = F.normalize(cxrbert_emb, dim=-1)                     # (B, 128)
                student_pred = F.normalize(self.kd_head(txt_raw), dim=-1)      # (B, 128)
                kd = (1.0 - (student_pred * teacher).sum(-1)).mean()

        # ── InfoNCE simétrica (diagonal = pares positivos) ──
        scale = self.logit_scale.exp().clamp(max=100)
        logits = scale * img_emb @ txt_emb.T                 # (B, B)
        labels = torch.arange(logits.shape[0], device=logits.device)
        infonce = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

        loss = infonce + kd_weight * kd + img_kd_weight * img_kd
        return {"loss": loss, "infonce": infonce.detach(),
                "kd": kd.detach(), "img_kd": img_kd.detach()}

    # ─── XAI INTERFACE ───────────────────────────────────────────────────────

    @property
    def cross_attention_layers(self):
        """Expose decoder cross-attention layers for XAI extraction."""
        return self.decoder.cross_attention_layers

    @property
    def patch_grid(self) -> tuple[int, int]:
        return self.encoder.patch_grid
