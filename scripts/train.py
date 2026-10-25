"""
Training entry point — supports Stage 1 (contrastive) and Stage 2 (generation).

Usage:
    # Stage 1 — visual-text alignment (~4-6h on RTX 4070 with Open-I)
    python scripts/train.py stage=1

    # Stage 2 — report generation fine-tuning (loads Stage 1 encoder)
    python scripts/train.py stage=2 stage1_checkpoint=outputs/checkpoints/stage1_best.ckpt

    # Override any config key
    python scripts/train.py stage=1 training.stage1.epochs=10 dataloader.batch_size=8
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer

from src.data import OpenIDataModule, MIMICCXRDataModule
from src.training.lightning_module import RadiologyLightningModule


def build_callbacks(cfg, stage: int, run_name: str | None = None) -> list:
    monitor = "val/rouge_l" if stage == 2 else "val/loss"
    mode = "max" if stage == 2 else "min"

    # run_name separa checkpoints por brazo experimental (evita colisiones);
    # sin run_name se mantiene el comportamiento por defecto (dir raíz)
    from pathlib import Path as _P
    ckpt_dir = str(_P(cfg.paths.checkpoint_dir) / run_name) if run_name else cfg.paths.checkpoint_dir

    monitor_key = monitor.replace("/", "_")
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"stage{stage}_{{epoch:02d}}_{{{monitor_key}:.4f}}",
        monitor=monitor,
        mode=mode,
        save_top_k=cfg.logging.save_top_k,
        verbose=True,
    )
    early_stop_cb = EarlyStopping(
        monitor=monitor,
        patience=5,
        mode=mode,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    return [checkpoint_cb, early_stop_cb, lr_monitor]


def main():
    # Activa Tensor Cores en la RTX 4070 para operaciones float32
    torch.set_float32_matmul_precision("medium")

    # ── Load config ──────────────────────────────────────────────────────────
    base_cfg = OmegaConf.load("configs/base_config.yaml")

    # Merge CLI overrides: python train.py stage=2 dataloader.batch_size=8
    cli_overrides = OmegaConf.from_dotlist(
        [a for a in sys.argv[1:] if "=" in a]
    )
    cfg = OmegaConf.merge(base_cfg, cli_overrides)
    # Accept both `stage=2` (top-level shorthand) and `training.stage=2`
    stage = int(cfg.get("stage", cfg.training.get("stage", 1)))

    print(OmegaConf.to_yaml(cfg))
    print(f"\n{'='*50}")
    print(f"  TRAINING STAGE {stage}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
          if torch.cuda.is_available() else "")
    print(f"{'='*50}\n")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.decoder.name)
    tokenizer.pad_token = tokenizer.eos_token
    # LM causal: pad a la DERECHA en entrenamiento (los tokens reales empiezan en pos 0,
    # position_ids correctos). El left-padding solo es para generación batcheada con prompt
    # de texto — aquí generate() arranca desde un único BOS, así que no lo necesita.
    tokenizer.padding_side = "right"

    # ── Data ─────────────────────────────────────────────────────────────────
    # Los embeddings CXR-BERT solo hacen falta en el Stage 1 con brazo B/C
    text_tower = cfg.model.get("stage1", {}).get("text_tower", "baseline")
    use_cxrbert = stage == 1 and text_tower != "baseline"
    use_img_kd = stage == 1 and bool(cfg.model.get("stage1", {}).get("image_kd", False))

    # RAD-DINO requiere 518px + su propia normalización (NO ImageNet)
    encoder_type = cfg.model.encoder.get("type", "vit")
    if encoder_type == "rad_dino":
        from src.data.preprocessing import CXRPreprocessor as _P
        img_size, img_mean, img_std = 518, list(_P.RADDINO_MEAN), list(_P.RADDINO_STD)
        print(f"  Encoder RAD-DINO → imagen 518px, normalización propia CXR")
    else:
        img_size, img_mean, img_std = cfg.preprocessing.image_size, None, None

    dm_cls = OpenIDataModule if cfg.dataset.name == "open_i" else MIMICCXRDataModule
    dm = dm_cls(
        data_root=cfg.dataset.root,
        tokenizer=tokenizer,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        image_size=img_size,
        max_text_length=cfg.preprocessing.max_text_length,
        **({
            "splits_cache": cfg.dataset.splits_cache,
            "text_target": cfg.dataset.text_target,
            "split_ratios": dict(cfg.dataset.split_ratios),
            "cxrbert_cache": cfg.model.stage1.cxrbert_cache if use_cxrbert else None,
            "raddino_cache": cfg.model.stage1.raddino_cache if use_img_kd else None,
            "img_mean": img_mean, "img_std": img_std,
        } if cfg.dataset.name == "open_i" else {}),
    )
    if use_img_kd:
        print(f"  KD de imagen Stage 1: ViT → RAD-DINO (M3)")
    if use_cxrbert:
        print(f"  Torre de texto Stage 1: {text_tower} (CXR-BERT teacher)")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = RadiologyLightningModule(cfg=cfg, tokenizer=tokenizer, stage=stage)

    # run_name opcional: separa checkpoints y logs por brazo experimental
    run_name = cfg.get("run_name", None)

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = TensorBoardLogger(
        save_dir=cfg.paths.output_dir,
        name=f"stage{stage}",
        version=run_name or "latest",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    stage_cfg = cfg.training.stage1 if stage == 1 else cfg.training.stage2
    trainer = pl.Trainer(
        max_epochs=stage_cfg.epochs,
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.gradient_clip_val,
        callbacks=build_callbacks(cfg, stage, run_name),
        logger=logger,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        val_check_interval=1.0,
        enable_progress_bar=True,
        deterministic=False,   # slightly faster on RTX 4070
    )

    # ── Fit ───────────────────────────────────────────────────────────────────
    # Llamar setup() explícitamente antes de fit() para que estimated_stepping_batches
    # funcione correctamente en configure_optimizers (evita num_samples=0)
    ckpt_base = Path(cfg.paths.checkpoint_dir) / run_name if run_name else Path(cfg.paths.checkpoint_dir)
    ckpt_base.mkdir(parents=True, exist_ok=True)
    dm.prepare_data()
    dm.setup()
    print(f"  Dataset splits: {dm.split_sizes}")

    # Resume from checkpoint if provided: python train.py stage=2 resume_from=path/to/ckpt.ckpt
    resume_ckpt = cfg.get("resume_from", None)
    if resume_ckpt:
        print(f"  Resuming from: {resume_ckpt}")
    trainer.fit(model, datamodule=dm, ckpt_path=resume_ckpt)

    best = trainer.checkpoint_callback.best_model_path
    print(f"\n✓ Training complete. Best checkpoint: {best}")
    return best


if __name__ == "__main__":
    main()
