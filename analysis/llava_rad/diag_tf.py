"""Teacher-forcing: ¿predice palabras con contexto real? (discrimina training vs generación)."""
import sys
from pathlib import Path
sys.path.insert(0, r"./")

import torch
torch.set_num_threads(4)
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.data import OpenIDataModule
from src.training.lightning_module import RadiologyLightningModule

CKPT = r"./outputs/checkpoints/baseline_v2/stage2_epoch=00_val_rouge_l=0.0000.ckpt"

cfg = OmegaConf.load(r"./configs/base_config.yaml")
tok = AutoTokenizer.from_pretrained(cfg.model.decoder.name)
tok.pad_token = tok.eos_token
tok.padding_side = "right"

dm = OpenIDataModule(
    data_root=cfg.dataset.root, tokenizer=tok,
    batch_size=cfg.dataloader.batch_size, num_workers=0,
    image_size=cfg.preprocessing.image_size, max_text_length=cfg.preprocessing.max_text_length,
    splits_cache=cfg.dataset.splits_cache, text_target=cfg.dataset.text_target,
    split_ratios=dict(cfg.dataset.split_ratios),
)
dm.prepare_data(); dm.setup()

model = RadiologyLightningModule(cfg=cfg, tokenizer=tok, stage=2)
ckpt = torch.load(CKPT, map_location="cpu")
model.load_state_dict(ckpt["state_dict"], strict=False)
model = model.eval()

batch = next(iter(dm.val_dataloader()))
img = batch["image"][:1]
ids = batch["input_ids"][:1]
am = batch["attention_mask"][:1]
n_real = int(am[0].sum().item())

with torch.no_grad():
    out = model.model(pixel_values=img, input_ids=ids, attention_mask=am, labels=None)
    logits = out.logits            # (1, T, V)
    pred = logits[0].argmax(-1)    # (T,)

real_ids = ids[0][:n_real]
# predicción del siguiente token en cada posición (teacher-forced)
pred_next = pred[:n_real-1]
print("Tokens reales (primeros 40):", n_real, "tokens")
print("[REAL   ]:", repr(tok.decode(real_ids[:40])))
print("[PRED-TF]:", repr(tok.decode(pred_next[:40])))
# ¿cuántas predicciones aciertan el siguiente token real?
target_next = real_ids[1:n_real]
acc = (pred_next == target_next).float().mean().item()
print(f"\nNext-token acc (teacher-forced) sobre {n_real-1} posiciones: {acc:.3f}")
uniq = len(set(pred_next.tolist()))
print(f"Tokens únicos predichos: {uniq}  (si ~1 y son puntos ⇒ colapso profundo)")
