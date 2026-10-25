"""Diagnóstico rápido en CPU: ¿genera palabras o puntos? (greedy, no toca la GPU)."""
import sys
from pathlib import Path
sys.path.insert(0, r"./")

import torch
torch.set_num_threads(4)
from collections import Counter
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.data import OpenIDataModule
from src.training.lightning_module import RadiologyLightningModule

CKPT = r"./outputs/checkpoints/baseline_v2/stage2_epoch=00_val_rouge_l=0.0000.ckpt"
N = 3

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
model = model.eval()  # CPU

batch = next(iter(dm.val_dataloader()))
imgs = batch["image"][:N]
refs = batch["report_text"][:N]

print(f"Generando {N} informes en CPU (greedy, epoch 0 de M0 v2)...\n" + "="*70)
with torch.no_grad():
    gens = model.model.generate(imgs, tokenizer=tok, max_new_tokens=40, num_beams=1)

for i, (g, r) in enumerate(zip(gens, refs)):
    toks = tok.encode(g)
    c = Counter(toks)
    rep = (c.most_common(1)[0][1] / max(len(toks), 1)) if c else 0
    print(f"\n--- Muestra {i} ---")
    print(f"[REF]: {r[:160]}")
    print(f"[GEN]: {repr(g[:220])}")
    print(f"[STAT]: len={len(toks)} unique={len(c)} max_rep_frac={rep:.2f}")
print("\n" + "="*70)
print("Palabras reales ⇒ FIX OK. Solo puntos ⇒ sigue roto.")
