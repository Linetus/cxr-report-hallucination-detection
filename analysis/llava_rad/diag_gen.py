"""Diagnóstico del colapso de val/rouge_l: inspeccionar generaciones reales."""
import sys
from pathlib import Path
sys.path.insert(0, r"./")

import torch
from collections import Counter
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.data import OpenIDataModule
from src.training.lightning_module import RadiologyLightningModule

CKPT = r"./outputs/checkpoints/stage2_epoch=05_val_rouge_l=0.0000.ckpt"
N = 6

cfg = OmegaConf.load(r"./configs/base_config.yaml")
tok = AutoTokenizer.from_pretrained(cfg.model.decoder.name)
tok.pad_token = tok.eos_token
tok.padding_side = "left"

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
missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
model = model.eval().cuda()

batch = next(iter(dm.val_dataloader()))
imgs = batch["image"][:N].cuda()
refs = batch["report_text"][:N]

print(f"\nGenerando {N} informes desde epoch 5 (beam=4)...\n" + "="*70)
with torch.no_grad():
    gens = model.model.generate(imgs, tokenizer=tok, max_new_tokens=cfg.model.decoder.max_length, num_beams=4)

for i, (g, r) in enumerate(zip(gens, refs)):
    toks = tok.encode(g)
    c = Counter(toks)
    top_tok, top_n = (c.most_common(1)[0] if c else (0, 0))
    rep = top_n / max(len(toks), 1)
    print(f"\n--- Muestra {i} ---")
    print(f"[REF ]: {r[:200]}")
    print(f"[GEN ]: {repr(g[:300])}")
    print(f"[STAT]: len_tokens={len(toks)} unique={len(c)} max_rep_frac={rep:.2f} "
          f"top_token={repr(tok.decode([top_tok]))}")
print("\n" + "="*70)
print("Interpretación: max_rep_frac alto (>0.5) o texto vacío/repetido ⇒ degeneración.")
