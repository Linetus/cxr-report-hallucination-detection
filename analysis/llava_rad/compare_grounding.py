"""Compara grounding (energy-lift, pointing) de M0-hiLR vs M1 sobre las MISMAS imágenes NIH."""
import sys, csv, gc
from pathlib import Path
sys.path.insert(0, r"./")
import numpy as np, torch
torch.set_num_threads(4)
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from src.data.preprocessing import CXRPreprocessor
from src.training.lightning_module import RadiologyLightningModule
from src.modules.grounding import evaluate_localization

IMGDIR = Path(r"./data/raw/nih/images")
MODELS = {
    "M0-hiLR": r"./outputs/checkpoints/m0_hilr/stage2_epoch=02_val_rouge_l=0.0000.ckpt",
    "M1-CXRBERT": r"./outputs/checkpoints/m1_s2/stage2_epoch=01_val_rouge_l=0.0000.ckpt",
}

# mismas imágenes: hasta 2 por patología (determinista)
rows = {}
with open(r"./data/raw/nih/BBox_List_2017.csv") as f:
    r = csv.reader(f); next(r)
    for row in r:
        if row and row[0] and (IMGDIR / row[0].strip()).exists():
            rows.setdefault(row[0].strip(), []).append((row[1].strip(), tuple(float(x) for x in row[2:6])))
per_path, chosen = {}, []
for name, anns in rows.items():
    lab = anns[0][0]; per_path.setdefault(lab, 0)
    if per_path[lab] < 2:
        per_path[lab] += 1; chosen.append(name)
print(f"Imágenes de evaluación: {len(chosen)} (≤2 por patología)")

cfg = OmegaConf.load(r"./configs/base_config.yaml")
tok = AutoTokenizer.from_pretrained(cfg.model.decoder.name); tok.pad_token = tok.eos_token; tok.padding_side = "right"
pre = CXRPreprocessor(image_size=224, mode="eval")
imgs = {n: pre(CXRPreprocessor.load_image(str(IMGDIR / n))).unsqueeze(0) for n in chosen}

results = {}
for tag, ckpt in MODELS.items():
    lm = RadiologyLightningModule(cfg=cfg, tokenizer=tok, stage=2)
    lm.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"], strict=False)
    model = lm.model.eval()
    hits, lifts = [], []
    for n in chosen:
        img = imgs[n]
        with torch.no_grad():
            gen = model.generate(img, tokenizer=tok, max_new_tokens=35, num_beams=1)[0]
            ids = tok(gen, return_tensors="pt").input_ids
            _ = model(pixel_values=img, input_ids=ids)
        attn = torch.stack([l.last_attn_weights.squeeze(0) for l in model.cross_attention_layers])
        gmap = attn.mean(1).mean(0).mean(0).reshape(14, 14).numpy()
        for _lab, bbox in rows[n]:
            s = evaluate_localization(gmap, bbox)
            hits.append(s.pointing_hit); lifts.append(s.lift)
    results[tag] = (float(np.mean(hits)), float(np.mean(lifts)), len(hits))
    print(f"  {tag}: pointing={np.mean(hits):.2f} energy-lift={np.mean(lifts):.2f} (n={len(hits)})")
    del lm, model; gc.collect()

print("\n=== COMPARACIÓN GROUNDING (mismas imágenes) ===")
for tag, (h, l, n) in results.items():
    print(f"  {tag:12s} lift={l:.2f}  pointing={h:.2f}")
