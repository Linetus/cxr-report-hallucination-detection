"""Grounding: M0-hiLR vs M1 vs M2(ep2) sobre las MISMAS imágenes NIH (preproc por modelo)."""
import sys, csv, gc
from pathlib import Path
sys.path.insert(0, r"./")
import numpy as np, torch
torch.set_num_threads(4)
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from src.data.preprocessing import CXRPreprocessor as P
from src.training.lightning_module import RadiologyLightningModule
from src.modules.grounding import evaluate_localization

IMG = Path(r"./data/raw/nih/images")
CK = r"./outputs/checkpoints"
# (tag, ckpt, encoder_type, img_size, mean, std)
MODELS = [
    ("M0-hiLR", rf"{CK}\m0_hilr\stage2_epoch=02_val_rouge_l=0.0000.ckpt", "vit", 224, None, None),
    ("M1-CXRBERT", rf"{CK}\m1_s2\stage2_epoch=01_val_rouge_l=0.0000.ckpt", "vit", 224, None, None),
    ("M2-RADDINO", rf"{CK}\m2_raddino\stage2_epoch=02_val_rouge_l=0.0000.ckpt", "rad_dino", 518, P.RADDINO_MEAN, P.RADDINO_STD),
]

rows = {}
with open(r"./data/raw/nih/BBox_List_2017.csv") as f:
    r = csv.reader(f); next(r)
    for row in r:
        if row and row[0] and (IMG / row[0].strip()).exists():
            rows.setdefault(row[0].strip(), []).append((row[1].strip(), tuple(float(x) for x in row[2:6])))
per, chosen = {}, []
for n, a in rows.items():
    lab = a[0][0]; per.setdefault(lab, 0)
    if per[lab] < 2: per[lab] += 1; chosen.append(n)
print(f"Imágenes: {len(chosen)} (≤2/patología)")

cfg = OmegaConf.load(r"./configs/base_config.yaml")
tok = AutoTokenizer.from_pretrained(cfg.model.decoder.name); tok.pad_token = tok.eos_token; tok.padding_side = "right"

res = {}
for tag, ckpt, etype, size, mean, std in MODELS:
    cfg.model.encoder.type = etype
    lm = RadiologyLightningModule(cfg=cfg, tokenizer=tok, stage=2)
    lm.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"], strict=False)
    model = lm.model.eval()
    pre = P(image_size=size, mode="eval", mean=mean, std=std)
    grid = model.patch_grid
    hits, lifts = [], []
    for n in chosen:
        img = pre(P.load_image(str(IMG / n))).unsqueeze(0)
        with torch.no_grad():
            gen = model.generate(img, tokenizer=tok, max_new_tokens=30, num_beams=1)[0]
            ids = tok(gen, return_tensors="pt").input_ids
            _ = model(pixel_values=img, input_ids=ids)
        attn = torch.stack([l.last_attn_weights.squeeze(0) for l in model.cross_attention_layers])
        gmap = attn.mean(1).mean(0).mean(0).reshape(*grid).numpy()
        for _lab, bbox in rows[n]:
            s = evaluate_localization(gmap, bbox)
            hits.append(s.pointing_hit); lifts.append(s.lift)
    res[tag] = (float(np.mean(lifts)), float(np.mean(hits)), len(hits), grid)
    print(f"  {tag:12s} grid{grid} lift={np.mean(lifts):.2f} pointing={np.mean(hits):.2f} (n={len(hits)})")
    del lm, model; gc.collect()

print("\n=== GROUNDING (mismas imágenes NIH) ===")
for tag, (l, h, n, g) in res.items():
    print(f"  {tag:12s} energy-lift={l:.2f}  pointing={h:.2f}")
