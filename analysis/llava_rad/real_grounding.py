"""Anclaje end-to-end sobre NIH real: heatmap de M0-hiLR vs bbox GT."""
import sys, csv, glob
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
CKPT = sorted(glob.glob(r"./outputs/checkpoints/m0_hilr/stage2_epoch=*.ckpt"))[-1]

# bbox por imagen (solo las presentes en disco)
rows = {}
with open(r"./data/raw/nih/BBox_List_2017.csv") as f:
    r = csv.reader(f); next(r)
    for row in r:
        if not row or not row[0]: continue
        name = row[0].strip()
        if (IMGDIR / name).exists():
            rows.setdefault(name, []).append((row[1].strip(), tuple(float(x) for x in row[2:6])))

# una imagen por patología (hasta 8)
by_path, chosen = {}, []
for name, anns in rows.items():
    lab = anns[0][0]
    if lab not in by_path:
        by_path[lab] = name; chosen.append(name)
print(f"Evaluando {len(chosen)} imágenes (1 por patología)")

cfg = OmegaConf.load(r"./configs/base_config.yaml")
tok = AutoTokenizer.from_pretrained(cfg.model.decoder.name); tok.pad_token = tok.eos_token; tok.padding_side = "right"
lm = RadiologyLightningModule(cfg=cfg, tokenizer=tok, stage=2)
lm.load_state_dict(torch.load(CKPT, map_location="cpu")["state_dict"], strict=False)
model = lm.model.eval()
pre = CXRPreprocessor(image_size=224, mode="eval")

hits, lifts = [], []
for name in chosen:
    img = pre(CXRPreprocessor.load_image(str(IMGDIR / name))).unsqueeze(0)
    with torch.no_grad():
        gen = model.generate(img, tokenizer=tok, max_new_tokens=35, num_beams=1)[0]
        ids = tok(gen, return_tensors="pt").input_ids
        _ = model(pixel_values=img, input_ids=ids)
    attn = torch.stack([l.last_attn_weights.squeeze(0) for l in model.cross_attention_layers])
    gmap = attn.mean(1).mean(0).mean(0).reshape(14, 14).numpy()  # global (heads,layers,tokens)
    for lab, bbox in rows[name]:
        s = evaluate_localization(gmap, bbox)
        hits.append(s.pointing_hit); lifts.append(s.lift)
        print(f"  {lab:13s} hit={s.pointing_hit!s:5s} lift={s.lift:.2f} energy={s.energy:.3f} base={s.baseline:.3f}")

print(f"\n=== M0-hiLR sobre NIH ({len(hits)} bboxes) ===")
print(f"Pointing-game hit-rate: {np.mean(hits):.2f}")
print(f"Energy lift medio: {np.mean(lifts):.2f}  (>1 = atiende la región más que el azar)")
