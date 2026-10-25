"""Grounding CONDICIONAL: solo cuando el modelo diagnostica el hallazgo, mira si
el heatmap de ESA frase cae en el bbox. Mide zona+diagnóstico juntos."""
import sys, csv, re
from pathlib import Path
sys.path.insert(0, r"./")
import numpy as np, torch
torch.set_num_threads(4)
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from src.data.preprocessing import CXRPreprocessor as P
from src.training.lightning_module import RadiologyLightningModule
from src.modules.xai import ReportHeatmapAligner
from src.modules.grounding import evaluate_localization

IMG = Path(r"./data/raw/nih/images")
CKPT = r"./outputs/checkpoints/m0_hilr/stage2_epoch=02_val_rouge_l=0.0000.ckpt"

# NIH → palabras clave en el informe (positivas)
KW = {
    "Cardiomegaly": ["cardiomegaly", "enlarged", "cardiac silhouette", "heart size"],
    "Effusion": ["effusion", "pleural fluid"],
    "Pneumonia": ["pneumonia"],
    "Atelectasis": ["atelectas"],
    "Nodule": ["nodule", "nodular"],
    "Mass": ["mass"],
    "Pneumothorax": ["pneumothorax"],
    "Infiltrate": ["infiltrat", "opacit"],
}
NEG = re.compile(r"\b(no|without|clear|normal|free of|negative)\b")

def positive_mention(sentence, pathology):
    s = sentence.lower()
    if not any(k in s for k in KW.get(pathology, [])):
        return False
    return NEG.search(s) is None   # descarta negaciones ("no effusion")

rows = {}
with open(r"./data/raw/nih/BBox_List_2017.csv") as f:
    r = csv.reader(f); next(r)
    for row in r:
        if row and row[0] and (IMG / row[0].strip()).exists():
            rows.setdefault(row[0].strip(), []).append((row[1].strip(), tuple(float(x) for x in row[2:6])))
names = list(rows)[:200]

cfg = OmegaConf.load(r"./configs/base_config.yaml"); cfg.model.encoder.type = "vit"
tok = AutoTokenizer.from_pretrained(cfg.model.decoder.name); tok.pad_token = tok.eos_token; tok.padding_side = "right"
lm = RadiologyLightningModule(cfg=cfg, tokenizer=tok, stage=2)
lm.load_state_dict(torch.load(CKPT, map_location="cpu")["state_dict"], strict=False)
model = lm.model.eval()
aligner = ReportHeatmapAligner(model, tok, xai_methods=("cross_attention",))
pre = P(image_size=224, mode="eval")

total_bbox, mentioned, hits, lifts = 0, 0, [], []
for n in names:
    img = pre(P.load_image(str(IMG / n))).unsqueeze(0)
    report = aligner.generate_report(img, max_new_tokens=40, num_beams=1)
    res = aligner.align(img, report)["cross_attention"]
    for lab, bbox in rows[n]:
        total_bbox += 1
        sent = next((sh for sh in res if positive_mention(sh.sentence, lab)), None)
        if sent is None:
            continue
        mentioned += 1
        s = evaluate_localization(sent.heatmap, bbox)
        hits.append(s.pointing_hit); lifts.append(s.lift)

print(f"=== M0-hiLR grounding CONDICIONAL ({len(names)} imgs, {total_bbox} bboxes) ===")
print(f"Diagnóstico positivo del hallazgo: {mentioned}/{total_bbox} = {mentioned/total_bbox:.1%}")
if lifts:
    print(f"Grounding CUANDO diagnostica: energy-lift={np.mean(lifts):.2f}  pointing={np.mean(hits):.2f}  (n={len(lifts)})")
else:
    print("El modelo casi nunca diagnostica positivamente las patologías NIH (sesgo a 'normal').")
