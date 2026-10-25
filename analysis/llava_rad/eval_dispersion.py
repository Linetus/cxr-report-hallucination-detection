"""Eval completa M2-PEFT en GPU: clínica (falso-normal) + dispersión del heatmap en fallos."""
import sys, re, json, math
sys.path.insert(0, r"./")
import numpy as np, torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from src.data import OpenIDataModule
from src.data.preprocessing import CXRPreprocessor as P
from src.training.lightning_module import RadiologyLightningModule

dev = "cuda" if torch.cuda.is_available() else "cpu"
N = 80
CKPT = r"./outputs/checkpoints/m2_peft/stage2_epoch=06_val_rouge_l=0.0000.ckpt"

# vocabulario de hallazgos (positivo = anormal si no está negado)
FIND = ["effusion","pneumonia","consolidation","opacit","nodule","mass","atelectas","edema",
        "emphysema","cardiomegaly","enlarged","infiltrat","pneumothorax","granuloma","scarring",
        "scoliosis","tortuous","fracture","hyperinflat","hyperaerat","pectus","calcified","prominence",
        "device","defibrillat","pacemaker","catheter","tube"]
NEG = re.compile(r"\b(no|without|clear|normal|free of|negative|unremarkable|within normal)\b")
def is_abnormal(text):
    t = text.lower()
    for f in FIND:
        for m in re.finditer(f, t):
            # ventana de 40 chars antes para negación
            if NEG.search(t[max(0,m.start()-40):m.start()]) is None:
                return True
    return False

cfg = OmegaConf.load(r"./configs/base_config.yaml"); cfg.model.encoder.type="rad_dino"
tok = AutoTokenizer.from_pretrained(cfg.model.decoder.name); tok.pad_token=tok.eos_token; tok.padding_side="right"
dm = OpenIDataModule(data_root=cfg.dataset.root, tokenizer=tok, batch_size=16, num_workers=0,
    image_size=518, max_text_length=256, splits_cache=cfg.dataset.splits_cache,
    text_target=cfg.dataset.text_target, split_ratios=dict(cfg.dataset.split_ratios),
    img_mean=list(P.RADDINO_MEAN), img_std=list(P.RADDINO_STD))
dm.prepare_data(); dm.setup()
lm = RadiologyLightningModule(cfg=cfg, tokenizer=tok, stage=2)
lm.load_state_dict(torch.load(CKPT, map_location="cpu")["state_dict"], strict=False)
model = lm.model.eval().to(dev)
print(f"M2-PEFT ep6 en {dev}")

def dispersion(img):
    with torch.no_grad():
        g = model.generate(img, tokenizer=tok, max_new_tokens=55, num_beams=4)[0]
        ids = tok(g, return_tensors="pt").input_ids.to(dev)
        _ = model(pixel_values=img, input_ids=ids)
    attn = torch.stack([l.last_attn_weights.squeeze(0) for l in model.cross_attention_layers])
    gmap = attn.mean(1).mean(0).mean(0).cpu().numpy()   # (S,) distribución sobre patches
    p = gmap / (gmap.sum()+1e-12)
    ent = -(p*np.log(p+1e-12)).sum() / math.log(len(p))  # entropía normalizada [0,1], 1=difuso
    conc = float(gmap.max()/ (gmap.mean()+1e-12))         # concentración, alto=focalizado
    return g, float(ent), conc

rows=[]; seen=0
for batch in dm.val_dataloader():
    for b in range(batch["image"].shape[0]):
        if seen>=N: break
        img = batch["image"][b:b+1].to(dev)
        gen, ent, conc = dispersion(img)
        ref = batch["report_text"][b]
        rows.append({"ref":ref,"gen":gen,"ref_ab":is_abnormal(ref),"gen_ab":is_abnormal(gen),"ent":ent,"conc":conc})
        seen+=1
    if seen>=N: break

json.dump(rows, open(r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad/disp.json","w"))
ref_ab=[r for r in rows if r["ref_ab"]]; ref_norm=[r for r in rows if not r["ref_ab"]]
fn=[r for r in rows if r["ref_ab"] and not r["gen_ab"]]           # falso-normal
tp=[r for r in rows if r["ref_ab"] and r["gen_ab"]]              # detecta anormal
cn=[r for r in rows if not r["ref_ab"] and not r["gen_ab"]]      # correcto-normal
print(f"\n=== CLÍNICA (N={len(rows)}) ===")
print(f"Reales anormales: {len(ref_ab)} | normales: {len(ref_norm)}")
print(f"Sensibilidad (detecta anormal): {len(tp)}/{len(ref_ab)} = {len(tp)/max(len(ref_ab),1):.0%}")
print(f"Falso-normal (real anormal → gen normal): {len(fn)}/{len(ref_ab)} = {len(fn)/max(len(ref_ab),1):.0%}")
def ms(a,k): return (np.mean([r[k] for r in a]) if a else float('nan'))
print(f"\n=== DISPERSIÓN DEL HEATMAP (entropía normalizada; alto=difuso) ===")
print(f"Correcto-normal (n={len(cn)}): entropía={ms(cn,'ent'):.3f} concentración={ms(cn,'conc'):.1f}")
print(f"Falso-normal   (n={len(fn)}): entropía={ms(fn,'ent'):.3f} concentración={ms(fn,'conc'):.1f}")
print(f"Detecta anormal(n={len(tp)}): entropía={ms(tp,'ent'):.3f} concentración={ms(tp,'conc'):.1f}")
print("\n→ Si entropía(falso-normal) > entropía(correcto), el heatmap AVISA del fallo (valor de fiabilidad).")
