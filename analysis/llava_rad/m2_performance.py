"""Rendimiento de M2 in-distribution: ¿usa la imagen? ¿son fieles sus explicaciones?
Decide si el TFM es completable con M2 o es callejón. GPU."""
import sys, glob
sys.path.insert(0, r"./")
import numpy as np, torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from src.data import OpenIDataModule
from src.data.preprocessing import CXRPreprocessor as P
from src.training.lightning_module import RadiologyLightningModule
from src.modules.xai import CrossAttentionExtractor, IntegratedGradientsVisualizer
from src.modules.faithfulness import evaluate_faithfulness

dev = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = r"./outputs/checkpoints/m2_raddino/stage2_epoch=02_val_rouge_l=0.0000.ckpt"
cfg = OmegaConf.load(r"./configs/base_config.yaml"); cfg.model.encoder.type = "rad_dino"
tok = AutoTokenizer.from_pretrained(cfg.model.decoder.name); tok.pad_token = tok.eos_token; tok.padding_side = "right"
dm = OpenIDataModule(data_root=cfg.dataset.root, tokenizer=tok, batch_size=8, num_workers=0,
    image_size=518, max_text_length=256, splits_cache=cfg.dataset.splits_cache,
    text_target=cfg.dataset.text_target, split_ratios=dict(cfg.dataset.split_ratios),
    img_mean=list(P.RADDINO_MEAN), img_std=list(P.RADDINO_STD))
dm.prepare_data(); dm.setup()
lm = RadiologyLightningModule(cfg=cfg, tokenizer=tok, stage=2)
lm.load_state_dict(torch.load(CKPT, map_location="cpu")["state_dict"], strict=False)
model = lm.model.eval().to(dev)
print(f"M2 cargado en {dev}. grid={model.patch_grid}")

batch = next(iter(dm.val_dataloader()))
imgs = batch["image"].to(dev); refs = batch["report_text"]

# 1) SENSIBILIDAD A LA IMAGEN
print("\n=== 1) ¿Usa la imagen? (informes de 6 imágenes distintas) ===")
gens = []
for i in range(6):
    with torch.no_grad():
        g = model.generate(imgs[i:i+1], tokenizer=tok, max_new_tokens=40, num_beams=1)[0]
    gens.append(g); print(f"  img{i}: {g[:85]}")
print(f"  → generaciones únicas: {len(set(gens))}/6")

# 2) FIDELIDAD (CA vs IG) sobre tokens de contenido
print("\n=== 2) ¿Son fieles las explicaciones? (comprehensiveness>0 = fiel) ===")
ca = CrossAttentionExtractor(model); ig = IntegratedGradientsVisualizer(model, n_steps=16)
accC, accI = [], []
for i in range(min(12, imgs.shape[0])):
    img = imgs[i:i+1]
    with torch.no_grad():
        g = model.generate(img, tokenizer=tok, max_new_tokens=35, num_beams=1)[0]
    ids = tok(g, return_tensors="pt").input_ids.to(dev)
    if ids.shape[1] < 4: continue
    try:
        pos = next(j for j in range(1, ids.shape[1]-1)
                   if tok.decode([ids[0,j+1]]).strip().isalpha() and len(tok.decode([ids[0,j+1]]).strip())>4)
    except StopIteration: continue
    tid = int(ids[0,pos+1]); vis = model.encode_image(img)
    impC = ca.extract(img, ids, model.patch_grid)[pos].heatmap.flatten()
    impI = ig.compute(vis, ids, pos, tid, model.patch_grid).heatmap.flatten()
    accC.append(evaluate_faithfulness(model, vis, ids, pos, tid, impC, n_steps=14))
    accI.append(evaluate_faithfulness(model, vis, ids, pos, tid, impI, n_steps=14))
def agg(a,k): return float(np.mean([getattr(s,k) for s in a]))
print(f"  n={len(accC)} tokens")
print(f"  Cross-Attn : compr={agg(accC,'comprehensiveness'):+.3f} suff={agg(accC,'sufficiency'):+.3f} del_auc={agg(accC,'deletion_auc'):.3f}")
print(f"  Integr.Grad: compr={agg(accI,'comprehensiveness'):+.3f} suff={agg(accI,'sufficiency'):+.3f} del_auc={agg(accI,'deletion_auc'):.3f}")
