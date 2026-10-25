"""Fidelidad Cross-Attn vs IG en M2-PEFT ep6 (GPU, muestra mayor)."""
import sys
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
CKPT = r"./outputs/checkpoints/m2_peft/stage2_epoch=06_val_rouge_l=0.0000.ckpt"
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

ca = CrossAttentionExtractor(model); ig = IntegratedGradientsVisualizer(model, n_steps=24)
accC, accI = [], []
for batch in dm.val_dataloader():
    for b in range(batch["image"].shape[0]):
        if len(accC)>=25: break
        img = batch["image"][b:b+1].to(dev)
        with torch.no_grad():
            g = model.generate(img, tokenizer=tok, max_new_tokens=35, num_beams=1)[0]
        ids = tok(g, return_tensors="pt").input_ids.to(dev)
        if ids.shape[1]<4: continue
        try:
            pos = next(j for j in range(1, ids.shape[1]-1)
                       if tok.decode([ids[0,j+1]]).strip().isalpha() and len(tok.decode([ids[0,j+1]]).strip())>4)
        except StopIteration: continue
        tid=int(ids[0,pos+1]); vis=model.encode_image(img)
        impC = ca.extract(img, ids, model.patch_grid)[pos].heatmap.flatten()
        impI = ig.compute(vis, ids, pos, tid, model.patch_grid).heatmap.flatten()
        accC.append(evaluate_faithfulness(model, vis, ids, pos, tid, impC, n_steps=16))
        accI.append(evaluate_faithfulness(model, vis, ids, pos, tid, impI, n_steps=16))
    if len(accC)>=25: break

def agg(a,k): return float(np.mean([getattr(s,k) for s in a]))
print(f"\n=== FIDELIDAD M2-PEFT (n={len(accC)} tokens) — compr>0=fiel ===")
print(f"{'':16}{'del_AUC↓':>9}{'ins_AUC↑':>9}{'compr↑':>9}{'suff→0':>9}")
print(f"{'Cross-Attn':16}{agg(accC,'deletion_auc'):>9.3f}{agg(accC,'insertion_auc'):>9.3f}{agg(accC,'comprehensiveness'):>+9.3f}{agg(accC,'sufficiency'):>+9.3f}")
print(f"{'Integr.Grad':16}{agg(accI,'deletion_auc'):>9.3f}{agg(accI,'insertion_auc'):>9.3f}{agg(accI,'comprehensiveness'):>+9.3f}{agg(accI,'sufficiency'):>+9.3f}")
