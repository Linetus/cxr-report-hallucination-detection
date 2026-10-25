"""Fidelidad: Cross-Attention vs Integrated Gradients en M0-hiLR (1 token)."""
import sys, glob
sys.path.insert(0, r"./")
import numpy as np, torch
torch.set_num_threads(4)
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from src.data import OpenIDataModule
from src.training.lightning_module import RadiologyLightningModule
from src.modules.xai import CrossAttentionExtractor, IntegratedGradientsVisualizer
from src.modules.faithfulness import evaluate_faithfulness

CKPT = sorted(glob.glob(r"./outputs/checkpoints/m0_hilr/stage2_epoch=*.ckpt"))[-1]
cfg = OmegaConf.load(r"./configs/base_config.yaml")
tok = AutoTokenizer.from_pretrained(cfg.model.decoder.name); tok.pad_token = tok.eos_token; tok.padding_side = "right"
dm = OpenIDataModule(data_root=cfg.dataset.root, tokenizer=tok, batch_size=16, num_workers=0,
    image_size=224, max_text_length=256, splits_cache=cfg.dataset.splits_cache,
    text_target=cfg.dataset.text_target, split_ratios=dict(cfg.dataset.split_ratios))
dm.prepare_data(); dm.setup()
lm = RadiologyLightningModule(cfg=cfg, tokenizer=tok, stage=2)
lm.load_state_dict(torch.load(CKPT, map_location="cpu")["state_dict"], strict=False)
model = lm.model.eval()

img = next(iter(dm.val_dataloader()))["image"][3:4]
with torch.no_grad():
    report = model.generate(img, tokenizer=tok, max_new_tokens=35, num_beams=1)[0]
ids = tok(report, return_tensors="pt").input_ids
toks = [tok.decode([i]) for i in ids[0]]
# posición pos tal que el TOKEN EXPLICADO (pos+1) sea de contenido (hallazgo)
pos = next(i for i in range(1, len(toks) - 1)
           if tok.decode([ids[0, i + 1]]).strip().isalpha()
           and len(tok.decode([ids[0, i + 1]]).strip()) > 4)
tid = int(ids[0, pos + 1])
print(f"Explicando token '{tok.decode([tid])}' (pos {pos+1}) en: {report[:80]}...")

visual = model.encode_image(img)
# importancias (196,) de cada método
ca_maps = CrossAttentionExtractor(model).extract(img, ids, (14, 14))
imp_ca = ca_maps[pos].heatmap.flatten()
imp_ig = IntegratedGradientsVisualizer(model, n_steps=8).compute(visual, ids, pos, tid, (14, 14)).heatmap.flatten()

print("\nCalculando fidelidad (deletion/insertion/compr/suff)...")
sc_ca = evaluate_faithfulness(model, visual, ids, pos, tid, imp_ca, n_steps=14)
sc_ig = evaluate_faithfulness(model, visual, ids, pos, tid, imp_ig, n_steps=14)

print(f"\n{'':16s}{'del_AUC↓':>10}{'ins_AUC↑':>10}{'compr↑':>9}{'suff→0':>9}")
print(f"{'Cross-Attn':16s}{sc_ca.deletion_auc:>10.3f}{sc_ca.insertion_auc:>10.3f}{sc_ca.comprehensiveness:>9.3f}{sc_ca.sufficiency:>9.3f}")
print(f"{'Integrated Grad':16s}{sc_ig.deletion_auc:>10.3f}{sc_ig.insertion_auc:>10.3f}{sc_ig.comprehensiveness:>9.3f}{sc_ig.sufficiency:>9.3f}")
mas_fiel = "Cross-Attn" if sc_ca.comprehensiveness > sc_ig.comprehensiveness else "Integrated Grad"
print(f"\nMás fiel (por comprehensiveness): {mas_fiel}")
print("✓ Módulo de fidelidad funciona.")
