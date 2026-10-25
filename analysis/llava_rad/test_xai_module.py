"""Valida la reescritura de src/modules/xai.py end-to-end (CPU)."""
import sys, glob
sys.path.insert(0, r"./")
import numpy as np, torch
torch.set_num_threads(4)
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from src.data import OpenIDataModule
from src.training.lightning_module import RadiologyLightningModule
from src.modules.xai import ReportHeatmapAligner, IntegratedGradientsVisualizer

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

# ── Cross-attention vía el aligner ──
aligner = ReportHeatmapAligner(model, tok, xai_methods=("cross_attention",))
report = aligner.generate_report(img, max_new_tokens=45, num_beams=1)
print("REPORT:", report[:150])
res = aligner.align(img, report)
sh = res["cross_attention"]
print(f"\n[CROSS-ATTENTION] {len(sh)} frases:")
for s in sh:
    conc = s.heatmap.max() / (s.heatmap.mean() + 1e-8)
    print(f"  · '{s.sentence[:45]}' heatmap{s.heatmap.shape} conc={conc:.1f} "
          f"peak_px={int(s.peak_region.sum())}")

# ── IG en 1 token (n_steps bajo para velocidad) ──
print("\n[INTEGRATED GRADIENTS] test 1 token (n_steps=6)...")
ig = IntegratedGradientsVisualizer(model, n_steps=6)
visual = model.encode_image(img)
ids = tok(report, return_tensors="pt").input_ids
tm = ig.compute(visual, ids, target_pos=3, target_token_id=int(ids[0, 4]))
print(f"  IG heatmap{tm.heatmap.shape} rango[{tm.heatmap.min():.2f},{tm.heatmap.max():.2f}] OK")
print("\n✓ xai.py funciona con la interfaz real del modelo.")
