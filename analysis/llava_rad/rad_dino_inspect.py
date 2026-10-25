"""Inspecciona RAD-DINO: preprocessing esperado + forma de salida."""
import torch
from transformers import AutoModel, AutoImageProcessor

M = "microsoft/rad-dino"
proc = AutoImageProcessor.from_pretrained(M)
print("=== Processor ===")
print("  image_mean:", getattr(proc, "image_mean", None))
print("  image_std :", getattr(proc, "image_std", None))
print("  size      :", getattr(proc, "size", None))
print("  crop_size :", getattr(proc, "crop_size", None))
print("  do_resize :", getattr(proc, "do_resize", None), "| do_normalize:", getattr(proc, "do_normalize", None))

model = AutoModel.from_pretrained(M).eval()
cfg = model.config
print("\n=== Config ===")
print("  hidden_size:", cfg.hidden_size)
print("  patch_size :", getattr(cfg, "patch_size", None))
print("  image_size :", getattr(cfg, "image_size", None))
print("  num_register_tokens:", getattr(cfg, "num_register_tokens", 0))

with torch.no_grad():
    out = model(pixel_values=torch.randn(1, 3, 518, 518))
hs = out.last_hidden_state
print("\n=== Salida (input 518x518) ===")
print("  last_hidden_state:", tuple(hs.shape))
n_patch_side = 518 // getattr(cfg, "patch_size", 14)
print(f"  patches esperados: {n_patch_side}x{n_patch_side} = {n_patch_side**2}")
print(f"  tokens totales - patches = {hs.shape[1] - n_patch_side**2} (=1 CLS + registers)")
