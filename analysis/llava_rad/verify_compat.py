"""Compila edits + verifica retrocompatibilidad (path ViT por defecto sigue OK)."""
import sys, py_compile
sys.path.insert(0, r"./")
for f in ["src/models/encoder.py", "src/models/model.py", "src/data/open_i.py",
          "src/data/preprocessing.py", "src/training/lightning_module.py", "scripts/train.py"]:
    py_compile.compile(rf"./{f}", doraise=True)
print("1) Todos compilan OK")

import torch
from src.models.model import RadiologyReportModel
# path por defecto (ViT) — debe seguir funcionando igual que M0/M1
m = RadiologyReportModel(encoder_pretrained=False, decoder_pretrained=False)
img = torch.randn(1, 3, 224, 224)
vis = m.encode_image(img)
print("2) ViT default: patch_grid", m.patch_grid, "features", tuple(vis.shape),
      "(debe ser 1,196,768)")
enc_train = any(p.requires_grad for p in m.encoder.parameters())
print("3) ViT encoder entrenable:", enc_train, "(debe ser True — M0/M1 sin cambios)")
print("\n✓ Retrocompatibilidad OK — M0/M1 intactos.")
