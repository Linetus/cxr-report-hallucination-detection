"""Test de integración CPU de los 3 brazos de torre de texto (Stage 1)."""
import sys
from pathlib import Path
sys.path.insert(0, r"./")

import torch
from src.models.model import RadiologyReportModel

torch.manual_seed(0)
B, L, VOCAB, CXR_DIM = 4, 32, 50257, 128
img = torch.randn(B, 3, 224, 224)
ids = torch.randint(0, VOCAB, (B, L))
mask = torch.ones(B, L, dtype=torch.long)
cxr = torch.randn(B, CXR_DIM)

for arm in ["baseline", "cxrbert_tower", "cxrbert_kd"]:
    print(f"\n=== Brazo: {arm} ===")
    m = RadiologyReportModel(
        encoder_pretrained=False, decoder_pretrained=False,
        text_tower=arm, cxrbert_dim=CXR_DIM,
    )
    # ¿Se construyeron las cabezas correctas?
    print("  tiene cxrbert_proj:", hasattr(m, "cxrbert_proj"),
          "| tiene kd_head:", hasattr(m, "kd_head"))
    emb = None if arm == "baseline" else cxr
    out = m.compute_contrastive_loss(img, ids, mask, cxrbert_emb=emb, kd_weight=0.5)
    print(f"  loss={out['loss'].item():.4f}  infonce={out['infonce'].item():.4f}  kd={out['kd'].item():.4f}")
    assert torch.isfinite(out["loss"]), "loss no finita!"

    out["loss"].backward()
    # ¿Qué cabezas reciben gradiente?
    def has_grad(mod):
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters())
    print("  grad image_proj:", has_grad(m.image_proj),
          "| text_proj:", has_grad(m.text_proj))
    if arm == "cxrbert_tower":
        print("  grad cxrbert_proj:", has_grad(m.cxrbert_proj))
    if arm == "cxrbert_kd":
        print("  grad kd_head:", has_grad(m.kd_head))

print("\n✓ Todos los brazos OK.")
