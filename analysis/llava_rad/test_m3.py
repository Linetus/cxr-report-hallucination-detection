"""Smoke test M3 Stage 1: KD texto (CXR-BERT) + KD imagen (RAD-DINO) simultáneos."""
import sys
sys.path.insert(0, r"./")
import torch
from src.models.model import RadiologyReportModel

torch.manual_seed(0)
B, L = 4, 16
m = RadiologyReportModel(encoder_pretrained=False, decoder_pretrained=False,
                         text_tower="cxrbert_kd", cxrbert_dim=128)
img = torch.randn(B, 3, 224, 224)
ids = torch.randint(0, 50257, (B, L))
mask = torch.ones(B, L, dtype=torch.long)
cxr = torch.randn(B, 128)      # teacher texto
rad = torch.randn(B, 768)      # teacher imagen (CLS RAD-DINO)

out = m.compute_contrastive_loss(img, ids, mask,
    cxrbert_emb=cxr, kd_weight=1.0, raddino_emb=rad, img_kd_weight=0.5)
print(f"loss={out['loss'].item():.4f}  infonce={out['infonce'].item():.4f}  "
      f"kd_texto={out['kd'].item():.4f}  kd_imagen={out['img_kd'].item():.4f}")
assert out["kd"].item() > 0 and out["img_kd"].item() > 0, "algún KD es 0"
assert torch.isfinite(out["loss"]), "loss no finita"

out["loss"].backward()
def has_grad(mod): return any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters())
print("grad encoder (KD imagen):", has_grad(m.encoder),
      "| text_proj:", has_grad(m.text_proj), "| kd_head (KD texto):", has_grad(m.kd_head))
assert has_grad(m.encoder), "el KD de imagen no propaga al encoder!"
print("\n✓ M3 Stage 1 (doble KD) funciona.")
