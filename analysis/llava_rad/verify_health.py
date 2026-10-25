import sys, zipfile
sys.path.insert(0, r"./")
base = r"./outputs/checkpoints"
kept = [
    rf"{base}\stage1_epoch=07_val_loss=0.0000-v1.ckpt",
    rf"{base}\baseline_v3\stage2_epoch=01_val_rouge_l=0.0000.ckpt",
    rf"{base}\m0_hilr\stage2_epoch=02_val_rouge_l=0.0000.ckpt",
    rf"{base}\cxrbert_kd\stage1_epoch=10_val_loss=0.0000.ckpt",
]
print("=== 2) Integridad de checkpoints (índice zip) ===")
for c in kept:
    try:
        z = zipfile.ZipFile(c)
        bad = z.testzip()  # None si todos los CRC OK
        name = c.split("checkpoints\\")[-1]
        print(f"  {'OK ' if bad is None else 'CRC-FAIL':4s} {name}  ({len(z.namelist())} entradas)")
    except Exception as e:
        print(f"  CORRUPTO {c}: {e}")

print("\n=== 3) Import del toolkit ===")
from src.modules import (ReportHeatmapAligner, CrossAttentionExtractor,
    IntegratedGradientsVisualizer, evaluate_localization, evaluate_faithfulness)
print("  src.modules importa OK:", ReportHeatmapAligner.__name__,
      "·", evaluate_localization.__name__, "·", evaluate_faithfulness.__name__)
print("\n✓ Comprobación funcional completa.")
