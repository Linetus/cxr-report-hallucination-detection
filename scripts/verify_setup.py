"""
Verifica que el entorno tiene todas las dependencias necesarias
y que la GPU está correctamente configurada para entrenar.

Uso:
    python scripts/verify_setup.py
"""
import sys
from pathlib import Path

# Añadir raíz del proyecto al path para importar src.*
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = "  [OK]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"

errors = []
warnings = []

def check(label, fn):
    try:
        result = fn()
        print(f"{PASS} {label}" + (f" — {result}" if result else ""))
    except Exception as e:
        print(f"{FAIL} {label} — {e}")
        errors.append(label)

def warn(label, fn):
    try:
        result = fn()
        print(f"{PASS} {label}" + (f" — {result}" if result else ""))
    except Exception as e:
        print(f"{WARN} {label} — {e}")
        warnings.append(label)

print("=" * 55)
print("  TFM Radiology — Verificación del entorno")
print("=" * 55)

# ── Core ML ──────────────────────────────────────────────
print("\n[Core ML]")
check("torch",            lambda: __import__("torch").__version__)
check("torchvision",      lambda: __import__("torchvision").__version__)
check("pytorch_lightning",lambda: __import__("pytorch_lightning").__version__)

# ── CUDA ─────────────────────────────────────────────────
print("\n[CUDA / GPU]")
def check_cuda():
    import torch
    assert torch.cuda.is_available(), "CUDA no disponible"
    name  = torch.cuda.get_device_name(0)
    vram  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{name} — {vram:.1f} GB VRAM"

def check_fp16():
    import torch
    x = torch.ones(4, 4, dtype=torch.float16, device="cuda")
    y = x @ x
    return "float16 matmul OK"

check("CUDA disponible + GPU",  check_cuda)
check("Float16 (Tensor Cores)", check_fp16)

# ── Hugging Face ──────────────────────────────────────────
print("\n[Hugging Face]")
check("transformers",     lambda: __import__("transformers").__version__)
check("datasets",         lambda: __import__("datasets").__version__)
check("evaluate",         lambda: __import__("evaluate").__version__)
check("tokenizers",       lambda: __import__("tokenizers").__version__)
check("timm (ViT)",       lambda: __import__("timm").__version__)

# ── XAI ──────────────────────────────────────────────────
print("\n[XAI]")
check("captum",           lambda: __import__("captum").__version__)
warn("grad-cam",          lambda: __import__("pytorch_grad_cam") and "OK")

# ── Image / Preprocessing ─────────────────────────────────
print("\n[Imagen & Preprocesamiento]")
check("albumentations",   lambda: __import__("albumentations").__version__)
check("cv2 (OpenCV)",     lambda: __import__("cv2").__version__)
check("PIL (Pillow)",     lambda: __import__("PIL").__version__)

# ── NLP Metrics ───────────────────────────────────────────
print("\n[Métricas NLP]")
check("nltk",             lambda: __import__("nltk").__version__)
check("sacrebleu",        lambda: __import__("sacrebleu").__version__)
warn("rouge_score",       lambda: __import__("rouge_score") and "OK")

# ── Config / Utils ────────────────────────────────────────
print("\n[Config & Utilidades]")
check("omegaconf",        lambda: __import__("omegaconf").__version__)
check("hydra",            lambda: __import__("hydra").__version__)
check("tensorboard",      lambda: __import__("tensorboard").__version__)
check("pandas",           lambda: __import__("pandas").__version__)
check("numpy",            lambda: __import__("numpy").__version__)
check("scipy",            lambda: __import__("scipy").__version__)
check("sklearn",          lambda: __import__("sklearn").__version__)
check("tqdm",             lambda: __import__("tqdm").__version__)

# ── Proyecto ──────────────────────────────────────────────
print("\n[Módulos del proyecto]")

def check_encoder():
    from src.models.encoder import VisualEncoder
    enc = VisualEncoder(pretrained=False)
    import torch
    x = torch.randn(1, 3, 224, 224)
    out = enc(x)
    patches = out["patch_features"].shape
    assert patches == (1, 196, 768), f"shape inesperado: {patches}"
    return f"patch_features {patches}"

def check_decoder():
    from src.models.decoder import ReportDecoder
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    dec = ReportDecoder(pretrained=True)
    return f"hidden_dim={dec.hidden_dim}, cross_attn_layers={len(dec.cross_attention_layers)}"

def check_model():
    from src.models.model import RadiologyReportModel
    m = RadiologyReportModel(encoder_pretrained=False, decoder_pretrained=True)
    import torch
    imgs = torch.randn(2, 3, 224, 224)
    feats = m.encode_image(imgs)
    assert feats.shape == (2, 196, 768)
    return f"encode_image OK — {feats.shape}"

def check_data():
    from src.data.preprocessing import CXRPreprocessor, ReportPreprocessor
    import numpy as np
    proc = CXRPreprocessor(image_size=224, mode="eval")
    dummy = (np.random.rand(512, 512) * 255).astype("uint8")
    t = proc(dummy)
    assert t.shape == (3, 224, 224)
    return f"tensor shape {tuple(t.shape)}"

check("src.models.encoder (ViT forward)", check_encoder)
check("src.models.decoder (GPT-2 + cross-attn)", check_decoder)
check("src.models.model (encode_image)", check_model)
check("src.data.preprocessing (CXRPreprocessor)", check_data)

# ── Resumen ───────────────────────────────────────────────
print("\n" + "=" * 55)
if not errors:
    print("  ✓ Todo OK — listo para entrenar")
    print("  Siguiente paso:")
    print("    python scripts/train.py stage=1")
else:
    print(f"  ✗ {len(errors)} error(es) crítico(s):")
    for e in errors:
        print(f"    - {e}")
    print("\n  Solución rápida:")
    print("    pip install -r requirements.txt --upgrade")
if warnings:
    print(f"\n  {len(warnings)} advertencia(s) no crítica(s):")
    for w in warnings:
        print(f"    ~ {w}")
print("=" * 55)

sys.exit(1 if errors else 0)
