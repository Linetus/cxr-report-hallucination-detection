"""
Precomputa y cachea los embeddings de texto de CXR-BERT-specialized para los
informes de Open-I. Habilita los brazos experimentales B (torre de texto) y
C (destilación KD) del Stage 1 SIN necesidad de imágenes ni de la GPU.

    Teacher: microsoft/BiomedVLP-CXR-BERT-specialized
    - Text encoder radiológico entrenado CLIP-style (proyección vision-aligned, 128D)
    - Licencia research-only (Boecking et al. 2022, BioViL). Citar en la memoria.

Se ejecuta en CPU a propósito: la GPU está reservada para el entrenamiento en curso.

Uso:
    python scripts/precompute_cxrbert.py                # cachea todos los informes
    python scripts/precompute_cxrbert.py --limit 8      # smoke test (8 textos)
    python scripts/precompute_cxrbert.py --force        # recomputa aunque exista caché
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf

from src.data.open_i import build_open_i_metadata
from src.data.preprocessing import ReportPreprocessor

TEACHER = "microsoft/BiomedVLP-CXR-BERT-specialized"


def clean_reports(cfg, limit=None):
    """Devuelve la lista de textos limpios ÚNICOS, igual que los ve el dataset."""
    df = build_open_i_metadata(cfg.dataset.root)
    text_target = cfg.dataset.get("text_target", "report")
    pre = ReportPreprocessor(max_length=cfg.preprocessing.max_text_length)

    seen, texts = set(), []
    for _, row in df.iterrows():
        raw = row[text_target] or row["report"]
        cleaned = pre.clean(raw)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            texts.append(cleaned)
    if limit:
        texts = texts[:limit]
    return texts, text_target


@torch.no_grad()
def embed_texts(texts, batch_size=32):
    """Embebe una lista de textos con CXR-BERT en CPU. Devuelve tensor (N, D)."""
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(4)  # deja CPU para los dataloaders del entrenamiento
    print(f"  Cargando teacher {TEACHER} (trust_remote_code, CPU)...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER, trust_remote_code=True)
    model = AutoModel.from_pretrained(TEACHER, trust_remote_code=True).eval()

    all_emb = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=chunk,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        emb = model.get_projected_text_embeddings(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
        )
        all_emb.append(emb.cpu())
        print(f"    {min(i + batch_size, len(texts))}/{len(texts)} informes embebidos", flush=True)
    return torch.cat(all_emb, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Nº de textos (smoke test)")
    ap.add_argument("--force", action="store_true", help="Recomputa aunque exista caché")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    cfg = OmegaConf.load("configs/base_config.yaml")
    text_target = cfg.dataset.get("text_target", "report")
    out_dir = Path("outputs/cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.limit else text_target
    out_path = out_dir / f"cxrbert_specialized_{suffix}.pt"

    if out_path.exists() and not args.force:
        cached = torch.load(out_path)
        print(f"Caché ya existe: {out_path} ({len(cached['text2emb'])} informes). "
              f"Usa --force para recomputar.")
        return

    print(f"{'='*60}\n  PRECOMPUTADO EMBEDDINGS CXR-BERT (brazos B/C)\n{'='*60}")
    texts, text_target = clean_reports(cfg, limit=args.limit)
    print(f"  Textos únicos a embeber: {len(texts)}  (text_target={text_target})")

    emb = embed_texts(texts, batch_size=args.batch_size)
    print(f"  Matriz de embeddings: {tuple(emb.shape)}  (dim proyección={emb.shape[1]})")

    text2emb = {t: emb[i].clone() for i, t in enumerate(texts)}
    torch.save(
        {
            "teacher": TEACHER,
            "text_target": text_target,
            "proj_dim": int(emb.shape[1]),
            "text2emb": text2emb,
        },
        out_path,
    )
    print(f"\n✓ Guardado: {out_path.resolve()}")
    print(f"  {len(text2emb)} informes → embedding {emb.shape[1]}D. "
          f"Lookup en el dataset por texto limpio.")


if __name__ == "__main__":
    main()
