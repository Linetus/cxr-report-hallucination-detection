"""
Precomputa y cachea el embedding CLS de RAD-DINO para cada imagen Open-I, para
habilitar el KD de IMAGEN del Stage 1 de M3 (ViT student → RAD-DINO teacher).

Análogo a precompute_cxrbert.py pero para imagen. Teacher congelado, sobre imagen
LIMPIA (eval, 518px, normalización RAD-DINO). Cachea por image_id.

Uso:
    python scripts/precompute_raddino.py               # todas las imágenes
    python scripts/precompute_raddino.py --limit 8     # smoke test
    python scripts/precompute_raddino.py --device cuda # si hay GPU libre (mucho más rápido)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf

from src.data.open_i import build_open_i_metadata
from src.data.preprocessing import CXRPreprocessor
from src.models.encoder import RadDinoEncoder


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = OmegaConf.load("configs/base_config.yaml")
    out_dir = Path("outputs/cache"); out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.limit else "cls"
    out_path = out_dir / f"raddino_{suffix}.pt"
    if out_path.exists() and not args.force:
        print(f"Caché ya existe: {out_path}. --force para recomputar.")
        return

    df = build_open_i_metadata(cfg.dataset.root)
    # imágenes únicas presentes en disco
    seen, items = set(), []
    for _, row in df.iterrows():
        iid = row["image_id"]
        if iid not in seen and Path(row["image_path"]).exists():
            seen.add(iid); items.append((iid, row["image_path"]))
    if args.limit:
        items = items[:args.limit]
    print(f"Imágenes a embeber: {len(items)}  (device={args.device})")

    pre = CXRPreprocessor(image_size=518, mode="eval",
                          mean=CXRPreprocessor.RADDINO_MEAN, std=CXRPreprocessor.RADDINO_STD)
    enc = RadDinoEncoder(freeze=True).eval().to(args.device)
    torch.set_num_threads(4)

    id2emb = {}
    batch, ids = [], []
    def flush():
        if not batch: return
        x = torch.stack(batch).to(args.device)
        cls = enc(x)["cls_embedding"].cpu()
        for j, iid in enumerate(ids):
            id2emb[iid] = cls[j].clone()
        batch.clear(); ids.clear()
    for k, (iid, path) in enumerate(items):
        batch.append(pre(CXRPreprocessor.load_image(path))); ids.append(iid)
        if len(batch) >= args.batch_size:
            flush(); print(f"  {k+1}/{len(items)}", flush=True)
    flush()

    dim = next(iter(id2emb.values())).shape[0]
    torch.save({"teacher": "microsoft/rad-dino", "dim": int(dim), "id2emb": id2emb}, out_path)
    print(f"\n✓ Guardado {out_path} — {len(id2emb)} imágenes → CLS {dim}D (para KD de imagen M3)")


if __name__ == "__main__":
    main()
