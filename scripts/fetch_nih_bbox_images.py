"""
Descarga SOLO las ~880 imágenes con bounding box de NIH ChestX-ray14, extrayéndolas
parcialmente de los 12 zips mirroreados en HuggingFace vía HTTP range requests
(remotezip) — sin bajar los 42 GB completos (~260 MB reales).

Uso: python scripts/fetch_nih_bbox_images.py
"""
import csv
import sys
from pathlib import Path

from remotezip import RemoteZip

REPO = "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/images"
BBOX_CSV = Path("data/raw/nih/BBox_List_2017.csv")
OUT = Path("data/raw/nih/images")
OUT.mkdir(parents=True, exist_ok=True)


def load_bbox_names() -> set:
    names = set()
    with open(BBOX_CSV) as f:
        r = csv.reader(f); next(r)
        for row in r:
            if row and row[0]:
                names.add(row[0].strip())
    return names


def main():
    bbox = load_bbox_names()
    print(f"Objetivo: {len(bbox)} imágenes con bbox")
    got = {p.name for p in OUT.glob("*.png")}
    print(f"Ya presentes: {len(got)}")

    total = len(got)
    for i in range(1, 13):
        url = f"{REPO}/images_{i:03d}.zip"
        try:
            with RemoteZip(url) as z:
                here = [n for n in z.namelist()
                        if n.split("/")[-1] in bbox and n.split("/")[-1] not in got]
                for n in here:
                    base = n.split("/")[-1]
                    (OUT / base).write_bytes(z.read(n))
                    got.add(base); total += 1
                print(f"  images_{i:03d}.zip → +{len(here)}  (acumulado {total}/{len(bbox)})",
                      flush=True)
        except Exception as e:
            print(f"  images_{i:03d}.zip ERROR: {e}", flush=True)

    faltan = bbox - {p.name for p in OUT.glob('*.png')}
    print(f"\n✓ Descargadas {total}/{len(bbox)}. Faltan: {len(faltan)}")
    if faltan:
        print("  (algunas pueden no estar en el mirror):", list(faltan)[:5])


if __name__ == "__main__":
    main()
