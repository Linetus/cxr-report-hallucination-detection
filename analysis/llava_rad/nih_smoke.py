"""Smoke test: ¿puedo listar y extraer 1 imagen bbox de images_001.zip remoto?"""
import csv, time
from remotezip import RemoteZip

REPO = "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/images"
url = f"{REPO}/images_001.zip"

# set de las 880 imágenes con bbox
bbox = set()
with open(r"./data/raw/nih/BBox_List_2017.csv") as f:
    r = csv.reader(f); next(r)
    for row in r:
        if row and row[0]: bbox.add(row[0].strip())
print(f"bbox filenames: {len(bbox)}")

t0 = time.time()
with RemoteZip(url) as z:
    names = z.namelist()
    print(f"images_001.zip: {len(names)} archivos, ej: {names[:2]}")
    # ¿cuáles de este zip están en bbox?
    here = [n for n in names if n.split("/")[-1] in bbox]
    print(f"bbox en images_001: {len(here)}")
    if here:
        info = z.getinfo(here[0])
        print(f"extrayendo {here[0]} (comprimido {info.compress_size/1024:.0f} KB)...")
        data = z.read(here[0])
        print(f"leído {len(data)/1024:.0f} KB en {time.time()-t0:.1f}s total")
