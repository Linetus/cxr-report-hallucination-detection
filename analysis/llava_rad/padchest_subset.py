import json,os,sys,glob
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
P=r"./data/raw/padchest_gr"
imgs={os.path.basename(f) for f in glob.glob(P+r"\extract\PadChest_GR\*.png")}
print("imágenes extraídas:",len(imgs))
ann=json.load(open(P+r"\grounded_reports.json",encoding="utf-8"))
byimg={a["ImageID"]:a for a in ann}
sub=[byimg[i] for i in imgs if i in byimg]
print("con anotación:",len(sub))
# ¿estructura present/absent? contar abnormal True/False
abn=Counter(); normfree=0; labelc=Counter()
for a in sub:
    fs=a["findings"]
    if not any(f["abnormal"] for f in fs): normfree+=1
    for f in fs:
        abn[f["abnormal"]]+=1
        if f["abnormal"]:
            for l in f.get("labels",[]): labelc[l]+=1
print("findings abnormal True/False:",dict(abn))
print("estudios SIN hallazgo abnormal (normales):",normfree)
print("\ntop 20 labels (present):")
for l,c in labelc.most_common(20): print(f"  {c:4d}  {l}")
# guardar índice del subset
json.dump([{"ImageID":a["ImageID"],"normal":not any(f["abnormal"] for f in a["findings"]),
    "present_labels":sorted({l for f in a["findings"] if f["abnormal"] for l in f.get("labels",[])}),
    "report_en":" ".join(f["sentence_en"] for f in a["findings"])} for a in sub],
    open(P+r"\subset_index.json","w"))
print("\nsubset_index.json guardado (",len(sub),"estudios )")
