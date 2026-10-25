import json,sys,csv
sys.stdout.reconfigure(encoding="utf-8")
P=r"./data/raw/padchest_gr"
d=json.load(open(P+r"\grounded_reports.json",encoding="utf-8"))
print("tipo:",type(d).__name__, "len:", len(d))
if isinstance(d,dict):
    k=list(d.keys())[:5]; print("keys top:",k)
    rec=d[k[0]]
else:
    rec=d[0]
print("\n=== 1 registro ===")
print(json.dumps(rec, ensure_ascii=False, indent=1)[:1800])
print("\n=== master_table.csv cabecera ===")
with open(P+r"\master_table.csv",encoding="utf-8") as f:
    r=csv.reader(f); h=next(r); print("cols:",h[:20]); print("fila0:",next(r)[:20])
