"""Construye los PARES para juicio G-Eval de MAIRA-2: informe generado vs referencia objetiva
   de PadChest (informe del radiólogo + hallazgos PRESENTES y AUSENTES). NO etiqueta por regex;
   solo prepara la evidencia para que el juez (Claude) decida CORRECTO/ALUCINACIÓN/MISS."""
import json,sys
sys.stdout.reconfigure(encoding="utf-8")
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
byimg={a["ImageID"]:a for a in json.load(open(r"./data/raw/padchest_gr/grounded_reports.json",encoding="utf-8"))}
recs=[json.loads(l) for l in open(SC+r"\maira2_padchest.jsonl",encoding="utf-8")]
pairs=[]
for r in recs:
    st=byimg.get(r["ImageID"]);
    if st is None: continue
    present=[f["sentence_en"] for f in st["findings"] if f["abnormal"]]
    absent =[f["sentence_en"] for f in st["findings"] if not f["abnormal"]]
    pairs.append({"id":r["ImageID"],"normal":not present,
        "maira2_report":r["gen"],
        "ref_present":present,"ref_absent":absent,
        "min_prob_content":round(sorted(r["probs"])[0] if len(r["probs"])<=3 else min(r["probs"][1:]),4)})
json.dump(pairs,open(SC+r"\geval_pairs.json","w"),ensure_ascii=False,indent=1)
print(f"{len(pairs)} pares -> geval_pairs.json")
print("\n--- muestra ---")
for p in pairs[:3]:
    print(f"\nID {p['id']} (normal={p['normal']})")
    print(f" MAIRA-2: {p['maira2_report'][:200]}")
    print(f" REF presentes: {p['ref_present']}")
    print(f" REF ausentes: {p['ref_absent'][:3]}")
