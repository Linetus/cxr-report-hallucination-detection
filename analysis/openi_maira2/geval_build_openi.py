"""Construye pares G-Eval para OOD-MAIRA (Open-I): informe MAIRA vs referencia del radiólogo
   (FINDINGS+IMPRESSION del XML) + diagnóstico MeSH oficial. NO etiqueta por regex."""
import json,glob,sys
import xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding="utf-8")
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
XD=r"./data/raw/open-i/ecgen-radiology"
img2ref={}
for f in glob.glob(XD+r"\*.xml"):
    try: root=ET.parse(f).getroot()
    except: continue
    majors=[m.text for m in root.findall(".//MeSH/major") if m.text]
    txt={a.get("Label"):(a.text or "") for a in root.findall(".//AbstractText")}
    ref=" ".join(filter(None,[txt.get("FINDINGS",""),txt.get("IMPRESSION","")])).strip()
    for pi in root.findall(".//parentImage"):
        if pi.get("id"): img2ref[pi.get("id")]=(majors,ref)
recs=[json.loads(l) for l in open(SC+r"\maira2_openi.jsonl",encoding="utf-8")]
pairs=[]
for r in recs:
    iid=r["ImageID"][:-4] if r["ImageID"].endswith(".png") else r["ImageID"]
    g=img2ref.get(iid)
    if g is None: continue
    majors,ref=g
    normal=(len(majors)==0) or all((m or "").strip().lower() in ("normal","no indexing") for m in majors)
    pairs.append({"id":r["ImageID"],"normal":normal,"maira2_report":r["gen"],
        "ref_mesh_findings":majors,"ref_report":ref[:500]})
json.dump(pairs,open(SC+r"\geval_pairs_openi.json","w"),ensure_ascii=False,indent=1)
print(f"{len(pairs)} pares OOD -> geval_pairs_openi.json")
for p in pairs[:3]:
    print(f"\n[{'normal' if p['normal'] else 'anormal'}] MeSH:{p['ref_mesh_findings']}")
    print(f" MAIRA: {p['maira2_report'][:150]}")
    print(f" REF:   {p['ref_report'][:120]}")
