import json,re,sys
sys.stdout.reconfigure(encoding="utf-8")
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
LAB=json.load(open(SC+r"\labels.json"))
recs=[json.loads(l) for l in open(SC+r"\allsignals_openi.jsonl",encoding="utf-8")]
def clean(s): return re.sub(r"\s+"," ",s).strip()
new=[r for r in recs if r["id"] not in LAB]
print(f"total extraídos={len(recs)}  ya etiquetados={sum(1 for r in recs if r['id'] in LAB)}  NUEVOS a juzgar={len(new)}\n")
for r in new:
    print(f"### {r['id']}")
    print("REF:", clean(r.get('ref',''))[:300])
    print("GEN:", clean(r['gen'])[:300])
    print()
