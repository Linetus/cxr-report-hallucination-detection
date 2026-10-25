import json,re,sys
sys.stdout.reconfigure(encoding="utf-8")
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
LAB=json.load(open(SC+r"\labels_full.json"))
recs=[json.loads(l) for l in open(SC+r"\allsignals_openi.jsonl",encoding="utf-8")]
def clean(s): return re.sub(r"\s+"," ",s).strip()
rem=[r for r in recs if r["id"] not in LAB]
out=open(SC+r"\remaining2.txt","w",encoding="utf-8")
out.write(f"# {len(rem)} pares Open-I por etiquetar (extraídos={len(recs)}, etiquetados={len(LAB)})\n\n")
for r in rem:
    out.write(f"### {r['id']}\nREF: {clean(r.get('ref',''))[:320]}\nGEN: {clean(r['gen'])[:320]}\n\n")
out.close()
print(f"extraidos={len(recs)} etiquetados={len(LAB)} PENDIENTES={len(rem)} -> remaining2.txt")
