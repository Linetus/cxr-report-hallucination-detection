import json
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
LAB=json.load(open(SC+r"\labels.json"))
recs=[json.loads(l) for l in open(SC+r"\allsignals_openi.jsonl",encoding="utf-8")]
BOUND="CXR1162_IM-0108-2001"   # último par juzgado (bloques 1-2)
ERR={"CXR1058_IM-0041-2001","CXR1059_IM-0041-1001","CXR1064_IM-0045-1001",
"CXR1072_IM-0052-1001-0001","CXR1072_IM-0052-1001-0002","CXR1074_IM-0054-2001",
"CXR1076_IM-0054-3001","CXR1094_IM-0065-2001","CXR1104_IM-0071-1001","CXR1104_IM-0071-3001",
"CXR111_IM-0076-1002","CXR1110_IM-0076-2001","CXR1111_IM-0077-1001","CXR1111_IM-0077-4004",
"CXR1112_IM-0078-2001","CXR1117_IM-0079-2001","CXR1125_IM-0082-2001","CXR1128_IM-0084-6001",
"CXR1133_IM-0090-1002","CXR1140_IM-0096-1002","CXR1130_IM-0087-1001","CXR1130_IM-0087-3003",
"CXR1152_IM-0103-2001","CXR1157_IM-0106-2001","CXR116_IM-0107-2001"}
exp=dict(LAB)  # arranca con las 120 originales
judging=True
for r in recs:
    i=r["id"]
    if i in LAB: continue          # ya etiquetado
    if not judging: continue
    exp[i]=1 if i in ERR else 0
    if i==BOUND: judging=False      # dejar de etiquetar tras el último juzgado
json.dump(exp,open(SC+r"\labels_expanded.json","w"))
print(f"etiquetados original={len(LAB)}  ampliado={len(exp)}  positivos={sum(exp.values())}")
faltan=ERR-set(exp)
if faltan: print("WARN error-ids no encontrados:",faltan)
