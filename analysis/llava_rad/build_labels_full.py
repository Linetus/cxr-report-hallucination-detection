import json
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
LABX=json.load(open(SC+r"\labels_expanded.json"))
recs=[json.loads(l) for l in open(SC+r"\allsignals_openi.jsonl",encoding="utf-8")]
SUB={"CXR1174_IM-0118-1001","CXR1174_IM-0118-2001","CXR1187_IM-0126-2001","CXR1193_IM-0129-2001",
"CXR1197_IM-0131-1002","CXR1202_IM-0136-2001","CXR1226_IM-0150-1001","CXR1226_IM-0150-1002",
"CXR1229_IM-0152-2001","CXR1230_IM-0154-0001","CXR1230_IM-0154-1001","CXR1230_IM-0154-2001",
"CXR1233_IM-0157-1001","CXR1237_IM-0159-1002","CXR1245_IM-0167-1001","CXR1250_IM-0169-2001",
"CXR1255_IM-0172-1001-0001","CXR1255_IM-0172-1001-0002","CXR1270_IM-0181-2001"}
import re
judged=set()
for ln in open(SC+r"\remaining_pairs.txt",encoding="utf-8"):
    m=re.match(r"### (\S+)",ln.strip())
    if m: judged.add(m.group(1))   # solo los 224 que el subagente evaluó
full=dict(LABX)
for r in recs:
    i=r["id"]
    if i in LABX or i not in judged: continue
    full[i]=1 if i in SUB else 0
json.dump(full,open(SC+r"\labels_full.json","w"))
print(f"expanded={len(LABX)} full={len(full)} positivos={sum(full.values())}")
falt=SUB-set(full); print("no encontrados:",falt) if falt else print("todos los ids del subagente OK")
