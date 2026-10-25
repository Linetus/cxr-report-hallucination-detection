import json, re
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
M=json.load(open(SC+r"\pertoken_meta.json"))
def clean(s): return re.sub(r"\s+"," ",s).strip()
for r in M:
    print(f"### {r['id']}")
    print("REF:", clean(r['ref'])[:320])
    print("GEN:", clean(r['gen'])[:320])
    print()
