import json,sys
sys.stdout.reconfigure(encoding="utf-8")
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
R=json.load(open(SC+r"\nih_results.json"))
print("keys:",list(R[0].keys()))
det=[r for r in R if r["detected"]]
print("detectados:",len(det))
# bien anclados (pointing True, lift alto) por patología distinta
good=sorted([r for r in det if r["pointing"]], key=lambda r:-r["lift"])
bad=sorted([r for r in det if not r["pointing"]], key=lambda r:-r["energy_in_bbox"])
seen=set(); gsel=[]
for r in good:
    if r["label"] not in seen: gsel.append(r); seen.add(r["label"])
    if len(gsel)>=3: break
print("\nBIEN ANCLADOS:")
for r in gsel: print(f"  {r['img']} | {r['label']} lift={r['lift']:.1f} point={r['pointing']} ein={r['energy_in_bbox']:.2f}")
print("MAL ANCLADOS:")
for r in bad[:2]: print(f"  {r['img']} | {r['label']} lift={r['lift']:.1f} point={r['pointing']} ein={r['energy_in_bbox']:.2f}")
