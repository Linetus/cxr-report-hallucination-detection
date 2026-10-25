"""Analiza la batería: imprime pares para G-Eval + montaje de heatmaps para juzgar."""
import json, glob, numpy as np
from pathlib import Path
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

B = json.load(open(r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad/battery.json"))
IMG = Path(r"./data/raw/open-i")
def cls(r):
    if r["ref_ab"] and not r["gen_ab"]: return "FALSO-NORMAL"
    if not r["ref_ab"] and r["gen_ab"]: return "FALSO-POSITIVO"
    return "ACIERTO"

# 1) pares para G-Eval (mezcla de clases)
print("=== PARES real-vs-generado (muestra para G-Eval) ===")
sample = []
for c in ["ACIERTO","FALSO-NORMAL","FALSO-POSITIVO"]:
    sample += [r for r in B if cls(r)==c][:6]
for r in sample:
    print(f"\n#{r['i']} [{cls(r)}] ent={r['ent']:.3f}")
    print(f"  REAL: {r['ref'][:180]}")
    print(f"  GEN : {r['gen'][:180]}")

# 2) montaje de heatmaps (3 por clase)
picks = []
for c in ["ACIERTO","FALSO-NORMAL","FALSO-POSITIVO"]:
    picks += [r for r in B if cls(r)==c][:3]
fig, axes = plt.subplots(3, 3, figsize=(11, 11))
for ax, r in zip(axes.flat, picks):
    p = IMG / f"{r['id']}.png"
    disp = np.array(Image.open(p).convert("L").resize((222,222)))
    hm = np.array(r["hm"]).reshape(37,37)
    up = np.kron(hm, np.ones((6,6)))[:222,:222]
    ax.imshow(disp, cmap="gray"); ax.imshow(up, cmap="jet", alpha=0.5)
    ax.set_title(f"{cls(r)} ent={r['ent']:.2f}\n{r['gen'][:38]}", fontsize=7); ax.axis("off")
plt.tight_layout()
OUT = r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad/battery_montage.png"
plt.savefig(OUT, dpi=85, bbox_inches="tight"); print("\nMontaje:", OUT)
