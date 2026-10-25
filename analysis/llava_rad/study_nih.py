"""Estudio NIH por patología: top/bottom/mediana de lift, aciertos (detección), y
   correlación lift ↔ acierto. Selecciona casos low-lift (correctos y fallidos) para
   análisis manual. Todo sobre datos ya recogidos (nih_results.json) — SIN GPU."""
import json,csv,math,sys
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding="utf-8")
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
NIH=Path(r"./data/raw/nih")
R=json.load(open(SC+r"\nih_results.json"))
bbox={}
with open(NIH/"BBox_List_2017.csv") as f:
    rr=csv.reader(f); next(rr)
    for row in rr:
        if len(row)>=6 and row[0]: bbox[(row[0],row[1])]=[float(x) for x in row[2:6]]
for r in R:
    bb=bbox.get((r["img"],r["label"]))
    r["bbox_area"]=round((bb[2]*bb[3])/(1024*1024),3) if bb else None
def spear(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if np.ptp(a)==0 or np.ptp(b)==0: return 0.0
    ra=rankdata(a);rb=rankdata(b);ra-=ra.mean();rb-=rb.mean()
    d=math.sqrt((ra*ra).sum()*(rb*rb).sum());return (ra*rb).sum()/d if d else 0.0

det=np.array([r["detected"] for r in R]); lift=np.array([r["lift"] for r in R])
labels=np.array([r["label"] for r in R])
out=[]
out.append(f"TOTAL casos bbox={len(R)}  detectados(aciertos)={int(det.sum())} ({det.mean():.0%})")
out.append(f"\n=== GLOBAL: ¿el lift (anclaje) acompaña al acierto? ===")
out.append(f"lift medio ACIERTO={lift[det==1].mean():.2f}  vs  FALLO={lift[det==0].mean():.2f}")
out.append(f"AUC(lift -> acierto) = {roc_auc_score(det,lift):.3f}   (>0.5 => más anclaje acompaña al acierto)")

by=defaultdict(list)
for r in R: by[r["label"]].append(r)
out.append("\n=== POR PATOLOGÍA ===")
out.append(f"{'patología':13}{'N':>4}{'det%':>6}{'liftMedia':>10}{'liftMed':>8}{'lift_ac':>8}{'lift_fa':>8}{'AUC':>6}")
xpath=[]
for lab in sorted(by):
    g=by[lab]; L=np.array([r["lift"] for r in g]); d=np.array([r["detected"] for r in g])
    la=L[d==1].mean() if (d==1).any() else float('nan'); lf=L[d==0].mean() if (d==0).any() else float('nan')
    auc=roc_auc_score(d,L) if 0<d.sum()<len(d) else float('nan')
    out.append(f"{lab:13}{len(g):>4}{d.mean():>6.2f}{L.mean():>10.2f}{np.median(L):>8.2f}{la:>8.2f}{lf:>8.2f}{auc:>6.2f}")
    xpath.append((lab,L.mean(),d.mean()))
ml=[x[1] for x in xpath]; dr=[x[2] for x in xpath]
out.append(f"\nCROSS-PATOLOGÍA: Spearman(lift_medio, detección) = {spear(ml,dr):+.2f}")
out.append("(negativo => las patologías que MÁS detecta NO son las que mejor ancla)")

# top5/bottom5/mediana por patología
sel_low=[]
out.append("\n=== TOP/BOTTOM/MEDIANA de lift por patología (img · lift · acierto · área_bbox) ===")
for lab in sorted(by):
    g=sorted(by[lab], key=lambda r:r["lift"])
    out.append(f"\n-- {lab} (N={len(g)}) --")
    def fmt(r): return f"{r['img']} · lift={r['lift']:.2f} · {'ACIERTO' if r['detected'] else 'FALLO'} · área={r['bbox_area']}"
    out.append("  BOTTOM5: "); [out.append("    "+fmt(r)) for r in g[:5]]
    mid=len(g)//2; out.append("  MEDIANA5: "); [out.append("    "+fmt(r)) for r in g[max(mid-2,0):mid+3]]
    out.append("  TOP5: "); [out.append("    "+fmt(r)) for r in g[-5:]]
    sel_low += g[:3]   # los 3 de menor lift por patología para análisis manual

# guardar selección low-lift para renderizar (mezcla acierto/fallo)
json.dump([{"img":r["img"],"label":r["label"],"lift":round(r["lift"],2),"detected":r["detected"],
            "area":r["bbox_area"],"gini":round(r["gini"],3),"entropy":round(r["entropy"],3)} for r in sel_low],
          open(SC+r"\lowlift_sel.json","w"))
txt="\n".join(out)
open(r"./outputs/estudio_nih_por_patologia.txt","w",encoding="utf-8").write(txt)
print(txt)
print(f"\n[low-lift seleccionados para figuras/análisis: {len(sel_low)}]")
