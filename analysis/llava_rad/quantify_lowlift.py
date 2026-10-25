"""Cuantifica patrones de los low-lift: distracción por bordes y cajas por debajo de la
   resolución del mapa 37x37 (1 celda = (1/37)^2 ≈ 0.00073 de la imagen)."""
import json,csv,math,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding="utf-8")
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
NIH=Path(r"./data/raw/nih")
R=json.load(open(SC+r"\nih_results.json")); G=37
bbox={}
with open(NIH/"BBox_List_2017.csv") as f:
    rr=csv.reader(f); next(rr)
    for row in rr:
        if len(row)>=6 and row[0]: bbox[(row[0],row[1])]=[float(x) for x in row[2:6]]
# máscara de borde (anillo exterior de 3 celdas)
border=np.zeros((G,G),bool); border[:3,:]=True; border[-3:,:]=True; border[:,:3]=True; border[:,-3:]=True
border[:2,:2]=False  # excluir sink
def edgefrac(hm):
    h=np.array(hm,float).reshape(G,G); h=np.clip(h,0,None); tot=h.sum()+1e-9
    return float(h[border].sum()/tot)
for r in R:
    r["edge"]=edgefrac(r["hm"])
    bb=bbox.get((r["img"],r["label"])); r["area"]=(bb[2]*bb[3])/(1024*1024) if bb else 0
lift=np.array([r["lift"] for r in R]); det=np.array([r["detected"] for r in R])
edge=np.array([r["edge"] for r in R]); area=np.array([r["area"] for r in R])
cell=(1/G)**2
print(f"1 celda del mapa = {cell:.5f} de la imagen (≈{cell*100:.2f}%)")
low=lift<1.5; high=lift>=3
print(f"\n=== DISTRACCIÓN POR BORDES (fracción de atención en el anillo exterior) ===")
print(f"low-lift={edge[low].mean():.2f}  high-lift={edge[high].mean():.2f}   | FALLO={edge[det==0].mean():.2f} ACIERTO={edge[det==1].mean():.2f}")
print(f"Spearman-like AUC(edge -> FALLO) = {max(roc_auc_score(1-det,edge),1-roc_auc_score(1-det,edge)):.3f}")
print(f"\n=== TAMAÑO DE CAJA vs RESOLUCIÓN ===")
for thr in [3*cell, 8*cell, 20*cell]:
    m=area<thr
    print(f"cajas < {thr/cell:.0f} celdas ({thr:.4f}): n={m.sum()}  detección={det[m].mean():.2f}  lift_medio={lift[m].mean():.2f}")
big=area>0.10
print(f"cajas MUY grandes (>10% img): n={big.sum()}  detección={det[big].mean():.2f}  lift_medio={lift[big].mean():.2f}  (mayoría cardiomegalia)")
print(f"\ncorrelación área↔lift (todas): AUC(area pequeña -> lift bajo) contexto: lift_medio caja pequeña vs grande arriba.")
