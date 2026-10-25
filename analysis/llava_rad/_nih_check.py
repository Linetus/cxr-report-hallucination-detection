"""¿peak_conc predice el anclaje CONTINUO (no solo el argmax)? Robustez a la etiqueta."""
import os,sys,json,math,numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.stdout.reconfigure(encoding="utf-8")
from scipy import ndimage
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
def spear(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if np.ptp(a)==0 or np.ptp(b)==0:return 0.0
    ra=rankdata(a);rb=rankdata(b);ra=ra-ra.mean();rb=rb-rb.mean()
    d=math.sqrt((ra*ra).sum()*(rb*rb).sum());return float((ra*rb).sum()/d) if d>0 else 0.0
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
R=[r for r in json.load(open(SC+r"\nih_results.json")) if r["detected"]]
G=37; SINK=np.ones((G,G),bool); SINK[:2,:2]=False; ys,xs=np.mgrid[0:G,0:G]
def peakconc(hm):
    H=np.array(hm,float).reshape(G,G);H=np.where(SINK,H,0.0)
    py,px=np.unravel_index(H.argmax(),H.shape);d=np.sqrt((ys-py)**2+(xs-px)**2)
    return float(H[d<=3].sum()/(H.sum()+1e-12))
pc=np.array([peakconc(r["hm"]) for r in R])
ein=np.array([r["energy_in_bbox"] for r in R]); lift=np.array([r["lift"] for r in R])
point=np.array([r["pointing"] for r in R],int)
print(f"n={len(R)}")
print(f"Spearman(peak_conc, energy-in-bbox continuo) = {spear(pc,ein):+.3f}")
print(f"Spearman(peak_conc, lift continuo)           = {spear(pc,lift):+.3f}")
print(f"AUC peak_conc -> lift<1 (mal anclado x energía) = {max(roc_auc_score((lift<1).astype(int),pc),1-roc_auc_score((lift<1).astype(int),pc)):.3f}  (n_mal={int((lift<1).sum())})")
print(f"AUC peak_conc -> pointing fuera (argmax)        = {max(roc_auc_score(1-point,pc),1-roc_auc_score(1-point,pc)):.3f}")
# ¿solapa la señal entre ambas etiquetas? correlación pointing vs lift
print(f"acuerdo pointing vs lift>1: {np.mean(point==(lift>1)):.2f}")
