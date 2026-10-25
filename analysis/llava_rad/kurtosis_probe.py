"""¿La curtosis 'muestra mucho'? Media difiere pero AUC monótono=0.51. Investigo:
   distribución por clase, AUC monótono vs no-monótono (distancia a la mediana), y por terciles."""
import os,sys,json,math,numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
sys.stdout.reconfigure(encoding="utf-8")
from scipy.stats import kurtosis as kurt
from sklearn.metrics import roc_auc_score
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
META=json.load(open(SC+r"\pertoken_meta.json")); NPZ=np.load(SC+r"\pertoken_hm.npz")
LAB=json.load(open(SC+r"\labels.json")); G=37
items=[r for r in META if f"hm_{r['k']}" in NPZ.files and r["id"] in LAB]
rows=np.concatenate([NPZ[f"hm_{r['k']}"].astype(np.float64) for r in items],0)
pm=rows.mean(0); thr=pm.mean()+6*pm.std(); sink=np.where(pm>thr)[0]
nonsink=np.ones(G*G,bool); nonsink[sink]=False
def norm_ns(v):
    a=v[nonsink].astype(np.float64);a=a-a.min();s=a.sum();return a/s if s>0 else np.ones_like(a)/len(a)
mean_pat=norm_ns(rows.mean(0))
ids,K=[],[]
for r in items:
    hm=NPZ[f"hm_{r['k']}"].astype(np.float64)
    P=np.array([norm_ns(hm[i]) for i in range(len(r["tokens"]))])
    w=np.sqrt(((P-mean_pat)**2).sum(1)); w=w if w.sum()>0 else np.ones(len(P))
    agg=(w[:,None]*P).sum(0)/w.sum()
    ids.append(r["id"]); K.append(float(kurt(agg)))
K=np.array(K); y=np.array([LAB[i] for i in ids])

def rep(mask,name):
    v=K[mask]; q=np.percentile(v,[10,25,50,75,90])
    print(f"{name:10} n={mask.sum():3d}  mediana={np.median(v):7.1f}  IQR=[{q[1]:.1f},{q[3]:.1f}]  p10={q[0]:.1f} p90={q[4]:.1f}")
print("=== distribución curtosis (heatmap distinct-weighted, sink-excl) ===")
rep(y==0,"fiable"); rep(y==1,"alucina")
med=np.median(K)
print(f"\nAUC monótono            = {max(roc_auc_score(y,K),1-roc_auc_score(y,K)):.3f}")
print(f"AUC |curt - mediana|    = {roc_auc_score(y,np.abs(K-med)):.3f}   (no-monótono: extremos=problema)")
print(f"AUC -|curt - mediana|   = {roc_auc_score(y,-np.abs(K-med)):.3f}   (centro=problema)")
# tasa de alucinación por tercil
order=np.argsort(K); ter=np.array_split(order,3)
print("\ntasa alucinación por tercil de curtosis (bajo→alto):")
for i,idx in enumerate(ter):
    print(f"  T{i+1}: curt∈[{K[idx].min():.0f},{K[idx].max():.0f}]  aluc={y[idx].mean():.2f} ({y[idx].sum()}/{len(idx)})")
