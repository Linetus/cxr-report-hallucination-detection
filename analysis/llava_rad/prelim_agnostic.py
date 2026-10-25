"""Fórmula AGNÓSTICA (sin lexicón de keywords). Señales:
   - min_prob / entropía (probabilidades de token)
   - auto-consistencia = similitud de texto entre muestras (SelfCheckGPT real)
   - ablación en blanco = similitud gen-vs-imagen-en-blanco (¿cambia al quitar la imagen?)
   Similitud por n-gramas de carácter (robusto a la salida sin espacios). Env: tfm-radiology."""
import os,sys,json,math
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
LAB=json.load(open(SC+r"\labels.json"))
recs=[json.loads(l) for l in open(SC+r"\allsignals_openi.jsonl",encoding="utf-8")]
def ng(t,n=4): t=t.lower(); return set(t[i:i+n] for i in range(max(len(t)-n+1,1)))
def sim(a,b):
    A,B=ng(a),ng(b); u=len(A|B); return len(A&B)/u if u else 1.0
rows=[]
for r in recs:
    if r["id"] not in LAB: continue
    p=np.array(r["probs"]); e=np.array(r["ents"]); gen=r["gen"]; sm=r.get("samples",[]); bg=r.get("blank_gen","")
    # auto-consistencia AGNÓSTICA: similitud media gen<->muestra (baja => varía => alucina)
    consist=float(np.mean([sim(gen,s) for s in sm])) if sm else 1.0
    # ablación en blanco AGNÓSTICA: similitud gen<->blank (alta => no usa imagen => prior)
    blank_sim=float(sim(gen,bg)) if bg else 0.0
    rows.append(dict(y=LAB[r["id"]], min_prob=float(p.min()), max_ent=float(e.max()),
        mean_ent=float(e.mean()), consist=consist, blank_sim=blank_sim))
y=np.array([r["y"] for r in rows]); ids=[r["id"] for r in recs if r["id"] in LAB]
FEATS=["min_prob","max_ent","mean_ent","consist","blank_sim"]
def col(f): return np.array([r[f] for r in rows])
print(f"n={len(rows)} alucina={int(y.sum())}  (SEÑALES AGNÓSTICAS, sin keywords)")
print(f"\n{'señal':11}{'fiable':>9}{'alucina':>9}{'AUC':>7}")
for f in FEATS:
    v=col(f); a=roc_auc_score(y,v); print(f"{f:11}{v[y==0].mean():>9.3f}{v[y==1].mean():>9.3f}{max(a,1-a):>7.2f}")
def composite(mask,tag):
    X=np.column_stack([col(f)[mask] for f in FEATS]); yy=y[mask]
    lg=make_pipeline(StandardScaler(),LogisticRegression(C=0.5,penalty="l1",solver="liblinear",class_weight="balanced"))
    pr=cross_val_predict(lg,X,yy,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1]; a0=roc_auc_score(yy,pr)
    rng=np.random.default_rng(0); null=[]
    for _ in range(500):
        yp=rng.permutation(yy)
        try: null.append(roc_auc_score(yp,cross_val_predict(lg,X,yp,cv=StratifiedKFold(5,shuffle=True,random_state=1),method="predict_proba")[:,1]))
        except Exception: pass
    null=np.array(null); print(f"{tag}: COMPOSITE AGNÓSTICO OOF AUC={a0:.3f}  p={(1+np.sum(null>=a0))/(1+len(null)):.3f}")
composite(np.ones(len(y),bool),"\nFULL")
FOC={"CXR1001_IM-0004-1002","CXR1016_IM-0013-2001","CXR1019_IM-0015-2001","CXR103_IM-0023-2001","CXR1030_IM-0024-1001","CXR1030_IM-0024-2001","CXR1035_IM-0028-2001","CXR1044_IM-0036-1001","CXR1047_IM-0036-1001","CXR1047_IM-0036-2001","CXR1000_IM-0003-1001","CXR1000_IM-0003-3001","CXR1001_IM-0004-1001","CXR1003_IM-0005-2002","CXR101_IM-0011-2001","CXR101_IM-0011-4004","CXR1012_IM-0013-1001","CXR1014_IM-0013-1001","CXR1014_IM-0013-2001","CXR1015_IM-0001-1001","CXR1015_IM-0001-2001","CXR1015_IM-0013-1001","CXR1015_IM-0013-2001","CXR1016_IM-0013-1001","CXR1021_IM-0017-1001-0001","CXR1034_IM-0028-1001","CXR1037_IM-0029-1001","CXR1043_IM-0035-0001-0001","CXR1043_IM-0035-0001-0002","CXR1050_IM-0038-4004","CXR1053_IM-0040-1001","CXR1053_IM-0040-3003","CXR105_IM-0037-1001","CXR1022_IM-0017-2001"}
fm=np.array([i in FOC for i in ids])
print("univar FOCAL:", ", ".join(f"{f}:{max(roc_auc_score(y[fm],col(f)[fm]),1-roc_auc_score(y[fm],col(f)[fm])):.2f}" for f in FEATS))
composite(fm,"FOCAL")
