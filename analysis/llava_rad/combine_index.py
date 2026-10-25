"""Índice combinado min_prob (incertidumbre) + spread (dispersión heatmap). ¿Diferencia
   notable entre errores y correctos? Separación por clase, Cohen's d, AUC, permutación."""
import os,sys,json,math,numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
sys.stdout.reconfigure(encoding="utf-8")
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
def spear(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if np.ptp(a)==0 or np.ptp(b)==0:return 0.0
    ra=rankdata(a);rb=rankdata(b);ra=ra-ra.mean();rb=rb-rb.mean()
    d=math.sqrt((ra*ra).sum()*(rb*rb).sum());return float((ra*rb).sum()/d) if d>0 else 0.0
def cohen(a,b):
    na,nb=len(a),len(b); s=math.sqrt(((na-1)*a.var(ddof=1)+(nb-1)*b.var(ddof=1))/(na+nb-2)+1e-12)
    return (a.mean()-b.mean())/s
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
GC={r["id"]:r for r in json.load(open(SC+r"\gen_conf.json"))}
META=json.load(open(SC+r"\pertoken_meta.json")); NPZ=np.load(SC+r"\pertoken_hm.npz")
LAB=json.load(open(SC+r"\labels.json")); G=37
items=[r for r in META if f"hm_{r['k']}" in NPZ.files and r["id"] in LAB and r["id"] in GC]
rows=np.concatenate([NPZ[f"hm_{r['k']}"].astype(np.float64) for r in items],0)
pm=rows.mean(0); thr=pm.mean()+6*pm.std(); sink=np.where(pm>thr)[0]
nonsink=np.ones(G*G,bool); nonsink[sink]=False; smask=nonsink.reshape(G,G)
def norm_ns(v):
    a=v[nonsink].astype(np.float64);a=a-a.min();s=a.sum();return a/s if s>0 else np.ones_like(a)/len(a)
mean_pat=norm_ns(rows.mean(0)); ys,xs=np.mgrid[0:G,0:G]
def spread_of(r):
    hm=NPZ[f"hm_{r['k']}"].astype(np.float64)
    P=np.array([norm_ns(hm[i]) for i in range(len(r["tokens"]))])
    w=np.sqrt(((P-mean_pat)**2).sum(1)); w=w if w.sum()>0 else np.ones(len(P))
    agg=(w[:,None]*P).sum(0)/w.sum(); H=np.zeros(G*G);H[nonsink]=agg;H=H.reshape(G,G)
    w2=H/(H.sum()+1e-12);cy,cx=(w2*ys).sum(),(w2*xs).sum()
    return math.sqrt((w2*((ys-cy)**2+(xs-cx)**2)).sum())
ids=[r["id"] for r in items]
mp=np.array([min(GC[i]["probs"]) for i in ids]); sp=np.array([spread_of(r) for r in items])
y=np.array([LAB[i] for i in ids])
print(f"n={len(ids)}  corr(min_prob,spread)={spear(mp,sp):+.2f}  (independientes si ~0 => combinar ayuda)")

FOC={"CXR1001_IM-0004-1002","CXR1016_IM-0013-2001","CXR1019_IM-0015-2001","CXR103_IM-0023-2001","CXR1030_IM-0024-1001","CXR1030_IM-0024-2001","CXR1035_IM-0028-2001","CXR1044_IM-0036-1001","CXR1047_IM-0036-1001","CXR1047_IM-0036-2001","CXR1000_IM-0003-1001","CXR1000_IM-0003-3001","CXR1001_IM-0004-1001","CXR1003_IM-0005-2002","CXR101_IM-0011-2001","CXR101_IM-0011-4004","CXR1012_IM-0013-1001","CXR1014_IM-0013-1001","CXR1014_IM-0013-2001","CXR1015_IM-0001-1001","CXR1015_IM-0001-2001","CXR1015_IM-0013-1001","CXR1015_IM-0013-2001","CXR1016_IM-0013-1001","CXR1021_IM-0017-1001-0001","CXR1034_IM-0028-1001","CXR1037_IM-0029-1001","CXR1043_IM-0035-0001-0001","CXR1043_IM-0035-0001-0002","CXR1050_IM-0038-4004","CXR1053_IM-0040-1001","CXR1053_IM-0040-3003","CXR105_IM-0037-1001","CXR1022_IM-0017-2001"}
fmask=np.array([i in FOC for i in ids])
# índice interpretable: z-scores, alucina = min_prob baja + spread bajo
def zc(v,mask): m,s=v[mask].mean(),v[mask].std()+1e-9; return (v-m)/s
def block(mask,tag):
    yy=y[mask]
    idx = zc(-mp,mask) + zc(-sp,mask)            # índice simple (suma de z, dirección alucina)
    print(f"\n===== {tag} (n={mask.sum()}, aluc={int(yy.sum())}) =====")
    for nm,v,lo in [("min_prob",mp,True),("spread",sp,True),("ÍNDICE z(-mp)+z(-sp)",idx,False)]:
        vv=v[mask]; a=roc_auc_score(yy,-vv if lo else vv); a=max(a,1-a)
        d=cohen(idx[mask][yy==1],idx[mask][yy==0]) if nm.startswith("Í") else cohen((-vv)[yy==1],(-vv)[yy==0])
        print(f"  {nm:20} correcto={vv[yy==0].mean():.3f} error={vv[yy==1].mean():.3f}  AUC={a:.3f}  Cohen_d={d:+.2f}")
    # logística 2-feat OOF + permutación
    X=np.column_stack([mp[mask],sp[mask]])
    cv=StratifiedKFold(5,shuffle=True,random_state=0)
    lg=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight="balanced",max_iter=2000))
    pr=cross_val_predict(lg,X,yy,cv=cv,method="predict_proba")[:,1]; a0=roc_auc_score(yy,pr)
    rng=np.random.default_rng(0);null=[]
    for _ in range(2000):
        yp=rng.permutation(yy)
        try:null.append(roc_auc_score(yp,cross_val_predict(lg,X,yp,cv=StratifiedKFold(5,shuffle=True,random_state=1),method="predict_proba")[:,1]))
        except Exception:pass
    null=np.array(null)
    print(f"  COMBINADO (logística 2-feat) OOF AUC={a0:.3f}  p={(1+np.sum(null>=a0))/(1+len(null)):.3f}")
block(np.ones(len(y),bool),"FULL")
block(fmask,"FOCAL")
