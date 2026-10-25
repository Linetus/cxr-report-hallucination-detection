"""MÉTRICA DE CONFIANZA por imagen desde la forma del heatmap ponderado-por-VALOR.
   Compara esquemas de peso data-driven (sin keywords), valida qué features de forma
   (incl. curtosis) se relacionan con alucinación, funde en score de confianza (logística),
   y evalúa significancia por PERMUTACIÓN dado el desbalance (14 positivos). Env: tfm-radiology."""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
sys.stdout.reconfigure(encoding="utf-8")
import faulthandler; faulthandler.enable()
import json, math, numpy as np
from scipy import ndimage
from scipy.stats import rankdata, kurtosis as kurt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

def spearman_manual(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if np.ptp(a)==0 or np.ptp(b)==0: return 0.0
    ra=rankdata(a); rb=rankdata(b); ra=ra-ra.mean(); rb=rb-rb.mean()
    d=math.sqrt((ra*ra).sum()*(rb*rb).sum()); return float((ra*rb).sum()/d) if d>0 else 0.0

SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
META=json.load(open(SC+r"\pertoken_meta.json")); NPZ=np.load(SC+r"\pertoken_hm.npz")
LAB=json.load(open(SC+r"\labels.json")); G=37
items=[r for r in META if f"hm_{r['k']}" in NPZ.files and r["id"] in LAB]

rows=np.concatenate([NPZ[f"hm_{r['k']}"].astype(np.float64) for r in items],0)
posmean=rows.mean(0); thr=posmean.mean()+6*posmean.std()
sink=np.where(posmean>thr)[0]; nonsink=np.ones(G*G,bool); nonsink[sink]=False; smask=nonsink.reshape(G,G)
print("sink:",[tuple(np.unravel_index(int(s),(G,G))) for s in sink])
def norm_ns(v):
    a=v[nonsink].astype(np.float64); a=a-a.min(); s=a.sum(); return a/s if s>0 else np.ones_like(a)/len(a)
mean_pat=norm_ns(rows.mean(0))

def tok_weight(P, scheme):                      # P: (n_tok,|nonsink|) normalizado
    if scheme=="uniform": return np.ones(len(P))
    if scheme=="distinct": return np.sqrt(((P-mean_pat)**2).sum(1))
    if scheme=="conc":     return P.max(1)/(P.mean(1)+1e-12)
    if scheme=="mass":     return P.sum(1)  # (ya norm => ~cte; placeholder)
def build_H(scheme):
    ids,H=[],[]
    for r in items:
        hm=NPZ[f"hm_{r['k']}"].astype(np.float64)
        P=np.array([norm_ns(hm[i]) for i in range(len(r["tokens"]))])
        w=tok_weight(P,scheme); w=w if w.sum()>0 else np.ones(len(P))
        agg=(w[:,None]*P).sum(0)/w.sum(); full=np.zeros(G*G); full[nonsink]=agg
        ids.append(r["id"]); H.append(full.reshape(G,G))
    return ids,np.array(H)

ys,xs=np.mgrid[0:G,0:G]
def feats(h):
    hm=np.where(smask,h,0.0).astype(np.float64); m=hm[smask]
    if m.max()<=m.min(): return [0.5,0,0,0,0,0,0,0,0]
    p=m/m.sum()
    ent=float(-(p*np.log(p+1e-12)).sum()/math.log(len(p)))
    srt=np.sort(p); n=len(p); gini=float((2*np.arange(1,n+1)-n-1).dot(srt)/(n*srt.sum()+1e-12))
    py,px=np.unravel_index(hm.argmax(),hm.shape); dist=np.sqrt((ys-py)**2+(xs-px)**2)
    peak=float(hm[dist<=3].sum()/(hm.sum()+1e-12))
    hf=hm.flatten(); radial=float(-spearman_manual(hf[nonsink],dist.flatten()[nonsink]))
    mask=hm>=np.percentile(m,85); lab,nc=ndimage.label(mask)
    sizes=ndimage.sum(mask,lab,range(1,nc+1)) if nc else [0]; maxc=float(max(sizes)/max(mask.sum(),1))
    w2=hm/(hm.sum()+1e-12); cy,cx=(w2*ys).sum(),(w2*xs).sum()
    spread=float(math.sqrt((w2*((ys-cy)**2+(xs-cx)**2)).sum()))
    mx=ndimage.maximum_filter(hm,size=3); npk=int(((hm==mx)&(hm>np.percentile(m,80))).sum())
    kv=float(kurt(m)); xg=hm-hm.mean()
    num=(xg*np.roll(hm,1,0)+xg*np.roll(hm,-1,0)+xg*np.roll(hm,1,1)+xg*np.roll(hm,-1,1)).sum()
    moran=float(num/(4*(xg**2).sum()+1e-12))
    return [ent,gini,peak,radial,maxc,spread,npk,kv,moran]
NAMES=["entropy","gini","peak_conc","radial","max_clus","spread","n_peaks","kurtosis","moran"]

def oof_auc(X,y,seed=0):
    cv=StratifiedKFold(5,shuffle=True,random_state=seed)
    pipe=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000,class_weight="balanced"))
    pr=cross_val_predict(pipe,X,y,cv=cv,method="predict_proba")[:,1]
    return roc_auc_score(y,pr),pr

# ids/y de referencia (mismo orden para todo)
ids,_=build_H("uniform"); y=np.array([LAB[i] for i in ids])
print(f"n={len(ids)}  fiables={int((y==0).sum())}  alucina={int((y==1).sum())}")

# --- COMPARA ESQUEMAS DE PESO ---
print("\n=== OOF AUC por esquema de ponderación por token ===")
Xs={}
for s in ["uniform","distinct","conc","mass"]:
    _,H=build_H(s); X=np.array([feats(h) for h in H]); Xs[s]=X
    a,_=oof_auc(X,y); print(f"   {s:9} AUC={a:.3f}")

# --- esquema elegido: distinct ---
X=Xs["distinct"]
print("\n=== UNIVARIANTE (distinct): media fiable vs alucina + AUC ===")
print(f"{'feature':11}{'fiable':>9}{'alucina':>9}{'AUC':>7}")
for j,nm in enumerate(NAMES):
    f0,f1=X[y==0,j].mean(),X[y==1,j].mean(); auc=roc_auc_score(y,X[:,j]); auc=max(auc,1-auc)
    print(f"{nm:11}{f0:>9.3f}{f1:>9.3f}{auc:>7.2f}")
auc0,proba=oof_auc(X,y); conf=1-proba
clf=LogisticRegression(C=1.0,max_iter=2000,class_weight="balanced").fit(StandardScaler().fit_transform(X),y)
print(f"\nComposite OOF AUC = {auc0:.3f}")
print("coef (+ = ALUCINA):", ", ".join(f"{n}:{c:+.2f}" for n,c in sorted(zip(NAMES,clf.coef_[0]),key=lambda z:-abs(z[1]))))

# --- SIGNIFICANCIA por permutación (¿0.679 supera al azar dado n=14?) ---
rng=np.random.default_rng(0); null=[]
for _ in range(300):
    yp=rng.permutation(y)
    try: null.append(oof_auc(X,yp,seed=1)[0])
    except Exception: pass
null=np.array(null); pval=(1+np.sum(null>=auc0))/(1+len(null))
print(f"\nPermutación (300): AUC nulo medio={null.mean():.3f}  p95={np.percentile(null,95):.3f}  p-valor={pval:.3f}")
json.dump({ids[i]:float(conf[i]) for i in range(len(ids))},open(SC+r"\confidence_scores.json","w"))
print("confidence_scores.json guardado")
