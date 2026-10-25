"""ÍNDICE DE CONFIANZA enriquecido (Open-I). Añade features de HESITACIÓN (el modelo duda):
   sink-escape, incoherencia entre tokens (centroides discrepan), entropía media por token,
   nº de modos, ratio de picos — junto a densidades/sparsity/concentración. Busca la
   combinación que puntúe alto en alucinación con validación honesta (OOF + permutación)."""
import os,sys,json,math,numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
sys.stdout.reconfigure(encoding="utf-8")
from scipy import ndimage
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
def spear(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if np.ptp(a)==0 or np.ptp(b)==0:return 0.0
    ra=rankdata(a);rb=rankdata(b);ra=ra-ra.mean();rb=rb-rb.mean()
    d=math.sqrt((ra*ra).sum()*(rb*rb).sum());return float((ra*rb).sum()/d) if d>0 else 0.0
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
META=json.load(open(SC+r"\pertoken_meta.json")); NPZ=np.load(SC+r"\pertoken_hm.npz")
LAB=json.load(open(SC+r"\labels.json")); G=37
items=[r for r in META if f"hm_{r['k']}" in NPZ.files and r["id"] in LAB]
rows=np.concatenate([NPZ[f"hm_{r['k']}"].astype(np.float64) for r in items],0)
pm=rows.mean(0); thr=pm.mean()+6*pm.std(); sinkpos=np.where(pm>thr)[0]
nonsink=np.ones(G*G,bool); nonsink[sinkpos]=False; smask=nonsink.reshape(G,G)
gy,gx=np.divmod(np.arange(G*G),G); ys_ns,xs_ns=gy[nonsink],gx[nonsink]
ys,xs=np.mgrid[0:G,0:G]
def norm_ns(v):
    a=v[nonsink].astype(np.float64);a=a-a.min();s=a.sum();return a/s if s>0 else np.ones_like(a)/len(a)
mean_pat=norm_ns(rows.mean(0))

def feats(r):
    hm=NPZ[f"hm_{r['k']}"].astype(np.float64); nT=len(r["tokens"])
    P=np.array([norm_ns(hm[i]) for i in range(nT)])          # (nT,|ns|) normalizado
    w=np.sqrt(((P-mean_pat)**2).sum(1)); w=w if w.sum()>0 else np.ones(nT); wn=w/w.sum()
    # centroides + entropía por token
    cyi=(P*ys_ns).sum(1); cxi=(P*xs_ns).sum(1)
    enti=-(P*np.log(P+1e-12)).sum(1)/math.log(P.shape[1])
    # HESITACIÓN
    cy=(wn*cyi).sum(); cx=(wn*cxi).sum()
    incoh=math.sqrt((wn*((cyi-cy)**2+(cxi-cx)**2)).sum())     # discrepancia de centroides
    mean_ent=float((wn*enti).sum())                          # difuso medio por token
    sink_mass=hm[:,sinkpos].sum(1)/(hm.sum(1)+1e-12)         # fuga al sink por token
    sink_esc=float((wn*sink_mass).sum())
    # agregado ponderado
    agg=(wn[:,None]*P).sum(0); H=np.zeros(G*G); H[nonsink]=agg; H=H.reshape(G,G)
    m=H[smask]; p=m/(m.sum()+1e-12)
    ent=float(-(p*np.log(p+1e-12)).sum()/math.log(len(p)))
    srt=np.sort(p);n=len(p);gini=float((2*np.arange(1,n+1)-n-1).dot(srt)/(n*srt.sum()+1e-12))
    py,px=np.unravel_index(H.argmax(),H.shape);d=np.sqrt((ys-py)**2+(xs-px)**2)
    peak=float(H[d<=3].sum()/(H.sum()+1e-12)); radial=float(-spear(H.flatten()[nonsink],d.flatten()[nonsink]))
    w2=H/(H.sum()+1e-12);spread=float(math.sqrt((w2*((ys-py)**2+(xs-px)**2)).sum()))
    mask=H>=np.percentile(m,85);lab,nc=ndimage.label(mask)
    sizes=sorted(ndimage.sum(mask,lab,range(1,nc+1)),reverse=True) if nc else [0]
    maxc=float(sizes[0]/max(mask.sum(),1)); nmodes=int(nc)
    pk2=float(sizes[1]/(sizes[0]+1e-9)) if len(sizes)>1 else 0.0   # 2º modo / 1º
    xg=H-H.mean();num=(xg*np.roll(H,1,0)+xg*np.roll(H,-1,0)+xg*np.roll(H,1,1)+xg*np.roll(H,-1,1)).sum()
    moran=float(num/(4*(xg**2).sum()+1e-12))
    return [ent,gini,peak,spread,radial,maxc,moran,nmodes,pk2,incoh,mean_ent,sink_esc]
NAMES=["entropy","gini","peak_conc","spread","radial","max_clus","moran","n_modes","peak2_ratio",
       "incoherence","mean_tok_ent","sink_escape"]
ids=[r["id"] for r in items]; X=np.array([feats(r) for r in items]); y=np.array([LAB[i] for i in ids])
print(f"n={len(ids)} alucina={int(y.sum())}  features={len(NAMES)}")

def evalset(mask,tag):
    Xs,ys_=X[mask],y[mask]
    print(f"\n########## {tag}  (n={mask.sum()}, alucina={int(ys_.sum())}) ##########")
    print("univariante |AUC|:")
    order=sorted(range(len(NAMES)),key=lambda j:-max(roc_auc_score(ys_,Xs[:,j]),1-roc_auc_score(ys_,Xs[:,j])))
    for j in order:
        a=roc_auc_score(ys_,Xs[:,j]);a=max(a,1-a)
        print(f"   {NAMES[j]:13} AUC={a:.3f}  (fiable {Xs[ys_==0,j].mean():.3f} vs aluc {Xs[ys_==1,j].mean():.3f})")
    cv=StratifiedKFold(5,shuffle=True,random_state=0)
    lg=make_pipeline(StandardScaler(),LogisticRegression(C=0.5,penalty="l1",solver="liblinear",class_weight="balanced"))
    pr=cross_val_predict(lg,Xs,ys_,cv=cv,method="predict_proba")[:,1]; aL=roc_auc_score(ys_,pr)
    gb=GradientBoostingClassifier(n_estimators=120,max_depth=2,learning_rate=0.05)
    prg=cross_val_predict(gb,Xs,ys_,cv=cv,method="predict_proba")[:,1]; aG=roc_auc_score(ys_,prg)
    print(f"OOF AUC  L1-logística={aL:.3f}   GradientBoosting={aG:.3f}")
    # permutación sobre la logística L1
    rng=np.random.default_rng(0);null=[]
    for _ in range(300):
        yp=rng.permutation(ys_)
        try:null.append(roc_auc_score(yp,cross_val_predict(lg,Xs,yp,cv=StratifiedKFold(5,shuffle=True,random_state=1),method="predict_proba")[:,1]))
        except Exception:pass
    null=np.array(null);pv=(1+np.sum(null>=aL))/(1+len(null))
    print(f"permutación L1: nulo={null.mean():.3f} p95={np.percentile(null,95):.3f}  p={pv:.3f}")
    clf=LogisticRegression(C=0.5,penalty="l1",solver="liblinear",class_weight="balanced").fit(StandardScaler().fit_transform(Xs),ys_)
    nz=[(NAMES[j],clf.coef_[0][j]) for j in range(len(NAMES)) if abs(clf.coef_[0][j])>1e-6]
    print("FÓRMULA L1 (coef estand., + = ALUCINA):", ", ".join(f"{n}:{c:+.2f}" for n,c in sorted(nz,key=lambda z:-abs(z[1]))))
    return pr

pr_full=evalset(np.ones(len(y),bool),"FULL 120")
FOC={"CXR1001_IM-0004-1002","CXR1016_IM-0013-2001","CXR1019_IM-0015-2001","CXR103_IM-0023-2001","CXR1030_IM-0024-1001","CXR1030_IM-0024-2001","CXR1035_IM-0028-2001","CXR1044_IM-0036-1001","CXR1047_IM-0036-1001","CXR1047_IM-0036-2001","CXR1000_IM-0003-1001","CXR1000_IM-0003-3001","CXR1001_IM-0004-1001","CXR1003_IM-0005-2002","CXR101_IM-0011-2001","CXR101_IM-0011-4004","CXR1012_IM-0013-1001","CXR1014_IM-0013-1001","CXR1014_IM-0013-2001","CXR1015_IM-0001-1001","CXR1015_IM-0001-2001","CXR1015_IM-0013-1001","CXR1015_IM-0013-2001","CXR1016_IM-0013-1001","CXR1021_IM-0017-1001-0001","CXR1034_IM-0028-1001","CXR1037_IM-0029-1001","CXR1043_IM-0035-0001-0001","CXR1043_IM-0035-0001-0002","CXR1050_IM-0038-4004","CXR1053_IM-0040-1001","CXR1053_IM-0040-3003","CXR105_IM-0037-1001","CXR1022_IM-0017-2001"}
fmask=np.array([i in FOC for i in ids])
evalset(fmask,"FOCAL subset")
json.dump({ids[i]:float(pr_full[i]) for i in range(len(ids))},open(SC+r"\confidence_index.json","w"))
print("\n→ confidence_index.json guardado (P(alucina) por imagen)")
