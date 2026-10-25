"""TEST DECISIVO: entre reportes donde el modelo AFIRMA un hallazgo focal, ¿la forma del
   heatmap distingue focal-CORRECTO de focal-ALUCINADO? Si gini/AUC cae a ~0.5 => la señal
   previa (0.73) era artefacto de tasa base ('afirmó algo focal'), no detector de alucinación."""
import os,sys,json,math,numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
sys.stdout.reconfigure(encoding="utf-8")
from scipy import ndimage
from scipy.stats import rankdata, kurtosis as kurt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
def spearman_manual(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if np.ptp(a)==0 or np.ptp(b)==0:return 0.0
    ra=rankdata(a);rb=rankdata(b);ra=ra-ra.mean();rb=rb-rb.mean()
    d=math.sqrt((ra*ra).sum()*(rb*rb).sum());return float((ra*rb).sum()/d) if d>0 else 0.0

SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
META=json.load(open(SC+r"\pertoken_meta.json")); NPZ=np.load(SC+r"\pertoken_hm.npz"); G=37

FOCAL_HALLUC={  # afirma patología focal que el real no respalda
 "CXR1001_IM-0004-1002","CXR1016_IM-0013-2001","CXR1019_IM-0015-2001","CXR103_IM-0023-2001",
 "CXR1030_IM-0024-1001","CXR1030_IM-0024-2001","CXR1035_IM-0028-2001","CXR1044_IM-0036-1001",
 "CXR1047_IM-0036-1001","CXR1047_IM-0036-2001"}
FOCAL_CORRECT={  # afirma un hallazgo localizado REAL (coincide con el informe)
 "CXR1000_IM-0003-1001","CXR1000_IM-0003-3001","CXR1001_IM-0004-1001","CXR1003_IM-0005-2002",
 "CXR101_IM-0011-2001","CXR101_IM-0011-4004","CXR1012_IM-0013-1001","CXR1014_IM-0013-1001",
 "CXR1014_IM-0013-2001","CXR1015_IM-0001-1001","CXR1015_IM-0001-2001","CXR1015_IM-0013-1001",
 "CXR1015_IM-0013-2001","CXR1016_IM-0013-1001","CXR1021_IM-0017-1001-0001","CXR1034_IM-0028-1001",
 "CXR1037_IM-0029-1001","CXR1043_IM-0035-0001-0001","CXR1043_IM-0035-0001-0002","CXR1050_IM-0038-4004",
 "CXR1053_IM-0040-1001","CXR1053_IM-0040-3003","CXR105_IM-0037-1001","CXR1022_IM-0017-2001"}
sub={i:1 for i in FOCAL_HALLUC}; sub.update({i:0 for i in FOCAL_CORRECT})
items=[r for r in META if r["id"] in sub and f"hm_{r['k']}" in NPZ.files]

rows=np.concatenate([NPZ[f"hm_{r['k']}"].astype(np.float64) for r in items],0)
posmean=rows.mean(0); thr=posmean.mean()+6*posmean.std()
sink=np.where(posmean>thr)[0]; nonsink=np.ones(G*G,bool); nonsink[sink]=False; smask=nonsink.reshape(G,G)
def norm_ns(v):
    a=v[nonsink].astype(np.float64);a=a-a.min();s=a.sum();return a/s if s>0 else np.ones_like(a)/len(a)
mean_pat=norm_ns(rows.mean(0))
ids,H=[],[]
for r in items:
    hm=NPZ[f"hm_{r['k']}"].astype(np.float64)
    P=np.array([norm_ns(hm[i]) for i in range(len(r["tokens"]))])
    w=np.sqrt(((P-mean_pat)**2).sum(1)); w=w if w.sum()>0 else np.ones(len(P))
    agg=(w[:,None]*P).sum(0)/w.sum(); full=np.zeros(G*G); full[nonsink]=agg
    ids.append(r["id"]); H.append(full.reshape(G,G))
H=np.array(H); y=np.array([sub[i] for i in ids])
print(f"focal subset: correcto={int((y==0).sum())}  alucinado={int((y==1).sum())}")

ys,xs=np.mgrid[0:G,0:G]
def feats(h):
    hm=np.where(smask,h,0.0).astype(np.float64); m=hm[smask]
    if m.max()<=m.min(): return [0.5,0,0,0,0,0,0,0,0]
    p=m/m.sum(); ent=float(-(p*np.log(p+1e-12)).sum()/math.log(len(p)))
    srt=np.sort(p);n=len(p);gini=float((2*np.arange(1,n+1)-n-1).dot(srt)/(n*srt.sum()+1e-12))
    py,px=np.unravel_index(hm.argmax(),hm.shape); dist=np.sqrt((ys-py)**2+(xs-px)**2)
    peak=float(hm[dist<=3].sum()/(hm.sum()+1e-12))
    radial=float(-spearman_manual(hm.flatten()[nonsink],dist.flatten()[nonsink]))
    mask=hm>=np.percentile(m,85);lab,nc=ndimage.label(mask)
    sizes=ndimage.sum(mask,lab,range(1,nc+1)) if nc else [0]; maxc=float(max(sizes)/max(mask.sum(),1))
    w2=hm/(hm.sum()+1e-12);cy,cx=(w2*ys).sum(),(w2*xs).sum()
    spread=float(math.sqrt((w2*((ys-cy)**2+(xs-cx)**2)).sum()))
    mx=ndimage.maximum_filter(hm,size=3);npk=int(((hm==mx)&(hm>np.percentile(m,80))).sum())
    kv=float(kurt(m));xg=hm-hm.mean()
    num=(xg*np.roll(hm,1,0)+xg*np.roll(hm,-1,0)+xg*np.roll(hm,1,1)+xg*np.roll(hm,-1,1)).sum()
    moran=float(num/(4*(xg**2).sum()+1e-12))
    return [ent,gini,peak,radial,maxc,spread,npk,kv,moran]
NAMES=["entropy","gini","peak_conc","radial","max_clus","spread","n_peaks","kurtosis","moran"]
X=np.array([feats(h) for h in H])
print(f"\n{'feature':11}{'correcto':>10}{'alucina':>9}{'AUC':>7}")
for j,nm in enumerate(NAMES):
    f0,f1=X[y==0,j].mean(),X[y==1,j].mean();auc=roc_auc_score(y,X[:,j]);auc=max(auc,1-auc)
    print(f"{nm:11}{f0:>10.3f}{f1:>9.3f}{auc:>7.2f}")
cv=StratifiedKFold(5,shuffle=True,random_state=0)
pipe=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000,class_weight="balanced"))
pr=cross_val_predict(pipe,X,y,cv=cv,method="predict_proba")[:,1]; auc0=roc_auc_score(y,pr)
rng=np.random.default_rng(0);null=[]
for _ in range(300):
    yp=rng.permutation(y)
    try:null.append(roc_auc_score(yp,cross_val_predict(pipe,X,yp,cv=StratifiedKFold(5,shuffle=True,random_state=1),method="predict_proba")[:,1]))
    except Exception:pass
null=np.array(null);pval=(1+np.sum(null>=auc0))/(1+len(null))
print(f"\nComposite OOF AUC={auc0:.3f}  | permutación nulo={null.mean():.3f} p95={np.percentile(null,95):.3f} p={pval:.3f}")
print("\nINTERPRETA: AUC~0.5 => la señal previa era 'afirmó algo focal' (tasa base), NO detector de alucinación.")
