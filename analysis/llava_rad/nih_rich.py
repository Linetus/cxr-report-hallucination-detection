"""Índice de confianza sobre NIH (n=388, etiqueta OBJETIVA: mal-anclado = pico fuera de bbox).
   Busca la mejor combinación de features de forma con validación honesta (OOF + permutación).
   Es el test con potencia estadística real (vs 14 positivos de Open-I)."""
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
R=[r for r in json.load(open(SC+r"\nih_results.json")) if r["detected"]]
G=37; SINK=np.ones((G,G),bool); SINK[:2,:2]=False; ys,xs=np.mgrid[0:G,0:G]; nn=SINK.flatten()
def feats(hm):
    H=np.array(hm,float).reshape(G,G); H=np.where(SINK,H,0.0); m=H[SINK]
    if m.max()<=m.min(): return [1,0,0,0,0,0,0,0,0]
    p=m/m.sum(); ent=float(-(p*np.log(p+1e-12)).sum()/math.log(len(p)))
    srt=np.sort(p);n=len(p);gini=float((2*np.arange(1,n+1)-n-1).dot(srt)/(n*srt.sum()+1e-12))
    py,px=np.unravel_index(H.argmax(),H.shape);d=np.sqrt((ys-py)**2+(xs-px)**2)
    peak=float(H[d<=3].sum()/(H.sum()+1e-12)); radial=float(-spear(H.flatten()[nn],d.flatten()[nn]))
    w2=H/(H.sum()+1e-12);spread=float(math.sqrt((w2*((ys-py)**2+(xs-px)**2)).sum()))
    mask=H>=np.percentile(m,85);lab,nc=ndimage.label(mask)
    sizes=sorted(ndimage.sum(mask,lab,range(1,nc+1)),reverse=True) if nc else [0]
    maxc=float(sizes[0]/max(mask.sum(),1)); pk2=float(sizes[1]/(sizes[0]+1e-9)) if len(sizes)>1 else 0.0
    xg=H-H.mean();num=(xg*np.roll(H,1,0)+xg*np.roll(H,-1,0)+xg*np.roll(H,1,1)+xg*np.roll(H,-1,1)).sum()
    moran=float(num/(4*(xg**2).sum()+1e-12))
    return [ent,gini,peak,spread,radial,maxc,int(nc),pk2,moran]
NAMES=["entropy","gini","peak_conc","spread","radial","max_clus","n_modes","peak2_ratio","moran"]
X=np.array([feats(r["hm"]) for r in R]); y=np.array([0 if r["pointing"] else 1 for r in R])  # 1=mal-anclado
print(f"n={len(R)}  mal-anclado(1)={int(y.sum())}  bien-anclado(0)={int((y==0).sum())}")
print("\nunivariante |AUC| (predecir mal-anclado):")
for j in sorted(range(len(NAMES)),key=lambda j:-max(roc_auc_score(y,X[:,j]),1-roc_auc_score(y,X[:,j]))):
    a=roc_auc_score(y,X[:,j]);a=max(a,1-a)
    print(f"   {NAMES[j]:12} AUC={a:.3f}  (bien {X[y==0,j].mean():.3f} vs mal {X[y==1,j].mean():.3f})")
cv=StratifiedKFold(5,shuffle=True,random_state=0)
lg=make_pipeline(StandardScaler(),LogisticRegression(C=0.5,penalty="l1",solver="liblinear",class_weight="balanced"))
aL=roc_auc_score(y,cross_val_predict(lg,X,y,cv=cv,method="predict_proba")[:,1])
gb=GradientBoostingClassifier(n_estimators=150,max_depth=2,learning_rate=0.05)
aG=roc_auc_score(y,cross_val_predict(gb,X,y,cv=cv,method="predict_proba")[:,1])
print(f"\nOOF AUC  L1-logística={aL:.3f}   GradientBoosting={aG:.3f}")
rng=np.random.default_rng(0);null=[]
for _ in range(300):
    yp=rng.permutation(y)
    try:null.append(roc_auc_score(yp,cross_val_predict(lg,X,yp,cv=StratifiedKFold(5,shuffle=True,random_state=1),method="predict_proba")[:,1]))
    except Exception:pass
null=np.array(null);pv=(1+np.sum(null>=aL))/(1+len(null))
print(f"permutación L1: nulo={null.mean():.3f} p95={np.percentile(null,95):.3f}  p={pv:.3f}")
clf=LogisticRegression(C=0.5,penalty="l1",solver="liblinear",class_weight="balanced").fit(StandardScaler().fit_transform(X),y)
nz=[(NAMES[j],clf.coef_[0][j]) for j in range(len(NAMES)) if abs(clf.coef_[0][j])>1e-6]
print("FÓRMULA L1 (+ = MAL-ANCLADO):", ", ".join(f"{n}:{c:+.2f}" for n,c in sorted(nz,key=lambda z:-abs(z[1]))))
