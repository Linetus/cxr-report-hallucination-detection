"""Verifica min_prob: ¿significativo en focal? ¿artefacto de longitud? ¿composite robusto?"""
import os,sys,json,math,re,numpy as np
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
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
G=json.load(open(SC+r"\gen_conf.json")); LAB=json.load(open(SC+r"\labels.json"))
rows=[]
for r in G:
    if r["id"] not in LAB: continue
    p=np.array(r["probs"]); e=np.array(r["ents"])
    rows.append(dict(id=r["id"],y=LAB[r["id"]],min_prob=float(p.min()),max_ent=float(e.max()),
                     mean_ent=float(e.mean()),low=float((p<0.5).mean()),ntok=len(p),
                     # order-stats robustas
                     p05=float(np.percentile(p,5)),ent95=float(np.percentile(e,95))))
ids=[r["id"] for r in rows]; y=np.array([r["y"] for r in rows])
FOC={"CXR1001_IM-0004-1002","CXR1016_IM-0013-2001","CXR1019_IM-0015-2001","CXR103_IM-0023-2001","CXR1030_IM-0024-1001","CXR1030_IM-0024-2001","CXR1035_IM-0028-2001","CXR1044_IM-0036-1001","CXR1047_IM-0036-1001","CXR1047_IM-0036-2001","CXR1000_IM-0003-1001","CXR1000_IM-0003-3001","CXR1001_IM-0004-1001","CXR1003_IM-0005-2002","CXR101_IM-0011-2001","CXR101_IM-0011-4004","CXR1012_IM-0013-1001","CXR1014_IM-0013-1001","CXR1014_IM-0013-2001","CXR1015_IM-0001-1001","CXR1015_IM-0001-2001","CXR1015_IM-0013-1001","CXR1015_IM-0013-2001","CXR1016_IM-0013-1001","CXR1021_IM-0017-1001-0001","CXR1034_IM-0028-1001","CXR1037_IM-0029-1001","CXR1043_IM-0035-0001-0001","CXR1043_IM-0035-0001-0002","CXR1050_IM-0038-4004","CXR1053_IM-0040-1001","CXR1053_IM-0040-3003","CXR105_IM-0037-1001","CXR1022_IM-0017-2001"}
fmask=np.array([i in FOC for i in ids])
def arr(f):return np.array([r[f] for r in rows])
def permauc_uni(v,yy,pos_low=True):
    a=roc_auc_score(yy,-v if pos_low else v); a0=max(a,1-a)
    rng=np.random.default_rng(0);null=[roc_auc_score(rng.permutation(yy),-v if pos_low else v) for _ in range(2000)]
    null=np.array([max(x,1-x) for x in null]); return a0,(1+np.sum(null>=a0))/(1+len(null))

print("=== LONGITUD confound ===")
print(f"ntok: fiable={arr('ntok')[y==0].mean():.1f} alucina={arr('ntok')[y==1].mean():.1f}  | Spearman(min_prob,ntok)={spear(arr('min_prob'),arr('ntok')):+.2f}  AUC(ntok→aluc)={max(roc_auc_score(y,arr('ntok')),1-roc_auc_score(y,arr('ntok'))):.2f}")
for tag,mask in [("FULL",np.ones(len(y),bool)),("FOCAL",fmask)]:
    yy=y[mask]
    print(f"\n=== {tag} (n={mask.sum()}, aluc={int(yy.sum())}) — univar + permutación(2000) ===")
    for f,lo in [("min_prob",True),("p05",True),("max_ent",False),("ent95",False),("mean_ent",False),("low",False)]:
        a,pv=permauc_uni(arr(f)[mask],yy,lo); print(f"  {f:9} AUC={a:.3f}  p={pv:.3f}")
    # composite confianza (L1)
    X=np.column_stack([arr(f)[mask] for f in ["min_prob","p05","max_ent","ent95","mean_ent","low"]])
    cv=StratifiedKFold(5,shuffle=True,random_state=0)
    lg=make_pipeline(StandardScaler(),LogisticRegression(C=0.5,penalty="l1",solver="liblinear",class_weight="balanced"))
    pr=cross_val_predict(lg,X,yy,cv=cv,method="predict_proba")[:,1]; a0=roc_auc_score(yy,pr)
    rng=np.random.default_rng(0);null=[]
    for _ in range(300):
        yp=rng.permutation(yy)
        try:null.append(roc_auc_score(yp,cross_val_predict(lg,X,yp,cv=StratifiedKFold(5,shuffle=True,random_state=1),method="predict_proba")[:,1]))
        except Exception:pass
    null=np.array(null);print(f"  COMPOSITE confianza OOF AUC={a0:.3f}  p={(1+np.sum(null>=a0))/(1+len(null)):.3f}")
