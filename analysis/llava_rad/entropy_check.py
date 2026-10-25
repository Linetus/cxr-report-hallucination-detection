"""¿La entropía es válida o es confound de longitud/régimen? Y composite SIN entropía."""
import os,sys,json,math
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
def spear(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if np.ptp(a)==0 or np.ptp(b)==0:return 0.0
    ra=rankdata(a);rb=rankdata(b);ra-=ra.mean();rb-=rb.mean()
    return float((ra*rb).sum()/math.sqrt((ra*ra).sum()*(rb*rb).sum()+1e-9))
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
LAB=json.load(open(SC+r"\labels_expanded.json")); recs=[json.loads(l) for l in open(SC+r"\allsignals_openi.jsonl",encoding="utf-8")]
def ng(t,n=4): t=t.lower(); return set(t[i:i+n] for i in range(max(len(t)-n+1,1)))
def sim(a,b):
    A,B=ng(a),ng(b);u=len(A|B);return len(A&B)/u if u else 1.0
rows=[]
for r in recs:
    if r["id"] not in LAB: continue
    p=np.array(r["probs"]); e=np.array(r["ents"]); smp=r.get("samples",[])
    rows.append(dict(y=LAB[r["id"]],min_prob=float(p.min()),max_ent=float(e.max()),mean_ent=float(e.mean()),
        consist=float(np.mean([sim(r["gen"],s) for s in smp])) if smp else 1.0,
        blank_sim=float(sim(r["gen"],r.get("blank_gen",""))) if r.get("blank_gen") else 0.0,
        ntok=len(r["tokens"])))
y=np.array([r["y"] for r in rows]); col=lambda f:np.array([r[f] for r in rows])
nt=col("ntok")
print(f"n={len(rows)} alucina={int(y.sum())}")
print("\n=== ¿ENTROPÍA = LONGITUD? ===")
print(f"Spearman(max_ent, ntok)  = {spear(col('max_ent'),nt):+.2f}")
print(f"Spearman(mean_ent, ntok) = {spear(col('mean_ent'),nt):+.2f}")
print(f"AUC(ntok -> alucina)     = {max(roc_auc_score(y,nt),1-roc_auc_score(y,nt)):.3f}   (si alto: la LONGITUD sola ya 'predice')")
print(f"ntok: fiable={nt[y==0].mean():.0f} vs alucina={nt[y==1].mean():.0f}")
print(f"Spearman(min_prob, ntok) = {spear(col('min_prob'),nt):+.2f}  (min_prob poco ligado a longitud)")
print(f"Spearman(consist, ntok)  = {spear(col('consist'),nt):+.2f}")
def comp(feats,tag):
    X=np.column_stack([col(f) for f in feats])
    lg=make_pipeline(StandardScaler(),LogisticRegression(C=0.5,penalty="l1",solver="liblinear",class_weight="balanced"))
    pr=cross_val_predict(lg,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1]
    a0=roc_auc_score(y,pr); rng=np.random.default_rng(0);null=[]
    for _ in range(500):
        yp=rng.permutation(y)
        try:null.append(roc_auc_score(yp,cross_val_predict(lg,X,yp,cv=StratifiedKFold(5,shuffle=True,random_state=1),method="predict_proba")[:,1]))
        except Exception:pass
    null=np.array(null);print(f"  {tag:38} OOF AUC={a0:.3f}  p={(1+np.sum(null>=a0))/(1+len(null)):.3f}")
print("\n=== COMPOSITES ===")
comp(["min_prob","max_ent","mean_ent","consist","blank_sim"],"CON entropía (el reportado)")
comp(["min_prob","consist","blank_sim"],"SIN entropía (limpio)")
comp(["min_prob","consist"],"solo min_prob + consist")
