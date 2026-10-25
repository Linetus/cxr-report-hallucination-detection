"""Fórmula definitiva sobre n=559. Set rico + entropía RESIDUALIZADA por longitud."""
import os,sys,json,math
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
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
    ra=rankdata(a);rb=rankdata(b);ra-=ra.mean();rb-=rb.mean()
    return float((ra*rb).sum()/math.sqrt((ra*ra).sum()*(rb*rb).sum()+1e-9))
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
LAB=json.load(open(SC+r"\labels_full.json")); recs=[json.loads(l) for l in open(SC+r"\allsignals_openi.jsonl",encoding="utf-8")]
def ng(t,n=4): t=t.lower(); return set(t[i:i+n] for i in range(max(len(t)-n+1,1)))
def sim(a,b):
    A,B=ng(a),ng(b);u=len(A|B);return len(A&B)/u if u else 1.0
rows=[]
for r in recs:
    if r["id"] not in LAB: continue
    p=np.array(r["probs"]); e=np.array(r["ents"]); vm=np.array(r["vis_mass"]); smp=r.get("samples",[])
    lo=np.argsort(p)[:max(3,len(p)//5)]; sims=[sim(r["gen"],s) for s in smp]
    pw=[sim(smp[i],smp[j]) for i in range(len(smp)) for j in range(i+1,len(smp))]
    rows.append(dict(y=LAB[r["id"]],ntok=len(p),min_prob=float(p.min()),p05=float(np.percentile(p,5)),
        fraclow=float((p<0.3).mean()),consist=float(np.mean(sims)) if sims else 1.0,
        consist_min=float(np.min(sims)) if sims else 1.0,consist_pw=float(np.mean(pw)) if pw else 1.0,
        imgdep=float(vm.mean()),imgdep_lowp=float(vm[lo].mean()),
        blank_sim=float(sim(r["gen"],r.get("blank_gen",""))) if r.get("blank_gen") else 0.0,
        max_ent=float(e.max()),mean_ent=float(e.mean())))
y=np.array([r["y"] for r in rows]); col=lambda f:np.array([r[f] for r in rows],float); nt=col("ntok")
def resid(v):
    x=nt-nt.mean(); sl=(x*(v-v.mean())).sum()/((x*x).sum()+1e-9); return v-(v.mean()+sl*(nt-nt.mean()))
for r,rv in zip(rows,resid(col("max_ent"))): r["res_maxent"]=float(rv)
for r,rv in zip(rows,resid(col("mean_ent"))): r["res_meanent"]=float(rv)
print(f"n={len(rows)} alucina={int(y.sum())}\n{'feature':12}{'AUC':>6}{'corrLong':>9}")
for f in ["min_prob","p05","fraclow","consist","consist_min","consist_pw","imgdep_lowp","blank_sim","max_ent","res_maxent","res_meanent"]:
    v=col(f); print(f"{f:12}{max(roc_auc_score(y,v),1-roc_auc_score(y,v)):>6.3f}{spear(v,nt):>+9.2f}")
def comp(feats,tag,gbm=False):
    X=np.column_stack([col(f) for f in feats])
    mdl=GradientBoostingClassifier(n_estimators=150,max_depth=2,learning_rate=0.05) if gbm else make_pipeline(StandardScaler(),LogisticRegression(C=0.5,penalty="l1",solver="liblinear",class_weight="balanced"))
    pr=cross_val_predict(mdl,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1]
    a0=roc_auc_score(y,pr); rng=np.random.default_rng(0);null=[]
    for _ in range(400):
        yp=rng.permutation(y)
        try:null.append(roc_auc_score(yp,cross_val_predict(mdl,X,yp,cv=StratifiedKFold(5,shuffle=True,random_state=1),method="predict_proba")[:,1]))
        except Exception:pass
    null=np.array(null);print(f"  {tag:40} AUC={a0:.3f} p={(1+np.sum(null>=a0))/(1+len(null)):.3f}")
RICH=["min_prob","p05","fraclow","consist","consist_min","consist_pw","imgdep_lowp","blank_sim"]
print("\n=== COMPOSITES (n=559) ===")
comp(["min_prob","consist"],"base: min_prob+consist")
comp(RICH,"rico agnóstico (L1)")
comp(RICH+["res_maxent"],"rico + entropía RESIDUALIZADA (L1)")
comp(RICH+["res_maxent"],"rico + entropía RESIDUALIZADA (GBM)",gbm=True)
