"""Exploración de features para mejorar la fórmula. Uncertainty ricas, consistencia
   variada, dependencia-imagen, faithfulness (occ_drop), + control de longitud. n=335."""
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
LAB=json.load(open(SC+r"\labels_expanded.json")); recs=[json.loads(l) for l in open(SC+r"\allsignals_openi.jsonl",encoding="utf-8")]
def ng(t,n=4): t=t.lower(); return set(t[i:i+n] for i in range(max(len(t)-n+1,1)))
def sim(a,b):
    A,B=ng(a),ng(b);u=len(A|B);return len(A&B)/u if u else 1.0
rows=[]
for r in recs:
    if r["id"] not in LAB: continue
    p=np.array(r["probs"]); e=np.array(r["ents"]); vm=np.array(r["vis_mass"]); smp=r.get("samples",[])
    lo=np.argsort(p)[:max(3,len(p)//5)]                      # tokens más inseguros
    sims=[sim(r["gen"],s) for s in smp]
    pw=[sim(smp[i],smp[j]) for i in range(len(smp)) for j in range(i+1,len(smp))]
    rows.append(dict(y=LAB[r["id"]], ntok=len(p),
        min_prob=float(p.min()), p05=float(np.percentile(p,5)), fraclow=float((p<0.3).mean()),
        ppl=float(-np.log(p+1e-9).mean()), mean_ent=float(e.mean()), max_ent=float(e.max()),
        consist=float(np.mean(sims)) if sims else 1.0, consist_min=float(np.min(sims)) if sims else 1.0,
        consist_pw=float(np.mean(pw)) if pw else 1.0,
        imgdep=float(vm.mean()), imgdep_lowp=float(vm[lo].mean()),
        blank_sim=float(sim(r["gen"],r.get("blank_gen",""))) if r.get("blank_gen") else 0.0,
        occ=(r["occ_drop"] if r["occ_drop"] is not None else np.nan)))
y=np.array([r["y"] for r in rows]); col=lambda f:np.array([r[f] for r in rows],float)
od=col("occ"); od=np.where(np.isnan(od),np.nanmedian(od),od)
for i,r in enumerate(rows): r["occ"]=float(od[i])
nt=col("ntok")
FE=["min_prob","p05","fraclow","ppl","mean_ent","max_ent","consist","consist_min","consist_pw","imgdep","imgdep_lowp","blank_sim","occ"]
print(f"n={len(rows)} alucina={int(y.sum())}\n")
print(f"{'feature':12}{'AUC':>6}{'corr_long':>10}")
for f in FE:
    v=col(f); a=max(roc_auc_score(y,v),1-roc_auc_score(y,v))
    print(f"{f:12}{a:>6.3f}{spear(v,nt):>+10.2f}")
# entropía residualizada por longitud (OLS manual, sin BLAS): ¿señal más allá de longitud?
def resid(v):
    x=nt-nt.mean(); yv=v-v.mean(); sl=(x*yv).sum()/((x*x).sum()+1e-9); return v-(v.mean()+sl*(nt-nt.mean()))
for f in ["max_ent","mean_ent"]:
    r=resid(col(f)); print(f"  {f} residual (sin longitud): AUC={max(roc_auc_score(y,r),1-roc_auc_score(y,r)):.3f}")
def comp(feats,tag,gbm=False):
    X=np.column_stack([col(f) for f in feats])
    if gbm: mdl=GradientBoostingClassifier(n_estimators=150,max_depth=2,learning_rate=0.05)
    else: mdl=make_pipeline(StandardScaler(),LogisticRegression(C=0.5,penalty="l1",solver="liblinear",class_weight="balanced"))
    pr=cross_val_predict(mdl,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1]
    a0=roc_auc_score(y,pr); rng=np.random.default_rng(0);null=[]
    for _ in range(400):
        yp=rng.permutation(y)
        try:null.append(roc_auc_score(yp,cross_val_predict(mdl,X,yp,cv=StratifiedKFold(5,shuffle=True,random_state=1),method="predict_proba")[:,1]))
        except Exception:pass
    null=np.array(null);print(f"  {tag:44} AUC={a0:.3f} p={(1+np.sum(null>=a0))/(1+len(null)):.3f}")
print("\n=== COMPOSITES (agnósticas, SIN longitud/entropía cruda) ===")
comp(["min_prob","consist"],"min_prob+consist (base)")
comp(["min_prob","consist","imgdep_lowp","blank_sim"],"+imgdep_lowp+blank")
comp(["min_prob","p05","fraclow","consist","consist_min","consist_pw","imgdep","imgdep_lowp","blank_sim"],"rico agnóstico (L1)")
comp(["min_prob","p05","fraclow","consist","consist_min","consist_pw","imgdep","imgdep_lowp","blank_sim"],"rico agnóstico (GBM)",gbm=True)
print("\n=== + señal model-specific (occ_drop faithfulness) ===")
comp(["min_prob","consist","occ"],"min_prob+consist+occ")
comp(["min_prob","p05","fraclow","consist","consist_min","consist_pw","imgdep","imgdep_lowp","blank_sim","occ"],"rico + occ (L1)")
