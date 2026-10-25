"""Fórmula AGNÓSTICA sobre labels ampliadas (n=335). Full + subconjunto gen-anormal."""
import os,sys,json,math,re
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
LAB=json.load(open(SC+r"\labels_expanded.json")); recs=[json.loads(l) for l in open(SC+r"\allsignals_openi.jsonl",encoding="utf-8")]
def ng(t,n=4): t=t.lower(); return set(t[i:i+n] for i in range(max(len(t)-n+1,1)))
def sim(a,b):
    A,B=ng(a),ng(b);u=len(A|B);return len(A&B)/u if u else 1.0
ABN=["opacit","consolidat","effusion","edema","cardiomeg","enlarg","nodul","mass","atelectas","infiltrat","pneumo","fibros","interstitial","congestion","tortuos","granuloma","emphysem","hyperinflat","hyperexpand"]
NEG=re.compile(r"\b(no|without|clear|normal|free of|negative|unremarkable|within normal|intact|grossly|resolved)\b")
def gen_abn(t):
    tl=t.lower()
    for s in ABN:
        for m in re.finditer(s,tl):
            if NEG.search(tl[max(0,m.start()-45):m.start()]) is None: return True
    return False
rows=[]
for r in recs:
    if r["id"] not in LAB: continue
    p=np.array(r["probs"]); e=np.array(r["ents"]); gen=r["gen"]; smp=r.get("samples",[]); bg=r.get("blank_gen","")
    consist=float(np.mean([sim(gen,s) for s in smp])) if smp else 1.0
    rows.append(dict(y=LAB[r["id"]],min_prob=float(p.min()),max_ent=float(e.max()),mean_ent=float(e.mean()),
        consist=consist,blank_sim=float(sim(gen,bg)) if bg else 0.0,abn=gen_abn(gen)))
y=np.array([r["y"] for r in rows]); FEATS=["min_prob","max_ent","mean_ent","consist","blank_sim"]
def col(f): return np.array([r[f] for r in rows])
print(f"n={len(rows)}  alucina={int(y.sum())}")
def run(mask,tag):
    yy=y[mask]
    if yy.sum()<5 or (yy==0).sum()<5: print(f"{tag}: n insuficiente"); return
    print(f"\n=== {tag} (n={mask.sum()}, alucina={int(yy.sum())}) ===")
    for f in FEATS:
        v=col(f)[mask]; a=roc_auc_score(yy,v); print(f"  {f:11} AUC={max(a,1-a):.3f}")
    X=np.column_stack([col(f)[mask] for f in FEATS])
    lg=make_pipeline(StandardScaler(),LogisticRegression(C=0.5,penalty="l1",solver="liblinear",class_weight="balanced"))
    pr=cross_val_predict(lg,X,yy,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1]; a0=roc_auc_score(yy,pr)
    rng=np.random.default_rng(0);null=[]
    for _ in range(500):
        yp=rng.permutation(yy)
        try: null.append(roc_auc_score(yp,cross_val_predict(lg,X,yp,cv=StratifiedKFold(5,shuffle=True,random_state=1),method="predict_proba")[:,1]))
        except Exception: pass
    null=np.array(null); print(f"  COMPOSITE AGNÓSTICO OOF AUC={a0:.3f}  p={(1+np.sum(null>=a0))/(1+len(null)):.3f}")
run(np.ones(len(y),bool),"FULL")
run(np.array([r["abn"] for r in rows]),"GEN-ANORMAL (control de régimen)")
