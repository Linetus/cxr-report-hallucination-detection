"""Clasificación SOLO alucinaciones (correcto vs alucina). Operating points + misses de propina."""
import os,sys,json,math
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
LAB=json.load(open(SC+r"\labels_full.json")); recs=[json.loads(l) for l in open(SC+r"\allsignals_openi.jsonl",encoding="utf-8")]
exec(open(SC+r"\_hm_sets.py").read()) if os.path.exists(SC+r"\_hm_sets.py") else None
from halluc_vs_miss import HALLUC,MISS  # reutiliza los sets
def ng(t,n=4): t=t.lower(); return set(t[i:i+n] for i in range(max(len(t)-n+1,1)))
def sim(a,b):
    A,B=ng(a),ng(b);u=len(A|B);return len(A&B)/u if u else 1.0
rows=[]
for r in recs:
    if r["id"] not in LAB: continue
    p=np.array(r["probs"]); e=np.array(r["ents"]); smp=r.get("samples",[])
    pw=[sim(smp[i],smp[j]) for i in range(len(smp)) for j in range(i+1,len(smp))]
    typ="correct" if LAB[r["id"]]==0 else ("halluc" if r["id"] in HALLUC else "miss")
    rows.append(dict(typ=typ,ntok=len(p),min_prob=float(p.min()),max_ent=float(e.max()),
        consist_pw=float(np.mean(pw)) if pw else 1.0))
nt=np.array([r["ntok"] for r in rows],float); me=np.array([r["max_ent"] for r in rows])
x=nt-nt.mean(); sl=(x*(me-me.mean())).sum()/((x*x).sum()+1e-9); resme=me-(me.mean()+sl*(nt-nt.mean()))
Xall=np.column_stack([resme,[r["min_prob"] for r in rows],[r["consist_pw"] for r in rows]])
typ=np.array([r["typ"] for r in rows])
# subconjunto correcto+alucina
sub=(typ=="correct")|(typ=="halluc"); X=Xall[sub]; ysub=(typ[sub]=="halluc").astype(int)
lg=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight="balanced",solver="liblinear",max_iter=2000))
prob=cross_val_predict(lg,X,ysub,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1]
Ns=len(ysub); H=int(ysub.sum())
print(f"CORRECTO vs ALUCINACIÓN: n={Ns} (correcto={Ns-H}, alucina={H})  AUC={roc_auc_score(ysub,prob):.3f}\n")
# umbral para score de TODAS las filas (para ver misses): reajusto la logística a todo el sub y aplico a misses
lg.fit(X,ysub); prob_all=lg.predict_proba(Xall)[:,1]
miss_scores=prob_all[typ=="miss"]
def report(thr,tag):
    fl=prob>=thr; tp=int((fl&(ysub==1)).sum()); fp=int((fl&(ysub==0)).sum())
    tn=int((~fl&(ysub==0)).sum()); fn=int((~fl&(ysub==1)).sum())
    prec=tp/(tp+fp) if tp+fp else 0; rec=tp/H; spec=tn/(tn+fp)
    missrec=float((miss_scores>=thr).mean())
    print(f"[{tag}] umbral={thr:.2f}")
    print(f"   alucina detectadas={tp}/{H} (recall {rec:.0%}) · falsas alarmas={fp} · precisión {prec:.0%} · especificidad {spec:.0%}")
    print(f"   (de propina: misses también marcados = {missrec:.0%} — con menos seguridad)\n")
fpr,tpr,ths=roc_curve(ysub,prob); j=np.argmax(tpr-fpr); report(ths[j],"Youden")
# especificidad 90% (marcar ~10% para revision)
i90=np.argmin(np.abs((1-fpr)-0.90)); report(ths[i90],"Especificidad 90% (revisar ~10%)")
order=np.argsort(-prob); best=0
for k in range(1,Ns+1):
    if ysub[order[:k]].mean()==1.0: best=k
report((prob[order[best-1]] if best else 1.0)-1e-9,"Alta precisión (0 falsos+)")
