"""Fórmula de 3 señales (res_maxent + min_prob + consist_pw): ¿cuántos clasificamos bien?"""
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
def ng(t,n=4): t=t.lower(); return set(t[i:i+n] for i in range(max(len(t)-n+1,1)))
def sim(a,b):
    A,B=ng(a),ng(b);u=len(A|B);return len(A&B)/u if u else 1.0
rows=[]
for r in recs:
    if r["id"] not in LAB: continue
    p=np.array(r["probs"]); e=np.array(r["ents"]); smp=r.get("samples",[])
    pw=[sim(smp[i],smp[j]) for i in range(len(smp)) for j in range(i+1,len(smp))]
    rows.append(dict(y=LAB[r["id"]],ntok=len(p),min_prob=float(p.min()),max_ent=float(e.max()),
        consist_pw=float(np.mean(pw)) if pw else 1.0))
y=np.array([r["y"] for r in rows]); nt=np.array([r["ntok"] for r in rows],float)
me=np.array([r["max_ent"] for r in rows]); x=nt-nt.mean(); sl=(x*(me-me.mean())).sum()/((x*x).sum()+1e-9)
res_maxent=me-(me.mean()+sl*(nt-nt.mean()))
X=np.column_stack([res_maxent,[r["min_prob"] for r in rows],[r["consist_pw"] for r in rows]])
lg=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight="balanced",solver="liblinear",max_iter=2000))
prob=cross_val_predict(lg,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1]
N=len(y); P=int(y.sum()); Neg=N-P
print(f"n={N}  alucina={P} ({P/N:.0%})  correctos={Neg}")
print(f"AUC out-of-fold = {roc_auc_score(y,prob):.3f}\n")
def report(thr,tag):
    pred=(prob>=thr).astype(int)
    tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum())
    tn=int(((pred==0)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
    acc=(tp+tn)/N; prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0; spec=tn/(tn+fp)
    print(f"[{tag}] umbral={thr:.2f}")
    print(f"   BIEN CLASIFICADOS = {tp+tn}/{N} ({acc:.1%})   |   mal = {fp+fn} (FP={fp}, FN={fn})")
    print(f"   alucina detectadas TP={tp}/{P} (recall {rec:.0%})  ·  falsas alarmas FP={fp}  ·  precisión {prec:.0%}  ·  especificidad {spec:.0%}\n")
# umbral de Youden (máx sensibilidad+especificidad)
fpr,tpr,ths=roc_curve(y,prob); j=np.argmax(tpr-fpr)
report(ths[j],"Youden (equilibrio)")
report(0.5,"umbral 0.5")
# alta precisión: mayor recall con 0 falsos positivos
order=np.argsort(-prob); best=0
for k in range(1,N+1):
    if y[order[:k]].mean()==1.0: best=k
thr_hp=prob[order[best-1]] if best else 1.0
report(thr_hp-1e-9,"ALTA PRECISIÓN (0 falsos positivos)")
