import os,sys,json,math
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from halluc_vs_miss import HALLUC,MISS
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
    rows.append(dict(y=LAB[r["id"]],typ=("correct" if LAB[r["id"]]==0 else ("halluc" if r["id"] in HALLUC else "miss")),
        ntok=len(p),min_prob=float(p.min()),max_ent=float(e.max()),consist_pw=float(np.mean(pw)) if pw else 1.0))
y=np.array([r["y"] for r in rows]); typ=np.array([r["typ"] for r in rows]); nt=np.array([r["ntok"] for r in rows],float)
me=np.array([r["max_ent"] for r in rows]); x=nt-nt.mean(); sl=(x*(me-me.mean())).sum()/((x*x).sum()+1e-9)
resme=me-(me.mean()+sl*(nt-nt.mean()))
X=np.column_stack([resme,[r["min_prob"] for r in rows],[r["consist_pw"] for r in rows]])
lg=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight="balanced",solver="liblinear",max_iter=2000))
prob=cross_val_predict(lg,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1]
pc=prob[typ=="correct"]; ph=prob[typ=="halluc"]; pm=prob[typ=="miss"]
NC,NH,NM=len(pc),len(ph),len(pm)
print(f"correcto={NC} alucina={NH} miss={NM}")
print(f"AUC alucina-vs-correcto={roc_auc_score(np.r_[np.zeros(NC),np.ones(NH)],np.r_[pc,ph]):.3f}")
def rep(thr,tag):
    tp=int((ph>=thr).sum()); fp=int((pc>=thr).sum()); mm=int((pm>=thr).sum())
    prec=tp/(tp+fp) if tp+fp else 0
    print(f"[{tag}] umbral={thr:.2f}: alucina {tp}/{NH} (recall {tp/NH:.0%}) · falsas alarmas {fp}/{NC} ({fp/NC:.0%}) · precisión {prec:.0%} · misses de propina {mm}/{NM} ({mm/NM:.0%})")
# Youden manual sobre correcto vs alucina
ths=np.unique(np.r_[pc,ph]); best=None
for t in ths:
    tpr=(ph>=t).mean(); fpr=(pc>=t).mean()
    if best is None or tpr-fpr>best[0]: best=(tpr-fpr,t)
rep(best[1],"Youden")
# especificidad 90%: umbral que deja pasar 10% de correctos
rep(np.quantile(pc,0.90),"Especificidad ~90% (revisar ~10%)")
# alta precisión: 0 falsos+
t_hp=None
for t in sorted(ths,reverse=True):
    if (pc>=t).sum()==0 and (ph>=t).sum()>0: t_hp=t; break
rep(t_hp if t_hp else 1.0,"Alta precisión (0 falsas alarmas)")
