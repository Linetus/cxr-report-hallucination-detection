"""¿La fórmula detecta igual ALUCINACIONES (falso+) que MISSES (falso-)? Hipótesis:
   los misses son 'normal' con alta confianza -> no detectables por incertidumbre."""
import os,sys,json,math
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
LAB=json.load(open(SC+r"\labels_full.json")); recs=[json.loads(l) for l in open(SC+r"\allsignals_openi.jsonl",encoding="utf-8")]
HALLUC={"CXR1001_IM-0004-1002","CXR1016_IM-0013-2001","CXR1019_IM-0015-2001","CXR103_IM-0023-2001","CXR1030_IM-0024-1001","CXR1030_IM-0024-2001","CXR1035_IM-0028-2001","CXR1044_IM-0036-1001","CXR1047_IM-0036-1001","CXR1047_IM-0036-2001","CXR1058_IM-0041-2001","CXR1059_IM-0041-1001","CXR1064_IM-0045-1001","CXR1074_IM-0054-2001","CXR1076_IM-0054-3001","CXR1094_IM-0065-2001","CXR1104_IM-0071-1001","CXR1104_IM-0071-3001","CXR111_IM-0076-1002","CXR1110_IM-0076-2001","CXR1111_IM-0077-1001","CXR1112_IM-0078-2001","CXR1128_IM-0084-6001","CXR1133_IM-0090-1002","CXR1140_IM-0096-1002","CXR116_IM-0107-2001","CXR1187_IM-0126-2001","CXR1197_IM-0131-1002","CXR1229_IM-0152-2001","CXR1230_IM-0154-0001","CXR1237_IM-0159-1002","CXR1250_IM-0169-2001","CXR1255_IM-0172-1001-0001","CXR1255_IM-0172-1001-0002","CXR1270_IM-0181-2001"}
MISS={"CXR1000_IM-0003-2001","CXR1013_IM-0013-1001","CXR1013_IM-0013-2001","CXR1021_IM-0017-1001-0002","CXR1072_IM-0052-1001-0001","CXR1072_IM-0052-1001-0002","CXR1111_IM-0077-4004","CXR1117_IM-0079-2001","CXR1125_IM-0082-2001","CXR1130_IM-0087-1001","CXR1130_IM-0087-3003","CXR1152_IM-0103-2001","CXR1157_IM-0106-2001","CXR1174_IM-0118-1001","CXR1174_IM-0118-2001","CXR1193_IM-0129-2001","CXR1202_IM-0136-2001","CXR1226_IM-0150-1001","CXR1226_IM-0150-1002","CXR1230_IM-0154-1001","CXR1230_IM-0154-2001","CXR1233_IM-0157-1001","CXR1245_IM-0167-1001"}
def ng(t,n=4): t=t.lower(); return set(t[i:i+n] for i in range(max(len(t)-n+1,1)))
def sim(a,b):
    A,B=ng(a),ng(b);u=len(A|B);return len(A&B)/u if u else 1.0
rows=[]
for r in recs:
    if r["id"] not in LAB: continue
    p=np.array(r["probs"]); e=np.array(r["ents"]); smp=r.get("samples",[])
    pw=[sim(smp[i],smp[j]) for i in range(len(smp)) for j in range(i+1,len(smp))]
    typ="correct" if LAB[r["id"]]==0 else ("halluc" if r["id"] in HALLUC else ("miss" if r["id"] in MISS else "err?"))
    rows.append(dict(id=r["id"],y=LAB[r["id"]],typ=typ,ntok=len(p),min_prob=float(p.min()),max_ent=float(e.max()),
        consist_pw=float(np.mean(pw)) if pw else 1.0))
y=np.array([r["y"] for r in rows]); nt=np.array([r["ntok"] for r in rows],float)
me=np.array([r["max_ent"] for r in rows]); x=nt-nt.mean(); sl=(x*(me-me.mean())).sum()/((x*x).sum()+1e-9)
resme=me-(me.mean()+sl*(nt-nt.mean()))
X=np.column_stack([resme,[r["min_prob"] for r in rows],[r["consist_pw"] for r in rows]])
lg=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight="balanced",solver="liblinear",max_iter=2000))
prob=cross_val_predict(lg,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1]
typ=np.array([r["typ"] for r in rows])
print("tipos:",{t:int((typ==t).sum()) for t in ["correct","halluc","miss","err?"]})
mp=np.array([r["min_prob"] for r in rows])
def grp(t): return prob[typ==t]
print(f"\nSCORE medio de la fórmula:  correcto={grp('correct').mean():.2f}  ALUCINA={grp('halluc').mean():.2f}  MISS={grp('miss').mean():.2f}")
print(f"min_prob medio:             correcto={mp[typ=='correct'].mean():.3f}  ALUCINA={mp[typ=='halluc'].mean():.3f}  MISS={mp[typ=='miss'].mean():.3f}")
print(f"entropía-resid media:       correcto={resme[typ=='correct'].mean():+.2f}  ALUCINA={resme[typ=='halluc'].mean():+.2f}  MISS={resme[typ=='miss'].mean():+.2f}")
c=typ=='correct'
def auc_vs_correct(t):
    m=c|(typ==t); return roc_auc_score((typ[m]==t).astype(int),prob[m])
print(f"\nAUC(ALUCINA vs correcto) = {auc_vs_correct('halluc'):.3f}   (n_alucina={int((typ=='halluc').sum())})")
print(f"AUC(MISS    vs correcto) = {auc_vs_correct('miss'):.3f}   (n_miss={int((typ=='miss').sum())})")
