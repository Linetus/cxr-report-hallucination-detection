"""¿Multiplicativa (AND) mejor que aditiva/logística? min_prob + consist en focal."""
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
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
LAB=json.load(open(SC+r"\labels.json")); recs=[json.loads(l) for l in open(SC+r"\allsignals_openi.jsonl",encoding="utf-8")]
def ng(t,n=4): t=t.lower(); return set(t[i:i+n] for i in range(max(len(t)-n+1,1)))
def sim(a,b):
    A,B=ng(a),ng(b);u=len(A|B);return len(A&B)/u if u else 1.0
FOC=set("CXR1001_IM-0004-1002 CXR1016_IM-0013-2001 CXR1019_IM-0015-2001 CXR103_IM-0023-2001 CXR1030_IM-0024-1001 CXR1030_IM-0024-2001 CXR1035_IM-0028-2001 CXR1044_IM-0036-1001 CXR1047_IM-0036-1001 CXR1047_IM-0036-2001 CXR1000_IM-0003-1001 CXR1000_IM-0003-3001 CXR1001_IM-0004-1001 CXR1003_IM-0005-2002 CXR101_IM-0011-2001 CXR101_IM-0011-4004 CXR1012_IM-0013-1001 CXR1014_IM-0013-1001 CXR1014_IM-0013-2001 CXR1015_IM-0001-1001 CXR1015_IM-0001-2001 CXR1015_IM-0013-1001 CXR1015_IM-0013-2001 CXR1016_IM-0013-1001 CXR1021_IM-0017-1001-0001 CXR1034_IM-0028-1001 CXR1037_IM-0029-1001 CXR1043_IM-0035-0001-0001 CXR1043_IM-0035-0001-0002 CXR1050_IM-0038-4004 CXR1053_IM-0040-1001 CXR1053_IM-0040-3003 CXR105_IM-0037-1001 CXR1022_IM-0017-2001".split())
mp,cs,y=[],[],[]
for r in recs:
    if r["id"] not in LAB or r["id"] not in FOC: continue
    mp.append(min(r["probs"])); sm=r.get("samples",[]); cs.append(np.mean([sim(r["gen"],s) for s in sm]) if sm else 1.0); y.append(LAB[r["id"]])
mp=np.array(mp,float); cs=np.array(cs,float); y=np.array(y)
# orientar a P(alucina) por PERCENTIL (1 = más sospechoso): min_prob bajo y consist bajo => alucina
hu=1-rankdata(mp)/len(mp); hc=1-rankdata(cs)/len(cs)   # en [0,1], 0.5=mediana
print(f"focal n={len(y)} alucina={int(y.sum())}")
def auc(v): a=roc_auc_score(y,v); return max(a,1-a)
print(f"\nmin_prob solo      AUC={auc(hu):.3f}")
print(f"consist solo       AUC={auc(hc):.3f}")
print(f"SUMA (hu+hc)       AUC={auc(hu+hc):.3f}   (aditiva)")
print(f"PRODUCTO (hu*hc)   AUC={auc(hu*hc):.3f}   (AND multiplicativa)")
print(f"MEDIA GEOM sqrt    AUC={auc(np.sqrt(hu*hc)):.3f}")
print(f"AND-suave (hu*hc con umbral 0.5): potencia solo si ambas>0.5")
both=((hu>0.5)&(hc>0.5)); print(f"   casos con AMBAS>0.5: {both.sum()} -> alucina={y[both].mean():.2f} ({y[both].sum()}/{both.sum()})")
# logística vs logística+interacción (OOF)
def oof(X):
    lg=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight="balanced",max_iter=2000))
    return roc_auc_score(y,cross_val_predict(lg,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1])
X2=np.column_stack([mp,cs]); Xint=np.column_stack([mp,cs,hu*hc])
print(f"\nLOGÍSTICA 2-feat        OOF AUC={oof(X2):.3f}")
print(f"LOGÍSTICA + interacción OOF AUC={oof(Xint):.3f}")
# alta precisión: nº marcados a precisión 100%
def maxrec(score):
    order=np.argsort(-score); best=0
    for k in range(1,len(order)+1):
        idx=order[:k]
        if y[idx].mean()==1.0: best=k
    return best
print(f"\nMarcados a PRECISIÓN 100% (recall):  suma={maxrec(hu+hc)}  producto={maxrec(hu*hc)}  (de {int(y.sum())} alucinas)")
