"""CALIBRACIÓN de la fórmula de detección de alucinaciones (Open-I, correcto vs alucina).
   OOF ranking -> Platt -> P(alucinación) con curva de fiabilidad + Brier + IC por bootstrap."""
import os,sys,json,math
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
LAB=json.load(open(SC+r"\labels_final.json")); recs=[json.loads(l) for l in open(SC+r"\allsignals_openi.jsonl",encoding="utf-8")]
exec(open(SC+r"\_hset.py").read()) if os.path.exists(SC+r"\_hset.py") else None
# HALLUC set (42) desde final_n800
HALLUC=set("CXR1001_IM-0004-1002 CXR1016_IM-0013-2001 CXR1019_IM-0015-2001 CXR103_IM-0023-2001 CXR1030_IM-0024-1001 CXR1030_IM-0024-2001 CXR1035_IM-0028-2001 CXR1044_IM-0036-1001 CXR1047_IM-0036-1001 CXR1047_IM-0036-2001 CXR1058_IM-0041-2001 CXR1059_IM-0041-1001 CXR1064_IM-0045-1001 CXR1074_IM-0054-2001 CXR1076_IM-0054-3001 CXR1094_IM-0065-2001 CXR1104_IM-0071-1001 CXR1104_IM-0071-3001 CXR111_IM-0076-1002 CXR1110_IM-0076-2001 CXR1111_IM-0077-1001 CXR1112_IM-0078-2001 CXR1128_IM-0084-6001 CXR1133_IM-0090-1002 CXR1140_IM-0096-1002 CXR116_IM-0107-2001 CXR1187_IM-0126-2001 CXR1197_IM-0131-1002 CXR1229_IM-0152-2001 CXR1230_IM-0154-0001 CXR1237_IM-0159-1002 CXR1250_IM-0169-2001 CXR1255_IM-0172-1001-0001 CXR1255_IM-0172-1001-0002 CXR1270_IM-0181-2001 CXR1284_IM-0188-2001 CXR1288_IM-0189-2001 CXR1303_IM-0199-2001-0001 CXR1317_IM-0205-1001 CXR1336_IM-0216-2001 CXR1345_IM-0223-1001 CXR1360_IM-0234-0001-0001".split())
def ng(t,n=4): t=t.lower(); return set(t[i:i+n] for i in range(max(len(t)-n+1,1)))
def sim(a,b):
    A,B=ng(a),ng(b);u=len(A|B);return len(A&B)/u if u else 1.0
rows=[]
for r in recs:
    if r["id"] not in LAB: continue
    if LAB[r["id"]]==1 and r["id"] not in HALLUC: continue  # excluir misses (no detectables)
    p=np.array(r["probs"]); e=np.array(r["ents"]); vm=np.array(r["vis_mass"]); smp=r.get("samples",[])
    lo=np.argsort(p)[:max(3,len(p)//5)]; pw=[sim(smp[i],smp[j]) for i in range(len(smp)) for j in range(i+1,len(smp))]
    rows.append(dict(y=(1 if r["id"] in HALLUC else 0),ntok=len(p),min_prob=float(p.min()),p05=float(np.percentile(p,5)),
        fraclow=float((p<0.3).mean()),max_ent=float(e.max()),consist=float(np.mean([sim(r["gen"],s) for s in smp])) if smp else 1.0,
        consist_pw=float(np.mean(pw)) if pw else 1.0,imgdep_lowp=float(vm[lo].mean()),
        blank_sim=float(sim(r["gen"],r.get("blank_gen",""))) if r.get("blank_gen") else 0.0))
y=np.array([r["y"] for r in rows]); col=lambda f:np.array([r[f] for r in rows],float); nt=col("ntok")
me=col("max_ent"); xx=nt-nt.mean(); sl=(xx*(me-me.mean())).sum()/((xx*xx).sum()+1e-9); resme=me-(me.mean()+sl*(nt-nt.mean()))
for r,v in zip(rows,resme): r["res_maxent"]=float(v)
RICH=["min_prob","p05","fraclow","consist","consist_pw","imgdep_lowp","blank_sim","res_maxent"]
X=np.column_stack([col(f) for f in RICH])
lg=make_pipeline(StandardScaler(),LogisticRegression(C=0.5,penalty="l1",solver="liblinear",class_weight="balanced"))
score=cross_val_predict(lg,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1]
print(f"correcto={int((y==0).sum())} alucina={int(y.sum())} (prevalencia {y.mean():.1%})  AUC ranking={roc_auc_score(y,score):.3f}")
# Platt: calibrar score OOF -> P(alucina) reflejando prevalencia real (SIN balanceo)
cal=LogisticRegression(solver="liblinear").fit(score.reshape(-1,1),y)
P=cal.predict_proba(score.reshape(-1,1))[:,1]
print(f"Brier={brier_score_loss(y,P):.4f} (más bajo mejor; base={y.mean()*(1-y.mean()):.4f})")
print("\nCURVA DE FIABILIDAD (bin de P -> tasa real de alucinación):")
for a,b in [(0,0.05),(0.05,0.1),(0.1,0.2),(0.2,0.4),(0.4,1.01)]:
    m=(P>=a)&(P<b)
    if m.sum(): print(f"  P∈[{a:.2f},{b:.2f}): n={m.sum():3d}  P_media={P[m].mean():.2f}  tasa_real={y[m].mean():.2f}")
# IC por bootstrap del AUC
rng=np.random.default_rng(0); aucs=[]
for _ in range(2000):
    idx=rng.integers(0,len(y),len(y))
    if len(set(y[idx]))>1: aucs.append(roc_auc_score(y[idx],score[idx]))
print(f"\nAUC alucinación = {roc_auc_score(y,score):.3f}  IC95% bootstrap = [{np.percentile(aucs,2.5):.3f}, {np.percentile(aucs,97.5):.3f}]")
# ejemplos
o=np.argsort(-P); print("\nEjemplos (P calibrada de alucinación, etiqueta real):")
print("  más sospechosos:", [(round(float(P[i]),2),int(y[i])) for i in o[:5]])
print("  más fiables:    ", [(round(float(P[i]),2),int(y[i])) for i in o[-5:]])
