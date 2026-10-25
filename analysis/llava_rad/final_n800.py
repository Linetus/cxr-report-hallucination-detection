"""ESTUDIO DEFINITIVO n=800: fórmula multi-señal + disociación alucina/miss + operating points."""
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
LABF=json.load(open(SC+r"\labels_full.json")); recs=[json.loads(l) for l in open(SC+r"\allsignals_openi.jsonl",encoding="utf-8")]
SUB2_H=["CXR1284_IM-0188-2001","CXR1288_IM-0189-2001","CXR1303_IM-0199-2001-0001","CXR1317_IM-0205-1001","CXR1336_IM-0216-2001","CXR1345_IM-0223-1001","CXR1360_IM-0234-0001-0001"]
SUB2_M=["CXR1282_IM-0188-1001","CXR1289_IM-0189-2001","CXR1303_IM-0199-2001-0002","CXR1326_IM-0210-0001-0001","CXR1326_IM-0210-0001-0002","CXR1329_IM-0211-2001","CXR1375_IM-0241-1001","CXR1375_IM-0241-3001","CXR1375_IM-0241-4001"]
SUB2=set(SUB2_H)|set(SUB2_M)
# labels_final: labels_full + 241 restantes (0 salvo SUB2)
LAB=dict(LABF)
for r in recs:
    if r["id"] in LABF: continue
    LAB[r["id"]]=1 if r["id"] in SUB2 else 0
json.dump(LAB,open(SC+r"\labels_final.json","w"))
HALLUC=set("CXR1001_IM-0004-1002 CXR1016_IM-0013-2001 CXR1019_IM-0015-2001 CXR103_IM-0023-2001 CXR1030_IM-0024-1001 CXR1030_IM-0024-2001 CXR1035_IM-0028-2001 CXR1044_IM-0036-1001 CXR1047_IM-0036-1001 CXR1047_IM-0036-2001 CXR1058_IM-0041-2001 CXR1059_IM-0041-1001 CXR1064_IM-0045-1001 CXR1074_IM-0054-2001 CXR1076_IM-0054-3001 CXR1094_IM-0065-2001 CXR1104_IM-0071-1001 CXR1104_IM-0071-3001 CXR111_IM-0076-1002 CXR1110_IM-0076-2001 CXR1111_IM-0077-1001 CXR1112_IM-0078-2001 CXR1128_IM-0084-6001 CXR1133_IM-0090-1002 CXR1140_IM-0096-1002 CXR116_IM-0107-2001 CXR1187_IM-0126-2001 CXR1197_IM-0131-1002 CXR1229_IM-0152-2001 CXR1230_IM-0154-0001 CXR1237_IM-0159-1002 CXR1250_IM-0169-2001 CXR1255_IM-0172-1001-0001 CXR1255_IM-0172-1001-0002 CXR1270_IM-0181-2001".split())|set(SUB2_H)
MISS=set("CXR1000_IM-0003-2001 CXR1013_IM-0013-1001 CXR1013_IM-0013-2001 CXR1021_IM-0017-1001-0002 CXR1072_IM-0052-1001-0001 CXR1072_IM-0052-1001-0002 CXR1111_IM-0077-4004 CXR1117_IM-0079-2001 CXR1125_IM-0082-2001 CXR1130_IM-0087-1001 CXR1130_IM-0087-3003 CXR1152_IM-0103-2001 CXR1157_IM-0106-2001 CXR1174_IM-0118-1001 CXR1174_IM-0118-2001 CXR1193_IM-0129-2001 CXR1202_IM-0136-2001 CXR1226_IM-0150-1001 CXR1226_IM-0150-1002 CXR1230_IM-0154-1001 CXR1230_IM-0154-2001 CXR1233_IM-0157-1001 CXR1245_IM-0167-1001".split())|set(SUB2_M)
def ng(t,n=4): t=t.lower(); return set(t[i:i+n] for i in range(max(len(t)-n+1,1)))
def sim(a,b):
    A,B=ng(a),ng(b);u=len(A|B);return len(A&B)/u if u else 1.0
rows=[]
for r in recs:
    if r["id"] not in LAB: continue
    p=np.array(r["probs"]); e=np.array(r["ents"]); vm=np.array(r["vis_mass"]); smp=r.get("samples",[])
    lo=np.argsort(p)[:max(3,len(p)//5)]; pw=[sim(smp[i],smp[j]) for i in range(len(smp)) for j in range(i+1,len(smp))]
    rows.append(dict(id=r["id"],y=LAB[r["id"]],typ=("correct" if LAB[r["id"]]==0 else ("halluc" if r["id"] in HALLUC else ("miss" if r["id"] in MISS else "err?"))),
        ntok=len(p),min_prob=float(p.min()),p05=float(np.percentile(p,5)),fraclow=float((p<0.3).mean()),
        max_ent=float(e.max()),consist=float(np.mean([sim(r["gen"],s) for s in smp])) if smp else 1.0,
        consist_pw=float(np.mean(pw)) if pw else 1.0,imgdep_lowp=float(vm[lo].mean()),
        blank_sim=float(sim(r["gen"],r.get("blank_gen",""))) if r.get("blank_gen") else 0.0))
y=np.array([r["y"] for r in rows]); typ=np.array([r["typ"] for r in rows]); col=lambda f:np.array([r[f] for r in rows],float); nt=col("ntok")
me=col("max_ent"); xx=nt-nt.mean(); sl=(xx*(me-me.mean())).sum()/((xx*xx).sum()+1e-9); resme=me-(me.mean()+sl*(nt-nt.mean()))
for r,v in zip(rows,resme): r["res_maxent"]=float(v)
print(f"n={len(rows)}  errores={int(y.sum())}  tipos={ {t:int((typ==t).sum()) for t in ['correct','halluc','miss','err?']} }")
def oof(feats,mask=None):
    m=np.ones(len(y),bool) if mask is None else mask
    X=np.column_stack([col(f)[m] for f in feats]); yy=y[m]
    lg=make_pipeline(StandardScaler(),LogisticRegression(C=0.5,penalty="l1",solver="liblinear",class_weight="balanced"))
    return cross_val_predict(lg,X,yy,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1],yy
RICH=["min_prob","p05","fraclow","consist","consist_pw","imgdep_lowp","blank_sim","res_maxent"]
pr,_=oof(RICH); a0=roc_auc_score(y,pr)
rng=np.random.default_rng(0);null=np.array([roc_auc_score(rng.permutation(y),pr) for _ in range(1000)])
print(f"\nFÓRMULA multi-señal (rico+res-entropía) OOF AUC={a0:.3f}  p={(1+np.sum(null>=max(a0,1-a0)))/(1001):.3f}")
# disociación: usar el score OOF de la fórmula global
pc=pr[typ=="correct"]; ph=pr[typ=="halluc"]; pm=pr[typ=="miss"]
print(f"score fórmula: correcto={pc.mean():.2f} ALUCINA={ph.mean():.2f} MISS={pm.mean():.2f}")
print(f"AUC(ALUCINA vs correcto)={roc_auc_score(np.r_[np.zeros(len(pc)),np.ones(len(ph))],np.r_[pc,ph]):.3f}  (n_halluc={len(ph)})")
print(f"AUC(MISS vs correcto)   ={roc_auc_score(np.r_[np.zeros(len(pc)),np.ones(len(pm))],np.r_[pc,pm]):.3f}  (n_miss={len(pm)})")
# operating points correcto vs alucina
NC,NH,NM=len(pc),len(ph),len(pm)
ths=np.unique(np.r_[pc,ph]); best=max(ths,key=lambda t:(ph>=t).mean()-(pc>=t).mean())
for lab,t in [("Youden",best),("Espec 90%",np.quantile(pc,0.90))]:
    tp=int((ph>=t).sum());fp=int((pc>=t).sum());mm=int((pm>=t).sum())
    print(f"[{lab}] recall alucina {tp}/{NH} ({tp/NH:.0%}) · falsas alarmas {fp}/{NC} ({fp/NC:.0%}) · precisión {tp/(tp+fp) if tp+fp else 0:.0%} · misses de propina {mm}/{NM} ({mm/NM:.0%})")
