"""OOD-MAIRA (Open-I): correlaciona etiquetas G-EVAL con min_prob-contenido. Cierra el 2x2."""
import os,sys,json
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
labD={}; lab={}
for line in open(SC+r"\maira2_openi_geval_labels.jsonl",encoding="utf-8"):
    line=line.strip()
    if not line: continue
    d=json.loads(line); labD[d["id"]]=d; lab[d["id"]]=d["label"]
recs={json.loads(l)["ImageID"]:json.loads(l) for l in open(SC+r"\maira2_openi.jsonl",encoding="utf-8")}
def ng(t,n=4): t=t.lower(); return set(t[i:i+n] for i in range(max(len(t)-n+1,1)))
def sim(a,b):
    A,B=ng(a),ng(b); u=len(A|B); return len(A&B)/u if u else 1.0
def rankavg(x):
    o=np.argsort(x,kind="mergesort"); r=np.empty(len(x)); r[o]=np.arange(1,len(x)+1); xs=x[o]; i=0
    while i<len(x):
        j=i
        while j+1<len(x) and xs[j+1]==xs[i]: j+=1
        if j>i:
            av=(r[o[i]]+r[o[j]])/2
            for k in range(i,j+1): r[o[k]]=av
        i=j+1
    return r
def auc(sc,lb):
    lb=np.asarray(lb); sc=np.asarray(sc,float); npos=lb.sum(); nneg=len(lb)-npos
    if npos==0 or nneg==0: return float("nan")
    return (rankavg(sc)[lb==1].sum()-npos*(npos+1)/2)/(npos*nneg)
def auc_ci(sc,lb,B=2000,seed=0):
    rng=np.random.default_rng(seed); lb=np.asarray(lb); sc=np.asarray(sc,float)
    pos=np.where(lb==1)[0]; neg=np.where(lb==0)[0]
    if len(pos)<3 or len(neg)<3: return auc(sc,lb),float('nan'),float('nan')
    a=auc(sc,lb); bs=[auc(sc[np.r_[rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)]],
        np.r_[np.ones(len(pos)),np.zeros(len(neg))]) for _ in range(B)]
    return a,*np.percentile(bs,[2.5,97.5])
rows=[]
for iid,t in lab.items():
    r=recs.get(iid)
    if r is None: continue
    p=np.array(r["probs"],float); smp=r.get("samples",[])
    rows.append(dict(typ=t,mp_full=float(p.min()),mp=float(p[1:].min()) if len(p)>3 else float(p.min()),
        consist=float(np.mean([sim(r["gen"],s) for s in smp])) if smp else 1.0,
        finding=labD[iid].get("finding","")))
typ=np.array([x["typ"] for x in rows])
nH=int((typ=="halluc").sum()); nM=int((typ=="miss").sum()); nC=int((typ=="correct").sum())
print(f"OOD-MAIRA Open-I (G-EVAL) n={len(rows)}: alucina={nH} miss={nM} correcto={nC}  tasa={nH/len(rows):.0%}")
mp=np.array([x["mp"] for x in rows]); mpf=np.array([x["mp_full"] for x in rows]); cs=np.array([x["consist"] for x in rows])
m=(typ=="halluc")|(typ=="correct"); y=(typ[m]=="halluc").astype(int)
print(f"\n=== AUC(ALUCINACIÓN vs CORRECTO)  n_pos={int(y.sum())} n_neg={int((y==0).sum())} ===")
for name,sc in [("min_prob COMPLETO",-mpf[m]),("min_prob CONTENIDO",-mp[m]),("1-consist",(1-cs)[m])]:
    a,lo,hi=auc_ci(sc,y); print(f"  {name:20s} AUC={a:.3f} IC[{lo:.3f},{hi:.3f}]")
if nH: print(f"  medias min_prob-contenido: correcto={mp[typ=='correct'].mean():.3f} alucina={mp[typ=='halluc'].mean():.3f}")
foc=[x for x in rows if not (x["typ"]=="halluc" and "cardiomeg" in x["finding"].lower())]
tf=np.array([x["typ"] for x in foc]); mpf2=np.array([x["mp"] for x in foc])
mf=(tf=="halluc")|(tf=="correct"); yf=(tf[mf]=="halluc").astype(int)
if yf.sum()>=4:
    a,lo,hi=auc_ci(-mpf2[mf],yf); print(f"\nFOCAL-only (sin cardiomegalia): AUC={a:.3f} IC[{lo:.3f},{hi:.3f}] (n_halluc={int(yf.sum())})")
print("\n=== 2x2 FINAL (min_prob-contenido) ===")
print("               in-distribution      OOD")
print("LLaVA-Rad          0.90 (Open-I)     0.61 (PadChest)")
print(f"MAIRA-2            0.80 (PadChest)   {auc(-mp[m],y):.2f} (Open-I) <- ESTE")
