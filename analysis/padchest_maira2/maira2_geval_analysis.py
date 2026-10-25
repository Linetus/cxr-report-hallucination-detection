"""Correlaciona las etiquetas G-EVAL (juez, no regex) de MAIRA-2 con las señales de incertidumbre.
   ¿Un modelo mejor + en-distribución recupera la detección de alucinaciones?"""
import os,sys,json
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
lab=json.load(open(SC+r"\maira2_geval_labels_all.json",encoding="utf-8"))
recs={json.loads(l)["ImageID"]:json.loads(l) for l in open(SC+r"\maira2_padchest.jsonl",encoding="utf-8")}
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
    pos=np.where(lb==1)[0]; neg=np.where(lb==0)[0]; a=auc(sc,lb); bs=[]
    if len(pos)==0 or len(neg)==0: return float("nan"),float("nan"),float("nan")
    for _ in range(B):
        idx=np.r_[rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)]; bs.append(auc(sc[idx],lb[idx]))
    return a,*np.percentile(bs,[2.5,97.5])
rows=[]
for iid,d in lab.items():
    r=recs.get(iid)
    if r is None: continue
    p=np.array(r["probs"],float); smp=r.get("samples",[])
    rows.append(dict(id=iid,typ=d["label"],finding=d.get("finding",""),
        mp_full=float(p.min()),mp=float(p[1:].min()) if len(p)>3 else float(p.min()),
        consist=float(np.mean([sim(r["gen"],s) for s in smp])) if smp else 1.0))
typ=np.array([x["typ"] for x in rows])
nH=int((typ=="halluc").sum()); nM=int((typ=="miss").sum()); nC=int((typ=="correct").sum())
print(f"MAIRA-2 (etiquetas G-EVAL) n={len(rows)}: alucina={nH} miss={nM} correcto={nC}  tasa aluc={nH/len(rows):.0%}")
print("(LLaVA-Rad OOD daba: alucina 33/381=9%, miss 130, AUC min_prob-contenido 0.61 / 0.48 completo)\n")
mp=np.array([x["mp"] for x in rows]); mpf=np.array([x["mp_full"] for x in rows]); cs=np.array([x["consist"] for x in rows])
def z(a): s=a.std(); return (a-a.mean())/(s if s>1e-9 else 1)
comp=z(-mp)+z(1-cs)
m=(typ=="halluc")|(typ=="correct"); y=(typ[m]=="halluc").astype(int)
print(f"=== AUC(ALUCINACIÓN vs CORRECTO)  n_pos={int(y.sum())} n_neg={int((y==0).sum())} ===")
for name,sc in [("min_prob COMPLETO",-mpf[m]),("min_prob CONTENIDO",-mp[m]),("1-consist",(1-cs)[m]),("compuesta",comp[m])]:
    a,lo,hi=auc_ci(sc,y); print(f"  {name:20s} AUC={a:.3f} IC[{lo:.3f},{hi:.3f}]")
if nH: print(f"  medias min_prob-contenido: correcto={mp[typ=='correct'].mean():.3f} alucina={mp[typ=='halluc'].mean():.3f}")
mm=(typ=="miss")|(typ=="correct"); ym=(typ[mm]=="miss").astype(int)
if ym.sum()>=4:
    a,lo,hi=auc_ci(comp[mm],ym); print(f"\n  AUC(MISS vs correcto) compuesta={a:.3f} IC[{lo:.3f},{hi:.3f}]  (esperado bajo: misses confiados)")
# focal-only: excluir alucinaciones de cardiomegalia (overcall perceptual)
foc=[x for x in rows if not (x["typ"]=="halluc" and "cardiomeg" in x["finding"].lower())]
tf=np.array([x["typ"] for x in foc]); mpf2=np.array([x["mp"] for x in foc])
mf=(tf=="halluc")|(tf=="correct"); yf=(tf[mf]=="halluc").astype(int)
if yf.sum()>=4:
    a,lo,hi=auc_ci(-mpf2[mf],yf); print(f"\nFOCAL-only (sin cardiomegalia) min_prob CONTENIDO: AUC={a:.3f} IC[{lo:.3f},{hi:.3f}] (n_halluc={int(yf.sum())})")

print("\ncasos alucinación (id corto :: finding):")
for x in rows:
    if x["typ"]=="halluc": print(f"  {x['id'][:12]} mp={x['mp']:.2f} :: {x['finding']}")
