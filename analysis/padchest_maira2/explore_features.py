"""Exploración de SEÑALES para mejorar la fórmula en TODOS los datasets.
   Señales agnósticas (disponibles en inferencia en cualquier modelo): probs, entropía, K muestras, blank.
   Mide AUC(alucina vs correcto) univariante y el valor INCREMENTAL sobre min_prob-contenido,
   buscando señales que generalicen (ayuden en Open-I + PadChest-LLaVA + MAIRA-2)."""
import os,sys,json
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
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
def auc_ci(sc,lb,B=1500,seed=0):
    rng=np.random.default_rng(seed); lb=np.asarray(lb); sc=np.asarray(sc,float)
    pos=np.where(lb==1)[0]; neg=np.where(lb==0)[0]
    if len(pos)<3 or len(neg)<3: return auc(sc,lb),float('nan'),float('nan')
    a=auc(sc,lb); bs=[auc(sc[np.r_[rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)]],
        np.r_[np.ones(len(pos)),np.zeros(len(neg))]) for _ in range(B)]
    return a,*np.percentile(bs,[2.5,97.5])

def feats(rec):
    """señales orientadas: MAYOR = más probable alucinación."""
    p=np.array(rec["probs"],float); e=np.array(rec["ents"],float); smp=rec.get("samples",[]) or []
    c=p[1:] if len(p)>3 else p                      # contenido (sin token apertura)
    lp=np.log(np.clip(p,1e-6,1))
    d={}
    d["cminp"]=-float(c.min())                      # min_prob contenido
    d["p05"]=-float(np.percentile(c,5))
    d["fraclow"]=float((c<0.3).mean())
    d["meanlp"]=-float(lp.mean())                   # verosimilitud media (baja=halluc)
    d["maxent"]=float(e.max())                      # (se residualiza luego)
    d["meanent"]=float(e.mean())
    d["iconsist"]=1-float(np.mean([sim(rec["gen"],s) for s in smp])) if smp else 0.0
    if len(smp)>=2:
        pw=[sim(smp[i],smp[j]) for i in range(len(smp)) for j in range(i+1,len(smp))]
        d["iconsist_pw"]=1-float(np.mean(pw))
    else: d["iconsist_pw"]=0.0
    bg=rec.get("blank_gen","")
    d["blanksim"]=float(sim(rec["gen"],bg)) if bg else 0.0   # dependencia imagen (alto=prior)
    d["_ntok"]=len(p)
    return d

def load(name):
    if name=="Open-I":
        recs=[json.loads(l) for l in open(SC+r"\allsignals_openi.jsonl",encoding="utf-8")]
        lab=json.load(open(SC+r"\openi_labels.json")); key="id"
    elif name=="PadChest-LLaVA":
        recs=[json.loads(l) for l in open(SC+r"\padchest_full.jsonl",encoding="utf-8")]
        lab={x["id"]:x["typ"] for x in json.load(open(SC+r"\padchest_rows.json",encoding="utf-8"))}; key="ImageID"
    else:  # MAIRA-2
        recs=[json.loads(l) for l in open(SC+r"\maira2_padchest.jsonl",encoding="utf-8")]
        gl=json.load(open(SC+r"\maira2_geval_labels_all.json",encoding="utf-8")); lab={k:v["label"] for k,v in gl.items()}; key="ImageID"
    rows=[]
    for r in recs:
        t=lab.get(r[key])
        if t in ("halluc","correct"): rows.append((t,feats(r)))
    return rows

FEATS=["cminp","p05","fraclow","meanlp","maxent_r","meanent","iconsist","iconsist_pw","blanksim"]
DS=["Open-I","PadChest-LLaVA","MAIRA-2"]
data={}
for name in DS:
    rows=load(name)
    y=np.array([1 if t=="halluc" else 0 for t,_ in rows])
    nt=np.array([f["_ntok"] for _,f in rows],float)
    # residualizar maxent por longitud (dentro del dataset)
    me=np.array([f["maxent"] for _,f in rows]); xx=nt-nt.mean()
    sl=(xx*(me-me.mean())).sum()/((xx*xx).sum()+1e-9); mer=me-(me.mean()+sl*(nt-nt.mean()))
    F={}
    for k in FEATS:
        if k=="maxent_r": F[k]=mer
        else: F[k]=np.array([f[k] for _,f in rows])
    data[name]=(y,F)
    print(f"{name}: n={len(y)} halluc={int(y.sum())} correcto={int((y==0).sum())}")

print("\n=== AUC UNIVARIANTE por señal (orientada: mayor=halluc) ===")
print(f"{'señal':13s}"+"".join(f"{d:>18s}" for d in DS))
for k in FEATS:
    cells=[]
    for name in DS:
        y,F=data[name]; a=auc(F[k],y); cells.append(f"{a:.2f}")
    print(f"{k:13s}"+"".join(f"{c:>18s}" for c in cells))

print("\n=== VALOR INCREMENTAL: min_prob-contenido + 1 señal (z-sum) ===")
print(f"{'+señal':13s}"+"".join(f"{d:>18s}" for d in DS))
def z(a): s=a.std(); return (a-a.mean())/(s if s>1e-9 else 1)
base={name:auc(data[name][1]["cminp"],data[name][0]) for name in DS}
print(f"{'(solo cminp)':13s}"+"".join(f"{base[d]:>18.2f}" for d in DS))
for k in FEATS:
    if k=="cminp": continue
    cells=[]
    for name in DS:
        y,F=data[name]; comp=z(F["cminp"])+z(F[k]); a=auc(comp,y); cells.append(f"{a:.2f}({a-base[name]:+.2f})")
    print(f"+{k:12s}"+"".join(f"{c:>18s}" for c in cells))

# composite propuesto: señales que generalizan
print("\n=== COMPOSITE propuesto (cminp + iconsist + maxent_r) con IC ===")
for name in DS:
    y,F=data[name]; comp=z(F["cminp"])+z(F["iconsist"])+z(F["maxent_r"])
    a,lo,hi=auc_ci(comp,y); b=base[name]
    print(f"  {name:15s} AUC={a:.3f} IC[{lo:.3f},{hi:.3f}]  (solo cminp {b:.3f})")
