"""RÉPLICA PadChest-GR: etiquetas OBJETIVAS de alucinación/miss (present/absent del radiólogo)
   + validación de la fórmula de incertidumbre. Todo numpy puro (evita crash BLAS Windows).
   AUC por Mann-Whitney, residualización y bootstrap manuales."""
import os,sys,json,re
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
ANN=json.load(open(r"./data/raw/padchest_gr/grounded_reports.json",encoding="utf-8"))
byimg={a["ImageID"]:a for a in ANN}
recs=[json.loads(l) for l in open(SC+r"\padchest_full.jsonl",encoding="utf-8")]

# ---- categorías focales (mismas que el extractor) + sinónimos por substring ----
FOCAL={"consolidation":["consolidat","airspace","alveolar"],"opacity":["opacit"],"nodule":["nodul"],
 "mass":["mass"],"effusion":["effusion"],"edema":["edema","congest"],"atelectasis":["atelectas","collapse"],
 "cardiomegaly":["cardiomeg","enlarg"],"pneumothorax":["pneumothorax"],
 "infiltrate":["infiltrat","interstitial pattern"],"pneumonia":["pneumonia"],"fibrosis":["fibros"]}
NEGSET={"no","not","without","free","negative","nor","never","clear","unremarkable","resolved","normal"}
SENTEND={".","!","?"}
def asserted_cats(toks):
    """Detecta hallazgo AFIRMADO sobre el flujo de tokens (el texto va pegado sin espacios).
       Un hallazgo cuenta si en su MISMA frase (hacia atrás hasta el '.') no hay cue de negación.
       Se SALTAN tokens vacíos / '\\n' (ajuste de línea a mitad de frase, no son fin de frase)."""
    spans=[]; s=""
    for t in toks: spans.append((len(s),len(s)+len(t))); s+=t
    tl=s.lower()
    def tokidx(pos):
        for i,(a,b) in enumerate(spans):
            if a<=pos<b: return i
        return len(spans)-1
    out=set()
    for c,syns in FOCAL.items():
        for syn in syns:
            hit=False
            for m in re.finditer(syn,tl):
                i0=tokidx(m.start()); neg=False
                for j in range(i0-1,-1,-1):
                    tj=toks[j].strip().lower()
                    if tj=="": continue              # vacío / salto de línea → saltar
                    if tj in SENTEND: break           # límite de frase real
                    if tj in NEGSET: neg=True; break
                if not neg: out.add(c); hit=True; break
            if hit: break
    return out
def cats_of_text(t):
    tl=t.lower(); return {c for c,syns in FOCAL.items() if any(s in tl for s in syns)}

# ---- similitud de texto (n-gram Jaccard, sin BLAS) ----
def ng(t,n=4): t=t.lower(); return set(t[i:i+n] for i in range(max(len(t)-n+1,1)))
def sim(a,b):
    A,B=ng(a),ng(b); u=len(A|B); return len(A&B)/u if u else 1.0

# ---- AUC Mann-Whitney + bootstrap (numpy puro) ----
def rankavg(x):
    o=np.argsort(x,kind="mergesort"); r=np.empty(len(x)); r[o]=np.arange(1,len(x)+1)
    # promediar empates
    x_s=x[o]; i=0
    while i<len(x):
        j=i
        while j+1<len(x) and x_s[j+1]==x_s[i]: j+=1
        if j>i:
            avg=(r[o[i]]+r[o[j]])/2.0
            for k in range(i,j+1): r[o[k]]=avg
        i=j+1
    return r
def auc(scores,labels):
    labels=np.asarray(labels); scores=np.asarray(scores,float)
    npos=labels.sum(); nneg=len(labels)-npos
    if npos==0 or nneg==0: return float("nan")
    r=rankavg(scores); return (r[labels==1].sum()-npos*(npos+1)/2)/(npos*nneg)
def auc_ci(scores,labels,B=2000,seed=0):
    rng=np.random.default_rng(seed); labels=np.asarray(labels); scores=np.asarray(scores,float)
    pos=np.where(labels==1)[0]; neg=np.where(labels==0)[0]; a=auc(scores,labels); bs=[]
    for _ in range(B):
        ip=rng.choice(pos,len(pos),True); ineg=rng.choice(neg,len(neg),True)
        idx=np.r_[ip,ineg]; bs.append(auc(scores[idx],labels[idx]))
    lo,hi=np.percentile(bs,[2.5,97.5]); return a,lo,hi

# ---- construir filas con etiquetas objetivas ----
rows=[]; miss_dbg=[]
for r in recs:
    st=byimg.get(r["ImageID"])
    if st is None: continue
    present=set(); absent=set()
    for f in st["findings"]:
        txt=" ".join(f.get("labels",[]))+" "+f.get("sentence_en","")
        cats=cats_of_text(txt)
        (present if f["abnormal"] else absent).update(cats)
    normal=not any(f["abnormal"] for f in st["findings"])
    asrt=asserted_cats(r["tokens"])
    halluc_cats={c for c in asrt if c not in present and (normal or c in absent)}
    miss_cats={c for c in present if c not in asrt}
    is_h=len(halluc_cats)>0; is_m=(not is_h) and len(miss_cats)>0
    typ="halluc" if is_h else ("miss" if is_m else "correct")
    p=np.array(r["probs"],float); e=np.array(r["ents"],float); smp=r.get("samples",[])
    rows.append(dict(id=r["ImageID"],normal=normal,typ=typ,present=sorted(present),asrt=sorted(asrt),
        hcats=sorted(halluc_cats),mcats=sorted(miss_cats),ntok=len(p),min_prob=float(p.min()),
        max_ent=float(e.max()),consist=float(np.mean([sim(r["gen"],s) for s in smp])) if smp else 1.0,
        occ=r.get("occ_drop")))

typ=np.array([x["typ"] for x in rows]); col=lambda f:np.array([x[f] for x in rows],float)
nt=col("ntok"); me=col("max_ent")
# res_maxent: residual OLS de max_ent sobre ntok (manual, sin polyfit)
xx=nt-nt.mean(); sl=(xx*(me-me.mean())).sum()/((xx*xx).sum()+1e-9); resme=me-(me.mean()+sl*(nt-nt.mean()))
for x,v in zip(rows,resme): x["res_maxent"]=float(v)

nH=(typ=="halluc").sum(); nM=(typ=="miss").sum(); nC=(typ=="correct").sum()
nnorm=sum(1 for x in rows if x["normal"])
print(f"n={len(rows)}  (normales={nnorm}, anormales={len(rows)-nnorm})")
print(f"ETIQUETAS OBJETIVAS →  alucinación={nH}  miss={nM}  correcto={nC}")
print(f"tasa alucinación={nH/len(rows):.1%}   (Open-I fue 42/800=5.3%)")

# ---- señales de incertidumbre: dirección = más incierto → más alucinación ----
def z(a): a=np.asarray(a,float); s=a.std(); return (a-a.mean())/(s if s>1e-9 else 1)
mp=col("min_prob"); rm=col("res_maxent"); cs=col("consist")
sig={"−min_prob":-mp, "res_maxent":rm, "1−consist":1-cs,
     "compuesta(z)": z(-mp)+z(rm)+z(1-cs)}

# máscara halluc-vs-correcto (excluye miss, igual que Open-I)
m_hc=(typ=="halluc")|(typ=="correct"); y_hc=(typ[m_hc]=="halluc").astype(int)
print(f"\n=== AUC( ALUCINACIÓN vs CORRECTO )  n_pos={int(y_hc.sum())} n_neg={int((y_hc==0).sum())} ===")
print("(referencia Open-I: 0.87)")
for name,s in sig.items():
    a,lo,hi=auc_ci(s[m_hc],y_hc)
    print(f"  {name:14s} AUC={a:.3f}  IC95%[{lo:.3f},{hi:.3f}]")

# miss vs correcto (esperamos ~0.5-0.6, disociación)
m_mc=(typ=="miss")|(typ=="correct"); y_mc=(typ[m_mc]=="miss").astype(int)
print(f"\n=== AUC( MISS vs CORRECTO )  n_pos={int(y_mc.sum())} n_neg={int((y_mc==0).sum())} ===")
print("(referencia Open-I: 0.61 — la incertidumbre NO detecta misses)")
if y_mc.sum()>=3:
    for name,s in sig.items():
        a,lo,hi=auc_ci(s[m_mc],y_mc)
        print(f"  {name:14s} AUC={a:.3f}  IC95%[{lo:.3f},{hi:.3f}]")

# subgrupo más limpio: SOLO estudios normales (halluc = inventar patología en tórax sano)
norm=np.array([x["normal"] for x in rows]); yn=(typ=="halluc").astype(int)
mnorm=norm & ((typ=="halluc")|(typ=="correct"))
print(f"\n=== SUBGRUPO tórax NORMAL: alucinación (inventar) vs correcto  n={int(mnorm.sum())} (halluc={int(yn[mnorm].sum())}) ===")
if yn[mnorm].sum()>=3 and (yn[mnorm]==0).sum()>=3:
    a,lo,hi=auc_ci((-mp)[mnorm],yn[mnorm]); print(f"  −min_prob      AUC={a:.3f}  IC95%[{lo:.3f},{hi:.3f}]")
    a,lo,hi=auc_ci(sig['compuesta(z)'][mnorm],yn[mnorm]); print(f"  compuesta(z)   AUC={a:.3f}  IC95%[{lo:.3f},{hi:.3f}]")

# medias por grupo (sanity)
print("\nmedias min_prob: correcto={:.3f}  ALUCINA={:.3f}  miss={:.3f}".format(
    mp[typ=="correct"].mean(), mp[typ=="halluc"].mean() if nH else float("nan"), mp[typ=="miss"].mean() if nM else float("nan")))

# guardar tabla para memoria/figuras
json.dump(rows,open(SC+r"\padchest_rows.json","w"),ensure_ascii=False)
print(f"\n✓ guardado padchest_rows.json ({len(rows)} filas)")
