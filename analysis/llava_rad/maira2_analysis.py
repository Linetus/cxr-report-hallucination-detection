"""Análisis MAIRA-2 sobre PadChest (in-distribution): etiquetas objetivas + min_prob CONTENIDO.
   ¿Un modelo mejor + en-distribución recupera la señal de detección de alucinaciones?
   Negación por tokens con manejo de 'except/aside' (MAIRA usa lenguaje más rico)."""
import os,sys,json,re
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
byimg={a["ImageID"]:a for a in json.load(open(r"./data/raw/padchest_gr/grounded_reports.json",encoding="utf-8"))}
recs=[json.loads(l) for l in open(SC+r"\maira2_padchest.jsonl",encoding="utf-8")]
FOCAL={"consolidation":["consolidat","airspace","alveolar"],"opacity":["opacit"],"nodule":["nodul"],
 "mass":["mass"],"effusion":["effusion"],"edema":["edema","congest"],"atelectasis":["atelectas","collapse"],
 "cardiomegaly":["cardiomeg","enlarg"],"pneumothorax":["pneumothorax"],
 "infiltrate":["infiltrat","interstitial pattern"],"pneumonia":["pneumonia"],"fibrosis":["fibros"]}
SOFT={"cardiomegaly"}
NEGSET={"no","not","without","free","negative","nor","never","clear","unremarkable","resolved","normal"}
CANCEL={"except","exception","aside","apart","however","but"}  # cancelan la negación previa
SENTEND={".","!","?"}
def cats_text(t):
    tl=t.lower(); return {c for c,syns in FOCAL.items() if any(s in tl for s in syns)}
def asserted_cats(toks):
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
                    if tj=="": continue
                    if tj in SENTEND or tj in CANCEL: break   # 'except for X' => X afirmado
                    if tj in NEGSET: neg=True; break
                if not neg: out.add(c); hit=True; break
            if hit: break
    return out
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
    for _ in range(B):
        idx=np.r_[rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)]; bs.append(auc(sc[idx],lb[idx]))
    return a,*np.percentile(bs,[2.5,97.5])
def ng(t,n=4): t=t.lower(); return set(t[i:i+n] for i in range(max(len(t)-n+1,1)))
def sim(a,b):
    A,B=ng(a),ng(b); u=len(A|B); return len(A&B)/u if u else 1.0

rows=[]
for r in recs:
    st=byimg.get(r["ImageID"])
    if st is None: continue
    present=set(); absent=set()
    for f in st["findings"]:
        txt=" ".join(f.get("labels",[]))+" "+f.get("sentence_en","")
        (present if f["abnormal"] else absent).update(cats_text(txt))
    normal=not any(f["abnormal"] for f in st["findings"])
    asrt=asserted_cats(r["tokens"])
    hcats={c for c in asrt if c not in present and (normal or c in absent)}
    mcats={c for c in present if c not in asrt}
    typ="halluc" if hcats else ("miss" if mcats else "correct")
    p=np.array(r["probs"],float); smp=r.get("samples",[])
    cmin=float(p[1:].min()) if len(p)>3 else float(p.min())  # min_prob CONTENIDO
    rows.append(dict(id=r["ImageID"],normal=normal,typ=typ,hcats=sorted(hcats),
        min_prob_full=float(p.min()),min_prob=cmin,
        consist=float(np.mean([sim(r["gen"],s) for s in smp])) if smp else 1.0,
        blank_sim=sim(r["gen"],r.get("blank_gen",""))))
typ=np.array([x["typ"] for x in rows])
nH=int((typ=="halluc").sum()); nM=int((typ=="miss").sum()); nC=int((typ=="correct").sum())
print(f"MAIRA-2 n={len(rows)}  alucina={nH} miss={nM} correcto={nC}  (tasa aluc {nH/len(rows):.0%})")
mp=np.array([x["min_prob"] for x in rows]); mpf=np.array([x["min_prob_full"] for x in rows])
m=(typ=="halluc")|(typ=="correct"); y=(typ[m]=="halluc").astype(int)
if y.sum()>=4:
    a,lo,hi=auc_ci(-mpf[m],y); print(f"AUC(alucina) min_prob COMPLETO = {a:.3f} IC[{lo:.3f},{hi:.3f}]")
    a,lo,hi=auc_ci(-mp[m],y);  print(f"AUC(alucina) min_prob CONTENIDO= {a:.3f} IC[{lo:.3f},{hi:.3f}]  (LLaVA-Rad OOD daba 0.48/0.61)")
    print(f"  medias min_prob-contenido: correcto={mp[typ=='correct'].mean():.3f} alucina={mp[typ=='halluc'].mean():.3f}")
    # focal-only (sin overcall cardiomegalia)
    foc=[x for x in rows if x["typ"]!="halluc" or not (set(x["hcats"])<=SOFT)]
    t2=np.array([x["typ"] for x in foc]); mp2=np.array([x["min_prob"] for x in foc])
    m2=(t2=="halluc")|(t2=="correct"); y2=(t2[m2]=="halluc").astype(int)
    if y2.sum()>=4:
        a,lo,hi=auc_ci(-mp2[m2],y2); print(f"AUC(alucina) focal-only min_prob CONTENIDO = {a:.3f} IC[{lo:.3f},{hi:.3f}] (n_halluc={int(y2.sum())})")
json.dump(rows,open(SC+r"\maira2_rows.json","w"),ensure_ascii=False)
print(f"guardado maira2_rows.json ({len(rows)})")
