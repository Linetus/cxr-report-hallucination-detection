"""RE-AUDIT Open-I con negación corregida y etiquetas OBJETIVAS (ref del radiólogo),
   idéntico a PadChest. ¿El 0.87 del juez-LLM se sostiene con criterio objetivo?
   (A) por-afirmación: incertidumbre local/global separa fabricado vs real?
   (B) por-estudio: AUC(halluc vs correcto) con etiquetas objetivas."""
import os,sys,json,re
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
recs=[json.loads(l) for l in open(SC+r"\allsignals_openi.jsonl",encoding="utf-8")]
FOCAL={"consolidation":["consolidat","airspace","alveolar"],"opacity":["opacit"],"nodule":["nodul"],
 "mass":["mass"],"effusion":["effusion"],"edema":["edema","congest"],"atelectasis":["atelectas","collapse"],
 "cardiomegaly":["cardiomeg","enlarg"],"pneumothorax":["pneumothorax"],
 "infiltrate":["infiltrat","interstitial pattern"],"pneumonia":["pneumonia"],"fibrosis":["fibros"]}
SOFT={"cardiomegaly"}
NEGSET={"no","not","without","free","negative","nor","never","clear","unremarkable","resolved","normal","within"}
SENTEND={".","!","?"}
# --- gen: afirmaciones sobre tokens (negación corregida) + incertidumbre local ---
def gen_assertions(toks,probs,ents):
    spans=[]; s=""
    for t in toks: spans.append((len(s),len(s)+len(t))); s+=t
    tl=s.lower()
    def tokidx(pos):
        for i,(a,b) in enumerate(spans):
            if a<=pos<b: return i
        return len(spans)-1
    out=[]
    for c,syns in FOCAL.items():
        for syn in syns:
            done=False
            for m in re.finditer(syn,tl):
                i0=tokidx(m.start()); i1=tokidx(m.end()-1); neg=False
                for j in range(i0-1,-1,-1):
                    tj=toks[j].strip().lower()
                    if tj=="": continue
                    if tj in SENTEND: break
                    if tj in NEGSET: neg=True; break
                if not neg:
                    lo=max(0,i0-1); hi=min(len(probs),i1+2)
                    out.append((c,float(min(probs[lo:hi])) if hi>lo else 1.0,float(np.mean(ents[lo:hi])) if hi>lo else 0.0)); done=True; break
            if done: break
    return out
# --- ref: presente/ausente sobre texto con espacios (negación por palabra, scoped a la frase) ---
def ref_cats(text):
    words=re.findall(r"[a-z]+|\.",text.lower()); present=set(); absent=set()
    for c,syns in FOCAL.items():
        for i,w in enumerate(words):
            if any(w.startswith(s) or s in w for s in syns):
                neg=False
                for j in range(i-1,-1,-1):
                    if words[j]==".": break
                    if words[j] in NEGSET: neg=True; break
                (absent if neg else present).add(c); break
    return present,absent
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

# ===== construir =====
ASS=[]; studies=[]
for r in recs:
    pres,abse=ref_cats(r["ref"]); normal=(len(pres)==0)
    probs=np.array(r["probs"]); ents=np.array(r["ents"])
    ga=gen_assertions(r["tokens"],probs,ents)
    asrt={c for c,_,_ in ga}
    for c,lmin,lent in ga:
        real=c in pres; fab=(not real) and (c in abse)   # solo contradicciones limpias
        if real or fab:
            ASS.append((c,c in SOFT,fab,float(probs.min()),lmin,lent))
    hcats={c for c in asrt if c in abse}          # afirma lo que ref niega
    mcats={c for c in pres if c not in asrt}       # omite lo que ref afirma
    typ="halluc" if hcats else ("miss" if mcats else "correct")
    smp=r.get("samples",[])
    studies.append(dict(id=r["id"],typ=typ,ntok=len(probs),min_prob=float(probs.min()),
        max_ent=float(ents.max()),consist=float(np.mean([sim(r["gen"],s) for s in smp])) if smp else 1.0))

# (A) por-afirmación
print("="*70); print("(A) POR-AFIRMACIÓN Open-I (etiqueta objetiva ref):")
def rep(name,sub):
    fab=np.array([a[2] for a in sub],int); g=np.array([a[3] for a in sub]); lm=np.array([a[4] for a in sub]); le=np.array([a[5] for a in sub])
    nT=int((fab==0).sum()); nF=int(fab.sum())
    print(f"\n {name}: {len(sub)} afirm. (reales={nT}, FABRICADAS={nF})")
    if nF>=5 and nT>=5:
        print(f"   min_prob GLOBAL AUC(fab)={auc(-g,fab):.3f}  (real {g[fab==0].mean():.3f} vs fab {g[fab==1].mean():.3f})")
        print(f"   min_prob LOCAL  AUC(fab)={auc(-lm,fab):.3f}  (real {lm[fab==0].mean():.3f} vs fab {lm[fab==1].mean():.3f})")
        print(f"   entropía LOCAL  AUC(fab)={auc(le,fab):.3f}  (real {le[fab==0].mean():.3f} vs fab {le[fab==1].mean():.3f})")
    else: print(f"   (pocas: fab={nF} real={nT})")
rep("TODAS focales",ASS); rep("SIN cardiomegalia",[a for a in ASS if not a[1]]); rep("SOLO cardiomegalia",[a for a in ASS if a[1]])

# (B) por-estudio objetivo
typ=np.array([s["typ"] for s in studies]); col=lambda f:np.array([s[f] for s in studies],float)
nt=col("ntok"); me=col("max_ent"); xx=nt-nt.mean(); sl=(xx*(me-me.mean())).sum()/((xx*xx).sum()+1e-9); rm=me-(me.mean()+sl*(nt-nt.mean()))
mp=col("min_prob"); cs=col("consist")
def z(a): s=a.std(); return (a-a.mean())/(s if s>1e-9 else 1)
comp=z(-mp)+z(rm)+z(1-cs)
nH=int((typ=="halluc").sum()); nM=int((typ=="miss").sum()); nC=int((typ=="correct").sum())
print("\n"+"="*70); print(f"(B) POR-ESTUDIO objetivo:  halluc={nH} miss={nM} correcto={nC}  (juez-LLM daba: halluc≈42, AUC 0.87)")
m=(typ=="halluc")|(typ=="correct"); y=(typ[m]=="halluc").astype(int)
for name,sc in [("min_prob",-mp[m]),("res_maxent",rm[m]),("1-consist",(1-cs)[m]),("compuesta",comp[m])]:
    a,lo,hi=auc_ci(sc,y); print(f"   AUC(halluc vs correcto) {name:11s}={a:.3f}  IC95%[{lo:.3f},{hi:.3f}]")
print(f"   medias min_prob: correcto={mp[typ=='correct'].mean():.3f}  ALUCINA={mp[typ=='halluc'].mean():.3f}")
