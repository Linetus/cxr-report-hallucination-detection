"""RE-AUDIT Open-I con VERDAD OFICIAL MeSH (Indiana). Sin parseo de negación en la verdad
   (MeSH solo lista lo PRESENTE). Etiqueta objetiva por-afirmación y por-estudio."""
import os,sys,json,re,glob
import xml.etree.ElementTree as ET
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
XD=r"./data/raw/open-i/ecgen-radiology"
recs=[json.loads(l) for l in open(SC+r"\allsignals_openi.jsonl",encoding="utf-8")]
FOCAL={"consolidation":["consolidat","airspace","alveolar"],"opacity":["opacit"],"nodule":["nodul"],
 "mass":["mass"],"effusion":["effusion"],"edema":["edema","congest"],"atelectasis":["atelectas","collapse"],
 "cardiomegaly":["cardiomeg","enlarg"],"pneumothorax":["pneumothorax"],
 "infiltrate":["infiltrat","interstitial pattern"],"pneumonia":["pneumonia"],"fibrosis":["fibros"]}
SOFT={"cardiomegaly"}
# MeSH major → categorías focales (por substring; MeSH solo lista PRESENTES)
MESHMAP={"cardiomeg":"cardiomegaly","opacit":"opacity","atelectas":"atelectasis","effusion":"effusion",
 "edema":"edema","consolidat":"consolidation","airspace":"consolidation","pneumonia":"pneumonia",
 "pneumothorax":"pneumothorax","nodule":"nodule","granuloma":"nodule","mass":"mass","fibros":"fibrosis",
 "infiltrat":"infiltrate","emphysema":"fibrosis"}
NEGSET={"no","not","without","free","negative","nor","never","clear","unremarkable","resolved","normal"}
SENTEND={".","!","?"}
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

# --- parsear MeSH: imageid -> (present_cats, is_normal) ---
img2gold={}
for f in glob.glob(XD+r"\*.xml"):
    try: root=ET.parse(f).getroot()
    except Exception: continue
    majors=[m.text.lower() for m in root.findall(".//MeSH/major") if m.text]
    normal = (len(majors)==0) or all(m.strip() in ("normal","no indexing") for m in majors)
    cats=set()
    for m in majors:
        for k,c in MESHMAP.items():
            if k in m: cats.add(c)
    for pi in root.findall(".//parentImage"):
        iid=pi.get("id")
        if iid: img2gold[iid]=(cats,normal)

# --- construir ---
ASS=[]; studies=[]; matched=0
for r in recs:
    g=img2gold.get(r["id"])
    if g is None: continue
    pres,normal=g; matched+=1
    probs=np.array(r["probs"]); ents=np.array(r["ents"])
    ga=gen_assertions(r["tokens"],probs,ents); asrt={c for c,_,_ in ga}
    for c,lmin,lent in ga:
        real=c in pres; fab=not real   # MeSH completo: no presente = fabricado
        ASS.append((c,c in SOFT,fab,float(probs.min()),lmin,lent))
    hcats={c for c in asrt if c not in pres}
    mcats={c for c in pres if c not in asrt}
    typ="halluc" if hcats else ("miss" if mcats else "correct")
    smp=r.get("samples",[])
    cmin=float(probs[1:].min()) if len(probs)>3 else float(probs.min())   # min_prob de CONTENIDO (sin token apertura)
    studies.append(dict(typ=typ,ntok=len(probs),min_prob=cmin,max_ent=float(ents.max()),
        consist=float(np.mean([sim(r["gen"],s) for s in smp])) if smp else 1.0))
print(f"emparejados con MeSH: {matched}/{len(recs)}")

print("="*70); print("(A) POR-AFIRMACIÓN (verdad MeSH):")
def rep(name,sub):
    fab=np.array([a[2] for a in sub],int); gg=np.array([a[3] for a in sub]); lm=np.array([a[4] for a in sub]); le=np.array([a[5] for a in sub])
    nT=int((fab==0).sum()); nF=int(fab.sum()); print(f"\n {name}: {len(sub)} (reales={nT}, FABRICADAS={nF})")
    if nF>=5 and nT>=5:
        print(f"   min_prob GLOBAL AUC(fab)={auc(-gg,fab):.3f}  (real {gg[fab==0].mean():.3f} vs fab {gg[fab==1].mean():.3f})")
        print(f"   min_prob LOCAL  AUC(fab)={auc(-lm,fab):.3f}  (real {lm[fab==0].mean():.3f} vs fab {lm[fab==1].mean():.3f})")
    else: print(f"   (pocas: fab={nF} real={nT})")
rep("TODAS focales",ASS); rep("SIN cardiomegalia",[a for a in ASS if not a[1]]); rep("SOLO cardiomegalia",[a for a in ASS if a[1]])

typ=np.array([s["typ"] for s in studies]); col=lambda f:np.array([s[f] for s in studies],float)
nt=col("ntok"); me=col("max_ent"); xx=nt-nt.mean(); sl=(xx*(me-me.mean())).sum()/((xx*xx).sum()+1e-9); rm=me-(me.mean()+sl*(nt-nt.mean()))
mp=col("min_prob"); cs=col("consist")
def z(a): s=a.std(); return (a-a.mean())/(s if s>1e-9 else 1)
comp=z(-mp)+z(rm)+z(1-cs)
nH=int((typ=="halluc").sum())
print("\n"+"="*70); print(f"(B) POR-ESTUDIO (verdad MeSH): halluc={nH} miss={int((typ=='miss').sum())} correcto={int((typ=='correct').sum())}")
m=(typ=="halluc")|(typ=="correct"); y=(typ[m]=="halluc").astype(int)
for name,sc in [("min_prob",-mp[m]),("res_maxent",rm[m]),("1-consist",(1-cs)[m]),("compuesta",comp[m])]:
    a,lo,hi=auc_ci(sc,y); print(f"   AUC(halluc vs correcto) {name:11s}={a:.3f}  IC95%[{lo:.3f},{hi:.3f}]")
print(f"   medias min_prob: correcto={mp[typ=='correct'].mean():.3f} ALUCINA={mp[typ=='halluc'].mean():.3f}")

# ---- ¿detecta también los MISS (fallos)? y detector de CUALQUIER error ----
mm=(typ=="miss")|(typ=="correct"); ym=(typ[mm]=="miss").astype(int)
a,lo,hi=auc_ci(comp[mm],ym); print(f"\n   AUC(MISS vs correcto)    compuesta={a:.3f}  IC95%[{lo:.3f},{hi:.3f}]  ← fallos por confianza-falsa")
ae=(typ!="correct").astype(int)  # cualquier error
a,lo,hi=auc_ci(comp,ae); print(f"   AUC(CUALQUIER error)     compuesta={a:.3f}  IC95%[{lo:.3f},{hi:.3f}]")

# ---- PUNTOS DE OPERACIÓN (umbral sobre compuesta) ----
NH=int((typ=="halluc").sum()); NM=int((typ=="miss").sum()); NC=int((typ=="correct").sum())
ph=comp[typ=="halluc"]; pm=comp[typ=="miss"]; pc=comp[typ=="correct"]
print(f"\n(C) PUNTOS DE OPERACIÓN  (N: halluc={NH}, miss={NM}, correcto={NC})")
print(f"{'umbral':22s} {'rec.ALUCINA':>12s} {'rec.MISS':>10s} {'falsas alarmas':>15s} {'precisión-halluc':>16s}")
def line(lbl,t):
    tp=int((ph>=t).sum()); mm_=int((pm>=t).sum()); fp=int((pc>=t).sum())
    prec=tp/(tp+fp) if tp+fp else 0
    print(f"{lbl:22s} {tp:3d}/{NH} ({tp/NH:4.0%}) {mm_:3d}/{NM} ({mm_/NM:3.0%}) {fp:3d}/{NC} ({fp/NC:4.0%})     {prec:5.0%}")
# umbrales prácticos
ths=np.unique(np.r_[pc,ph]); yj=max(ths,key=lambda t:(ph>=t).mean()-(pc>=t).mean())
line("Youden (halluc)",yj)
line("Especif. 90%",np.quantile(pc,0.90))
line("Especif. 95%",np.quantile(pc,0.95))
line("Sensib. 90% halluc",np.quantile(ph,0.10))
# dump para el semáforo
import json as _j
_j.dump({"typ":list(typ),"comp":[float(x) for x in comp],"youden":float(yj),
    "pc50":float(np.median(pc)),"pc25":float(np.quantile(pc,0.25))},open(SC+r"\openi_scores.json","w"))
