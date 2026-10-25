"""Test a nivel de AFIRMACIÓN: ¿la incertidumbre distingue una afirmación FABRICADA de una REAL?
   Incertidumbre GLOBAL (min_prob del informe) vs LOCAL (en los tokens del hallazgo).
   Separa cardiomegalia blanda ('mildly enlarged') de fabricaciones focales."""
import os,sys,json,re
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
byimg={a["ImageID"]:a for a in json.load(open(r"./data/raw/padchest_gr/grounded_reports.json",encoding="utf-8"))}
recs=[json.loads(l) for l in open(SC+r"\padchest_full.jsonl",encoding="utf-8")]
FOCAL={"consolidation":["consolidat","airspace","alveolar"],"opacity":["opacit"],"nodule":["nodul"],
 "mass":["mass"],"effusion":["effusion"],"edema":["edema","congest"],"atelectasis":["atelectas","collapse"],
 "cardiomegaly":["cardiomeg","enlarg"],"pneumothorax":["pneumothorax"],
 "infiltrate":["infiltrat","interstitial pattern"],"pneumonia":["pneumonia"],"fibrosis":["fibros"]}
FOCAL_SOFT={"cardiomegaly"}  # sobre-diagnóstico blando (umbral), se analiza aparte
NEGSET={"no","not","without","free","negative","nor","never","clear","unremarkable","resolved","normal"}
SENTEND={".","!","?"}
def cats_text(t):
    tl=t.lower(); return {c for c,syns in FOCAL.items() if any(s in tl for s in syns)}
def assertions(toks,probs,ents):
    """devuelve lista de (cat, min_prob_local, mean_ent_local) para cada hallazgo AFIRMADO (no negado)."""
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
                    lo=max(0,i0-1); hi=min(len(probs),i1+2)  # ventana del hallazgo ±1 token
                    pl=probs[lo:hi]; el=ents[lo:hi]
                    out.append((c,float(min(pl)) if len(pl) else 1.0,float(np.mean(el)) if len(el) else 0.0))
                    done=True; break
            if done: break
    return out
def rankavg(x):
    o=np.argsort(x,kind="mergesort"); r=np.empty(len(x)); r[o]=np.arange(1,len(x)+1)
    x_s=x[o]; i=0
    while i<len(x):
        j=i
        while j+1<len(x) and x_s[j+1]==x_s[i]: j+=1
        if j>i:
            avg=(r[o[i]]+r[o[j]])/2
            for k in range(i,j+1): r[o[k]]=avg
        i=j+1
    return r
def auc(sc,lb):
    lb=np.asarray(lb); sc=np.asarray(sc,float); npos=lb.sum(); nneg=len(lb)-npos
    if npos==0 or nneg==0: return float("nan")
    r=rankavg(sc); return (r[lb==1].sum()-npos*(npos+1)/2)/(npos*nneg)

# construir universo de afirmaciones etiquetadas
A=[]  # (cat, soft?, fabricated?, gmin, lmin, lent)
for r in recs:
    st=byimg.get(r["ImageID"])
    if st is None: continue
    present=set(); absent=set()
    for f in st["findings"]:
        txt=" ".join(f.get("labels",[]))+" "+f.get("sentence_en","")
        (present if f["abnormal"] else absent).update(cats_text(txt))
    normal=not any(f["abnormal"] for f in st["findings"])
    probs=r["probs"]; ents=r["ents"]
    gmin=float(min(probs))
    for c,lmin,lent in assertions(r["tokens"],probs,ents):
        fab = (c not in present) and (normal or c in absent)
        A.append((c,c in FOCAL_SOFT,fab,gmin,lmin,lent))

def report(name,sub):
    fab=np.array([a[2] for a in sub],int); gmin=np.array([a[3] for a in sub]); lmin=np.array([a[4] for a in sub]); lent=np.array([a[5] for a in sub])
    nT=int((fab==0).sum()); nF=int(fab.sum())
    print(f"\n=== {name}: {len(sub)} afirmaciones  (reales={nT}, FABRICADAS={nF}) ===")
    if nF>=3 and nT>=3:
        print(f"  min_prob GLOBAL  →  AUC(fab)={auc(-gmin,fab):.3f}   (media real={gmin[fab==0].mean():.3f} fab={gmin[fab==1].mean():.3f})")
        print(f"  min_prob LOCAL   →  AUC(fab)={auc(-lmin,fab):.3f}   (media real={lmin[fab==0].mean():.3f} fab={lmin[fab==1].mean():.3f})")
        print(f"  entropía LOCAL   →  AUC(fab)={auc(lent,fab):.3f}   (media real={lent[fab==0].mean():.3f} fab={lent[fab==1].mean():.3f})")
    else:
        print(f"  (insuficientes: fab={nF}, reales={nT})")

print(f"TOTAL afirmaciones focales: {len(A)}")
report("TODAS las afirmaciones focales", A)
report("SOLO FOCALES (excluye cardiomegalia blanda)", [a for a in A if not a[1]])
report("SOLO cardiomegalia", [a for a in A if a[1]])
