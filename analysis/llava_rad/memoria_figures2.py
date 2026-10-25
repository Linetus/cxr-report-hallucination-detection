"""Figuras memoria (2): disociación halluc-vs-miss, curvas ROC, curva de calibración del semáforo."""
import os,sys,json
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
FIG=r"./.claude/worktrees/strange-yalow-4533b2/memoria/figuras"
def cmin(p): p=np.array(p,float); return float(p[1:].min()) if len(p)>3 else float(p.min())
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
def roc(sc,lb):
    lb=np.asarray(lb); order=np.argsort(-np.asarray(sc,float)); lb=lb[order]
    P=lb.sum(); N=len(lb)-P; tpr=[0]; fpr=[0]; tp=fp=0
    for y in lb:
        if y: tp+=1
        else: fp+=1
        tpr.append(tp/P); fpr.append(fp/N)
    return np.array(fpr),np.array(tpr)
def load(kind):
    if kind=="llava_openi":
        recs={json.loads(l)["id"]:json.loads(l) for l in open(SC+r"\allsignals_openi.jsonl",encoding="utf-8")}
        lab=json.load(open(SC+r"\openi_labels.json"))
    elif kind=="llava_padchest":
        recs={json.loads(l)["ImageID"]:json.loads(l) for l in open(SC+r"\padchest_full.jsonl",encoding="utf-8")}
        lab={x["id"]:x["typ"] for x in json.load(open(SC+r"\padchest_rows.json",encoding="utf-8"))}
    elif kind=="maira_padchest":
        recs={json.loads(l)["ImageID"]:json.loads(l) for l in open(SC+r"\maira2_padchest.jsonl",encoding="utf-8")}
        lab={k:v["label"] for k,v in json.load(open(SC+r"\maira2_geval_labels_all.json",encoding="utf-8")).items()}
    else:
        recs={json.loads(l)["ImageID"]:json.loads(l) for l in open(SC+r"\maira2_openi.jsonl",encoding="utf-8")}
        lab={}
        for l in open(SC+r"\maira2_openi_geval_labels.jsonl",encoding="utf-8"):
            d=json.loads(l); lab[d["id"]]=d["label"]
    C,H,M=[],[],[]
    for iid,t in lab.items():
        r=recs.get(iid)
        if r is None: continue
        m=cmin(r["probs"])
        (C if t=="correct" else H if t=="halluc" else M).append(m)
    return np.array(C),np.array(H),np.array(M)
CELLS=[("llava_openi","LLaVA·Open-I"),("llava_padchest","LLaVA·PadChest"),
       ("maira_padchest","MAIRA·PadChest"),("maira_openi","MAIRA·Open-I")]
D={k:load(k) for k,_ in CELLS}
GREEN,RED,GREY="#2e8b57","#e5484d","#9aa0a6"

# ---- Fig 3: disociación halluc vs miss ----
fig,ax=plt.subplots(figsize=(8,4.8)); x=np.arange(len(CELLS)); w=.36
ah=[]; am=[]
for k,_ in CELLS:
    C,H,M=D[k]
    # score = -min_prob (alucinación tiene min_prob BAJO -> score alto)
    ah.append(auc(-np.r_[C,H],np.r_[np.zeros(len(C)),np.ones(len(H))]) if len(H) else np.nan)
    am.append(auc(-np.r_[C,M],np.r_[np.zeros(len(C)),np.ones(len(M))]) if len(M) else np.nan)
ax.bar(x-w/2,ah,w,color=RED,label="alucinación vs correcto")
ax.bar(x+w/2,am,w,color=GREY,label="miss (omisión) vs correcto")
ax.axhline(0.5,color="k",ls=":",lw=1)
for i,(v) in enumerate(ah): ax.text(i-w/2,v+.01,f"{v:.2f}",ha="center",fontsize=9,fontweight="bold")
for i,(v) in enumerate(am):
    if not np.isnan(v): ax.text(i+w/2,v+.01,f"{v:.2f}",ha="center",fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([n for _,n in CELLS],fontsize=9)
ax.set_ylabel("AUC"); ax.set_ylim(0.4,1.0)
ax.set_title("La incertidumbre detecta ALUCINACIONES de forma robusta; las OMISIONES (misses), solo parcialmente",fontsize=10.5,fontweight="bold")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(FIG+r"\fig_disociacion.png",dpi=150); plt.close(fig)
print("fig_disociacion.png OK")

# ---- Fig 4: ROC ----
fig,ax=plt.subplots(figsize=(6,6)); COL=["#2e8b57","#e5484d","#f2b134","#3b6fb5"]
for (k,name),c in zip(CELLS,COL):
    C,H,M=D[k]
    if len(H)<3: continue
    lab=np.r_[np.zeros(len(C)),np.ones(len(H))]; s=-np.r_[C,H]   # score=-min_prob (halluc = min_prob bajo)
    a=auc(s,lab); f,t=roc(s,lab)
    ax.plot(f,t,color=c,lw=2,label=f"{name}  AUC={a:.2f}")
ax.plot([0,1],[0,1],"k:",lw=1)
ax.set_xlabel("Tasa de falsos positivos"); ax.set_ylabel("Sensibilidad (recall alucinación)")
ax.set_title("Curvas ROC — detección de alucinaciones (min_prob-contenido)",fontsize=11.5,fontweight="bold")
ax.legend(fontsize=9,loc="lower right"); ax.set_xlim(0,1); ax.set_ylim(0,1)
fig.tight_layout(); fig.savefig(FIG+r"\fig_roc.png",dpi=150); plt.close(fig)
print("fig_roc.png OK")

# ---- Fig 5: calibración (todos los datos, min_prob -> tasa alucinación observada) ----
allmp=[]; ally=[]
for k,_ in CELLS:
    C,H,M=D[k]; allmp+= list(C)+list(H); ally+=[0]*len(C)+[1]*len(H)
allmp=np.array(allmp); ally=np.array(ally)
edges=np.quantile(allmp,np.linspace(0,1,9)); edges[0]-=1e-6
cx=[]; cy=[];
for i in range(len(edges)-1):
    sel=(allmp>edges[i])&(allmp<=edges[i+1]); n=sel.sum()
    if n>=10: cx.append(allmp[sel].mean()); cy.append(ally[sel].mean())
fig,ax=plt.subplots(figsize=(6.5,4.8))
ax.plot(cx,cy,"o-",color="#3b6fb5",lw=2,ms=7)
ax.set_xlabel("min_prob de contenido (media del bin)"); ax.set_ylabel("tasa de alucinación observada")
ax.set_title("Calibración: menor confianza → mayor tasa de alucinación\n(base para el semáforo)",fontsize=11.5,fontweight="bold")
ax.grid(alpha=.3); ax.invert_xaxis()
fig.tight_layout(); fig.savefig(FIG+r"\fig_calibracion.png",dpi=150); plt.close(fig)
print("fig_calibracion.png OK")

# ---- Fig 6: efecto de excluir el token de apertura (min_prob completo vs contenido) ----
def load_raw(kind):
    if kind=="llava_openi":
        recs={json.loads(l)["id"]:json.loads(l) for l in open(SC+r"\allsignals_openi.jsonl",encoding="utf-8")}
        lab=json.load(open(SC+r"\openi_labels.json"))
    elif kind=="llava_padchest":
        recs={json.loads(l)["ImageID"]:json.loads(l) for l in open(SC+r"\padchest_full.jsonl",encoding="utf-8")}
        lab={x["id"]:x["typ"] for x in json.load(open(SC+r"\padchest_rows.json",encoding="utf-8"))}
    elif kind=="maira_padchest":
        recs={json.loads(l)["ImageID"]:json.loads(l) for l in open(SC+r"\maira2_padchest.jsonl",encoding="utf-8")}
        lab={k:v["label"] for k,v in json.load(open(SC+r"\maira2_geval_labels_all.json",encoding="utf-8")).items()}
    else:
        recs={json.loads(l)["ImageID"]:json.loads(l) for l in open(SC+r"\maira2_openi.jsonl",encoding="utf-8")}
        lab={}
        for l in open(SC+r"\maira2_openi_geval_labels.jsonl",encoding="utf-8"):
            d=json.loads(l); lab[d["id"]]=d["label"]
    out=[]
    for iid,t in lab.items():
        r=recs.get(iid)
        if r and t in ("correct","halluc"): out.append((t,np.array(r["probs"],float)))
    return out
fig,ax=plt.subplots(figsize=(8.5,4.8)); x=np.arange(len(CELLS)); w=.36
full=[]; cont=[]
for k,_ in CELLS:
    raw=load_raw(k); y=np.array([1 if t=="halluc" else 0 for t,_ in raw])
    sf=np.array([-float(p.min()) for _,p in raw])                                 # min_prob COMPLETO (negado)
    sc=np.array([-float(p[1:].min() if len(p)>3 else p.min()) for _,p in raw])    # min_prob CONTENIDO
    full.append(auc(sf,y)); cont.append(auc(sc,y))
ax.bar(x-w/2,full,w,color="#9aa0a6",label="min_prob COMPLETO (incluye token de apertura)")
ax.bar(x+w/2,cont,w,color="#2e8b57",label="min_prob de CONTENIDO (excluye 1er token)")
for i,v in enumerate(full): ax.text(i-w/2,v+.01,f"{v:.2f}",ha="center",fontsize=9)
for i,v in enumerate(cont): ax.text(i+w/2,v+.01,f"{v:.2f}",ha="center",fontsize=9,fontweight="bold")
for i in range(len(CELLS)): ax.annotate("",xy=(i+w/2,cont[i]),xytext=(i-w/2,full[i]),arrowprops=dict(arrowstyle="->",color="#333",lw=1))
ax.axhline(0.5,color="k",ls=":",lw=1); ax.set_xticks(x); ax.set_xticklabels([n for _,n in CELLS],fontsize=9)
ax.set_ylabel("AUC (alucinación vs correcto)"); ax.set_ylim(0.4,1.0)
ax.set_title("Excluir el token de apertura del informe mejora la señal de incertidumbre",fontsize=11.5,fontweight="bold")
ax.legend(fontsize=8.5,loc="upper right")
fig.tight_layout(); fig.savefig(FIG+r"\fig_token_apertura.png",dpi=150); plt.close(fig)
print("fig_token_apertura.png OK")
print("Todas en:",FIG)
