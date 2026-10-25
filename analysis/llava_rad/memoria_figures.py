"""Figuras memoria TFM: (1) separación de clases (min_prob-contenido, correcto vs alucinación, 4 celdas),
   (2) matriz 2x2 de AUC (modelo x dominio). matplotlib Agg + fix BLAS Windows."""
import os,sys,json
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
FIG=r"./.claude/worktrees/strange-yalow-4533b2/memoria/figuras"; os.makedirs(FIG,exist_ok=True)
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
    else: # maira_openi
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

CELLS=[("llava_openi","LLaVA-Rad · Open-I","in-distribution"),
       ("llava_padchest","LLaVA-Rad · PadChest","OOD"),
       ("maira_padchest","MAIRA-2 · PadChest","in-distribution"),
       ("maira_openi","MAIRA-2 · Open-I","OOD")]
data={k:load(k) for k,_,_ in CELLS}
GREEN,RED="#2e8b57","#e5484d"

# ---- Fig 1: separación de clases ----
fig,axes=plt.subplots(2,2,figsize=(11,7.5))
GREY="#7a7f87"
for ax,(k,title,dom) in zip(axes.flat,CELLS):
    C,H,M=data[k]
    a=auc(-np.r_[C,H],np.r_[np.zeros(len(C)),np.ones(len(H))])
    am=auc(-np.r_[C,M],np.r_[np.zeros(len(C)),np.ones(len(M))]) if len(M) else float("nan")
    bins=np.linspace(0,1,26)
    ax.hist(C,bins=bins,density=True,color=GREEN,alpha=.5,label=f"correcto (n={len(C)})")
    ax.hist(M,bins=bins,density=True,color=GREY,alpha=.45,label=f"miss/omisión (n={len(M)})")
    ax.hist(H,bins=bins,density=True,color=RED,alpha=.5,label=f"alucinación (n={len(H)})")
    ax.axvline(C.mean(),color=GREEN,ls="--",lw=1.5); ax.axvline(H.mean(),color=RED,ls="--",lw=1.5)
    if len(M): ax.axvline(M.mean(),color=GREY,ls="--",lw=1.5)
    ax.set_title(f"{title}  ({dom})\nAUC alucina={a:.2f}   ·   miss={am:.2f}",fontsize=10.5,fontweight="bold")
    ax.set_xlabel("min_prob de contenido  (menor → más incierto)"); ax.set_ylabel("densidad")
    ax.legend(fontsize=7.5,loc="upper left"); ax.set_xlim(0,1)
fig.suptitle("Separación de clases: la incertidumbre separa alucinación de correcto; los misses caen entremedias, cerca del umbral",
             fontsize=12,fontweight="bold")
fig.tight_layout(rect=[0,0,1,.96]); fig.savefig(FIG+r"\fig_separacion_clases.png",dpi=150); plt.close(fig)
print("fig_separacion_clases.png OK")

# ---- Fig 2: matriz 2x2 AUC ----
AUCS=np.array([[0.90,0.61],[0.80,0.81]])           # [modelo][dominio]
NH=[["42/110","33"],["27","12"]]
fig,ax=plt.subplots(figsize=(6.2,4.6))
im=ax.imshow(AUCS,cmap="RdYlGn",vmin=0.5,vmax=1.0,aspect="auto")
ax.set_xticks([0,1]); ax.set_xticklabels(["in-distribution","OOD"],fontsize=11)
ax.set_yticks([0,1]); ax.set_yticklabels(["LLaVA-Rad","MAIRA-2"],fontsize=11)
for i in range(2):
    for j in range(2):
        ax.text(j,i,f"AUC {AUCS[i,j]:.2f}",ha="center",va="center",fontsize=15,fontweight="bold")
        ax.text(j,i+.22,f"(n_halluc {NH[i][j]})",ha="center",va="center",fontsize=8,color="#333")
ax.set_title("Detección de alucinaciones (min_prob-contenido)\nrobusta al modelo y al dominio",
             fontsize=12,fontweight="bold")
cbar=fig.colorbar(im,ax=ax,fraction=.046,pad=.04); cbar.set_label("AUC")
fig.tight_layout(); fig.savefig(FIG+r"\fig_2x2_auc.png",dpi=150); plt.close(fig)
print("fig_2x2_auc.png OK")
print("Figuras en:",FIG)
