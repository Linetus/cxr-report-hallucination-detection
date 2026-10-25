import os,sys,json,math,numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
GC={r["id"]:r for r in json.load(open(SC+r"\gen_conf.json"))}
META=json.load(open(SC+r"\pertoken_meta.json")); NPZ=np.load(SC+r"\pertoken_hm.npz")
LAB=json.load(open(SC+r"\labels.json")); G=37
items=[r for r in META if f"hm_{r['k']}" in NPZ.files and r["id"] in LAB and r["id"] in GC]
rows=np.concatenate([NPZ[f"hm_{r['k']}"].astype(np.float64) for r in items],0)
pm=rows.mean(0);thr=pm.mean()+6*pm.std();sink=np.where(pm>thr)[0];nonsink=np.ones(G*G,bool);nonsink[sink]=False
def norm_ns(v):
    a=v[nonsink].astype(np.float64);a=a-a.min();s=a.sum();return a/s if s>0 else np.ones_like(a)/len(a)
mean_pat=norm_ns(rows.mean(0));ys,xs=np.mgrid[0:G,0:G]
def spread_of(r):
    hm=NPZ[f"hm_{r['k']}"].astype(np.float64);P=np.array([norm_ns(hm[i]) for i in range(len(r["tokens"]))])
    w=np.sqrt(((P-mean_pat)**2).sum(1));w=w if w.sum()>0 else np.ones(len(P));agg=(w[:,None]*P).sum(0)/w.sum()
    H=np.zeros(G*G);H[nonsink]=agg;H=H.reshape(G,G);w2=H/(H.sum()+1e-12);cy,cx=(w2*ys).sum(),(w2*xs).sum()
    return math.sqrt((w2*((ys-cy)**2+(xs-cx)**2)).sum())
ids=[r["id"] for r in items]
mp=np.array([min(GC[i]["probs"]) for i in ids]);sp=np.array([spread_of(r) for r in items]);y=np.array([LAB[i] for i in ids])
FOC={"CXR1001_IM-0004-1002","CXR1016_IM-0013-2001","CXR1019_IM-0015-2001","CXR103_IM-0023-2001","CXR1030_IM-0024-1001","CXR1030_IM-0024-2001","CXR1035_IM-0028-2001","CXR1044_IM-0036-1001","CXR1047_IM-0036-1001","CXR1047_IM-0036-2001","CXR1000_IM-0003-1001","CXR1000_IM-0003-3001","CXR1001_IM-0004-1001","CXR1003_IM-0005-2002","CXR101_IM-0011-2001","CXR101_IM-0011-4004","CXR1012_IM-0013-1001","CXR1014_IM-0013-1001","CXR1014_IM-0013-2001","CXR1015_IM-0001-1001","CXR1015_IM-0001-2001","CXR1015_IM-0013-1001","CXR1015_IM-0013-2001","CXR1016_IM-0013-1001","CXR1021_IM-0017-1001-0001","CXR1034_IM-0028-1001","CXR1037_IM-0029-1001","CXR1043_IM-0035-0001-0001","CXR1043_IM-0035-0001-0002","CXR1050_IM-0038-4004","CXR1053_IM-0040-1001","CXR1053_IM-0040-3003","CXR105_IM-0037-1001","CXR1022_IM-0017-2001"}
fm=np.array([i in FOC for i in ids])
def z(v,m):return (v-v[m].mean())/(v[m].std()+1e-9)
idx=z(-mp,fm)+z(-sp,fm)
fig,ax=plt.subplots(1,3,figsize=(13,4))
def strip(a,v,mask,title,xl):
    yy=y[mask];vv=v[mask];rng=np.random.default_rng(1)
    for cls,col,lbl in [(0,"#2a9d8f","correcto"),(1,"#e76f51","error/alucina")]:
        d=vv[yy==cls];a.scatter(d,rng.normal(cls,0.06,len(d)),c=col,s=45,alpha=.7,label=f"{lbl} (μ={d.mean():.2f})",edgecolors="k",linewidths=.4)
        a.plot([d.mean(),d.mean()],[cls-0.22,cls+0.22],c=col,lw=3)
    a.set_yticks([0,1]);a.set_yticklabels(["correcto","error"]);a.set_title(title);a.set_xlabel(xl);a.legend(fontsize=7,loc="upper right")
strip(ax[0],mp,fm,"min_prob (incertidumbre)","prob del token más inseguro")
strip(ax[1],sp,fm,"spread (dispersión heatmap)","spread")
strip(ax[2],idx,np.ones(len(y),bool) & fm,"ÍNDICE combinado z(-mp)+z(-sp)","índice (↑ = más alucina)")
plt.suptitle("Separación errores vs correctos (subset FOCAL, n=34)  —  min_prob lleva la señal",fontweight="bold")
plt.tight_layout()
OUT=SC+r"\combined_index.png";plt.savefig(OUT,dpi=110,bbox_inches="tight");print("→",OUT)
