"""¿min_prob focal es robusto o lo mueven 1-2 puntos? Jackknife + longitud dentro de focal."""
import os,sys,json,math,numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.metrics import roc_auc_score
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
G=json.load(open(SC+r"\gen_conf.json")); LAB=json.load(open(SC+r"\labels.json"))
FOC={"CXR1001_IM-0004-1002","CXR1016_IM-0013-2001","CXR1019_IM-0015-2001","CXR103_IM-0023-2001","CXR1030_IM-0024-1001","CXR1030_IM-0024-2001","CXR1035_IM-0028-2001","CXR1044_IM-0036-1001","CXR1047_IM-0036-1001","CXR1047_IM-0036-2001","CXR1000_IM-0003-1001","CXR1000_IM-0003-3001","CXR1001_IM-0004-1001","CXR1003_IM-0005-2002","CXR101_IM-0011-2001","CXR101_IM-0011-4004","CXR1012_IM-0013-1001","CXR1014_IM-0013-1001","CXR1014_IM-0013-2001","CXR1015_IM-0001-1001","CXR1015_IM-0001-2001","CXR1015_IM-0013-1001","CXR1015_IM-0013-2001","CXR1016_IM-0013-1001","CXR1021_IM-0017-1001-0001","CXR1034_IM-0028-1001","CXR1037_IM-0029-1001","CXR1043_IM-0035-0001-0001","CXR1043_IM-0035-0001-0002","CXR1050_IM-0038-4004","CXR1053_IM-0040-1001","CXR1053_IM-0040-3003","CXR105_IM-0037-1001","CXR1022_IM-0017-2001"}
mp,nt,y=[],[],[]
for r in G:
    if r["id"] not in FOC: continue
    p=np.array(r["probs"]); mp.append(float(p.min())); nt.append(len(p)); y.append(LAB[r["id"]])
mp=np.array(mp);nt=np.array(nt);y=np.array(y)
base=roc_auc_score(y,-mp)
print(f"focal n={len(y)} aluc={int(y.sum())}  AUC(min_prob)={base:.3f}")
print(f"ntok dentro focal: correcto={nt[y==0].mean():.1f} aluc={nt[y==1].mean():.1f}  AUC(ntok)={max(roc_auc_score(y,nt),1-roc_auc_score(y,nt)):.3f}")
# jackknife: quitar cada positivo y cada negativo
aucs=[]
for i in range(len(y)):
    m=np.ones(len(y),bool);m[i]=False
    aucs.append(roc_auc_score(y[m],-mp[m]))
aucs=np.array(aucs)
print(f"jackknife AUC: min={aucs.min():.3f} max={aucs.max():.3f} media={aucs.mean():.3f}  (rango={aucs.max()-aucs.min():.3f})")
# ¿qué pasa al quitar los 2 positivos con menor min_prob?
posord=np.argsort(mp[y==1]); posidx=np.where(y==1)[0][posord]
m=np.ones(len(y),bool); m[posidx[:2]]=False
print(f"AUC quitando los 2 alucinados más inseguros = {roc_auc_score(y[m],-mp[m]):.3f}")
print(f"min_prob alucinados ordenados: {np.round(np.sort(mp[y==1]),3)}")
print(f"min_prob correctos (10 menores): {np.round(np.sort(mp[y==0])[:10],3)}")
