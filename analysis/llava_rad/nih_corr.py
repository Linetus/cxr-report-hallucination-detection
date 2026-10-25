"""¿Correlaciona nuestra métrica (incertidumbre) con la morfología del heatmap? NIH 984."""
import os,sys,json,math
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from scipy.stats import rankdata
def spear(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if np.ptp(a)==0 or np.ptp(b)==0:return 0.0
    ra=rankdata(a);rb=rankdata(b);ra-=ra.mean();rb-=rb.mean()
    return float((ra*rb).sum()/math.sqrt((ra*ra).sum()*(rb*rb).sum()+1e-9))
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
recs=[json.loads(l) for l in open(SC+r"\allsignals_nih.jsonl",encoding="utf-8")]
G=37; SINK=np.ones((G,G),bool); SINK[:2,:2]=False; ys,xs=np.mgrid[0:G,0:G]
rows=[]
for r in recs:
    p=np.array(r["probs"]); e=np.array(r["ents"])
    H=np.array(r["hm"],float).reshape(G,G); H=np.where(SINK,H,0.0); m=H[SINK]
    if m.max()<=m.min(): continue
    pn=m/m.sum(); ent=float(-(pn*np.log(pn+1e-12)).sum()/math.log(len(pn)))
    srt=np.sort(pn);n=len(pn);gini=float((2*np.arange(1,n+1)-n-1).dot(srt)/(n*srt.sum()+1e-12))
    py,px=np.unravel_index(H.argmax(),H.shape);d=np.sqrt((ys-py)**2+(xs-px)**2)
    peak=float(H[d<=3].sum()/(H.sum()+1e-12))
    rows.append(dict(min_prob=float(p.min()),max_ent=float(e.max()),mean_ent=float(e.mean()),
        hm_gini=gini,hm_entropy=ent,hm_peak=peak))
col=lambda f:np.array([r[f] for r in rows])
print(f"n={len(rows)}")
print("\nSpearman(MÉTRICA incertidumbre  vs  MORFOLOGÍA heatmap):")
print(f"{'':12}{'hm_gini':>9}{'hm_entr':>9}{'hm_peak':>9}")
for u in ["min_prob","max_ent","mean_ent"]:
    print(f"{u:12}"+"".join(f"{spear(col(u),col(h)):>+9.2f}" for h in ["hm_gini","hm_entropy","hm_peak"]))
print("\n(|corr|<0.15 => independientes: la incertidumbre y la forma del heatmap miden cosas distintas)")
