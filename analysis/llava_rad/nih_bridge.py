"""EXPERIMENTO PUENTE: ¿la incertidumbre de generación (INFERENCIA) correlaciona con el
   anclaje objetivo (lift, con bbox) y el acierto en NIH? Heatmap GLOBAL homogéneo (sin
   confound finding-vs-global). Corre sobre allsignals_nih.jsonl (parcial o completo)."""
import os,sys,json,math,re
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
def spear(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if np.ptp(a)==0 or np.ptp(b)==0:return 0.0
    ra=rankdata(a);rb=rankdata(b);ra-=ra.mean();rb-=rb.mean()
    return float((ra*rb).sum()/math.sqrt((ra*ra).sum()*(rb*rb).sum()+1e-9))
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
F=SC+r"\allsignals_nih.jsonl"
recs=[json.loads(l) for l in open(F,encoding="utf-8")]
SYN={"Atelectasis":["atelectas"],"Cardiomegaly":["cardiomeg","enlarg","cardiac silhouette"],
     "Effusion":["effusion"],"Infiltrate":["infiltrat","opacit","airspace","consolidat"],"Mass":["mass"],
     "Nodule":["nodul"],"Pneumonia":["pneumonia","consolidat"],"Pneumothorax":["pneumothorax"]}
NEG=re.compile(r"\b(no|without|clear|normal|free of|negative|unremarkable|within normal|intact|grossly|resolved)\b")
def detected(gen,label):
    t=gen.lower()
    for s in SYN.get(label,[]):
        for m in re.finditer(s,t):
            if NEG.search(t[max(0,m.start()-45):m.start()]) is None: return True
    return False
G=37; SINK=np.ones((G,G),bool); SINK[:2,:2]=False
rows=[]
for r in recs:
    p=np.array(r["probs"]); e=np.array(r["ents"]); lab=r.get("label"); bb=r.get("bbox")
    if not bb: continue
    H=np.array(r["hm"],float).reshape(G,G); H=np.where(SINK,H,0.0)
    x,y,w,h=bb; sc=G/1024.0
    gx0,gy0=int(x*sc),int(y*sc); gx1,gy1=int(math.ceil((x+w)*sc)),int(math.ceil((y+h)*sc))
    bm=np.zeros((G,G),bool); bm[max(gy0,0):min(gy1,G),max(gx0,0):min(gx1,G)]=True
    tot=H.sum()+1e-9; ein=float(H[bm].sum()/tot); area=float(bm.sum()/(G*G)); lift=ein/(area+1e-9)
    py,px=np.unravel_index(H.argmax(),H.shape)
    rows.append(dict(det=int(detected(r["gen"],lab)),min_prob=float(p.min()),mean_ent=float(e.mean()),
        max_ent=float(e.max()),lift=lift,ein=ein,point=int(bm[py,px]),ntok=len(p)))
det=np.array([r["det"] for r in rows]); mp=np.array([r["min_prob"] for r in rows])
me=np.array([r["max_ent"] for r in rows]); lift=np.array([r["lift"] for r in rows]); ein=np.array([r["ein"] for r in rows])
print(f"n={len(rows)}  detectados(acierto)={int(det.sum())} ({det.mean():.0%})  [heatmap GLOBAL homogéneo]")
print(f"\n=== ANCLAJE (global) vs ACIERTO ===")
print(f"lift medio: acierto={lift[det==1].mean():.2f} vs miss={lift[det==0].mean():.2f}  AUC(lift->acierto)={roc_auc_score(det,lift):.3f}")
print(f"\n=== PUENTE: ¿incertidumbre (inferencia) ~ anclaje/acierto (objetivo)? ===")
print(f"AUC(min_prob->acierto)={max(roc_auc_score(det,mp),1-roc_auc_score(det,mp)):.3f}  (min_prob: acierto={mp[det==1].mean():.3f} miss={mp[det==0].mean():.3f})")
print(f"AUC(max_ent->acierto)={max(roc_auc_score(det,me),1-roc_auc_score(det,me)):.3f}")
print(f"Spearman(min_prob, lift)       = {spear(mp,lift):+.2f}")
print(f"Spearman(min_prob, energy-bbox)= {spear(mp,ein):+.2f}")
print(f"Spearman(max_ent, lift)        = {spear(me,lift):+.2f}")
