"""¿Qué predice el ACIERTO (detección), sin usar el heatmap (confundido)? Features de
   imagen/caso: tamaño, posición y CONSPICUIDAD (contraste lesión vs entorno).
   Foco: por qué fallan los de lift ALTO y los de MEDIANA."""
import json,csv,math,sys
from pathlib import Path
import numpy as np
from collections import defaultdict
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding="utf-8")
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
NIH=Path(r"./data/raw/nih")
R=json.load(open(SC+r"\nih_results.json"))
bbox={}
with open(NIH/"BBox_List_2017.csv") as f:
    rr=csv.reader(f); next(rr)
    for row in rr:
        if len(row)>=6 and row[0]: bbox[(row[0],row[1])]=[float(x) for x in row[2:6]]
rows=[]
for r in R:
    key=(r["img"],r["label"]);
    if key not in bbox: continue
    x,y,w,h=bbox[key]
    try: im=np.asarray(Image.open(NIH/"images"/r["img"]).convert("L").resize((512,512)),float)/255
    except Exception: continue
    sc=512/1024; X0,Y0,X1,Y1=int(x*sc),int(y*sc),int((x+w)*sc),int((y+h)*sc)
    X0,Y0=max(X0,0),max(Y0,0); X1,Y1=min(X1,512),min(Y1,512)
    if X1<=X0 or Y1<=Y0: continue
    inb=im[Y0:Y1,X0:X1]
    mx0,my0=max(X0-(X1-X0)//2,0),max(Y0-(Y1-Y0)//2,0); mx1,my1=min(X1+(X1-X0)//2,512),min(Y1+(Y1-Y0)//2,512)
    ring=im[my0:my1,mx0:mx1].copy()
    # contraste: |media lesión - media entorno| / std global
    contrast=abs(inb.mean()-im.mean())/(im.std()+1e-6)
    rows.append(dict(det=int(r["detected"]),lift=r["lift"],label=r["label"],
        area=(w*h)/(1024*1024), cx=(x+w/2)/1024, cy=(y+h/2)/1024, aspect=w/(h+1e-6),
        bmean=float(inb.mean()), imstd=float(im.std()), contrast=float(contrast),
        gini=r["gini"], entropy=r["entropy"]))
y=np.array([r["det"] for r in rows]); lift=np.array([r["lift"] for r in rows])
print(f"n={len(rows)}  aciertos={int(y.sum())} ({y.mean():.0%})")
FEATS=["area","cx","cy","aspect","bmean","imstd","contrast"]
def col(f): return np.array([r[f] for r in rows])
print("\n=== UNIVARIANTE (feature imagen -> acierto), |AUC| ===")
for f in FEATS+["lift","gini","entropy"]:
    v=col(f); a=roc_auc_score(y,v); print(f"  {f:9} AUC={max(a,1-a):.3f}  (fallo {v[y==0].mean():.3f} vs acierto {v[y==1].mean():.3f})")
# ¿por qué fallan los de LIFT ALTO?  y ¿los de MEDIANA?
for name,mask in [("LIFT ALTO (>3)",lift>3),("MEDIANA (1.5-3)",(lift>=1.5)&(lift<=3)),("LIFT BAJO (<1.5)",lift<1.5)]:
    yy=y[mask]
    print(f"\n--- {name}: n={mask.sum()} acierto={yy.mean():.0%} ---")
    for f in ["contrast","area","bmean"]:
        v=col(f)[mask]
        if 0<yy.sum()<len(yy):
            print(f"   {f:9} acierto={v[yy==1].mean():.3f} vs fallo={v[yy==0].mean():.3f}  AUC={max(roc_auc_score(yy,v),1-roc_auc_score(yy,v)):.2f}")
    from collections import Counter
    labs=np.array([r["label"] for r in rows])[mask]
    print("   patologías fallo:", dict(Counter(labs[yy==0]).most_common(4)))
