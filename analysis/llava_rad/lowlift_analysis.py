"""¿Qué hay en los casos de BAJO LIFT? ¿Es artefacto de caja grande o mala-localización real?
   ¿La atención se va a un sitio consistente? Analiza + renderiza figuras. PIL+numpy."""
import json,csv,sys,math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.stdout.reconfigure(encoding="utf-8")
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
NIH=Path(r"./data/raw/nih"); OUT=Path(r"./outputs/figs"); OUT.mkdir(parents=True,exist_ok=True)
G=37; D=384
R=[x for x in json.load(open(SC+r"\nih_results.json")) if x["detected"]]
bbox={}
with open(NIH/"BBox_List_2017.csv") as f:
    r=csv.reader(f); next(r)
    for row in r:
        if len(row)>=6 and row[0]: bbox[(row[0],row[1])]=[float(x) for x in row[2:6]]
def spear(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    from scipy.stats import rankdata
    ra=rankdata(a)-0;rb=rankdata(b);ra=ra-ra.mean();rb=rb-rb.mean()
    d=math.sqrt((ra*ra).sum()*(rb*rb).sum());return (ra*rb).sum()/d if d>0 else 0
lift=np.array([x["lift"] for x in R]); area=np.array([x["area"] for x in R])
ein=np.array([x["energy_in_bbox"] for x in R]); gini=np.array([x["gini"] for x in R]); ent=np.array([x["entropy"] for x in R])
# peak (fila,col) del heatmap
pk=np.array([np.unravel_index(np.array(x["hm"]).reshape(G,G).argmax(),(G,G)) for x in R])
print(f"detectados={len(R)}")
print(f"Spearman(lift, area caja) = {spear(lift,area):+.2f}   (muy negativo => bajo lift ~ caja grande)")
print(f"Spearman(lift, energy-in-bbox) = {spear(lift,ein):+.2f}")
print(f"Spearman(lift, gini heatmap) = {spear(lift,gini):+.2f}   (+ => bajo lift ~ atención difusa)")

low=lift<1.5; high=lift>=3.0
from collections import Counter
labs=np.array([x["label"] for x in R])
print(f"\nBAJO LIFT (<1.5): n={low.sum()}  | ALTO LIFT (>=3): n={high.sum()}")
print("patologías bajo-lift:", dict(Counter(labs[low]).most_common()))
print("patologías alto-lift:", dict(Counter(labs[high]).most_common()))
print(f"área caja: bajo={area[low].mean():.3f} vs alto={area[high].mean():.3f}")
print(f"gini heatmap: bajo={gini[low].mean():.3f} vs alto={gini[high].mean():.3f}  (alto gini=concentrado)")
print(f"entropía heatmap: bajo={ent[low].mean():.3f} vs alto={ent[high].mean():.3f}")
# ¿el pico se va a un sitio consistente? (centro imagen ~ (18,18))
cy,cx=pk[low,0].mean(),pk[low,1].mean()
print(f"\npico medio (fila,col) bajo-lift = ({cy:.1f},{cx:.1f}) ± ({pk[low,0].std():.1f},{pk[low,1].std():.1f})  [centro=18]")
print(f"pico medio alto-lift = ({pk[high,0].mean():.1f},{pk[high,1].mean():.1f})")
# render 6 bajo-lift variados
def jet(x):
    r=np.clip(1.5-np.abs(4*x-3),0,1);g=np.clip(1.5-np.abs(4*x-2),0,1);b=np.clip(1.5-np.abs(4*x-1),0,1);return np.stack([r,g,b],-1)
def render(rec,fn):
    img=Image.open(NIH/"images"/rec["img"]).convert("L").resize((D,D)); gray=np.asarray(img,float)/255
    hm=np.array(rec["hm"],float).reshape(G,G);hm=(hm-hm.min())/(hm.max()-hm.min()+1e-9)
    up=np.asarray(Image.fromarray((hm*255).astype("uint8")).resize((D,D),Image.BILINEAR),float)/255
    a=(0.6*up)[...,None];over=np.stack([gray]*3,-1)*(1-a)+jet(up)*a
    o=Image.fromarray((np.clip(over,0,1)*255).astype("uint8"));dr=ImageDraw.Draw(o)
    if (rec["img"],rec["label"]) in bbox:
        x,y,w,h=bbox[(rec["img"],rec["label"])];sc=D/1024
        dr.rectangle([x*sc,y*sc,(x+w)*sc,(y+h)*sc],outline=(0,255,0),width=3)
    o.save(OUT/fn)
lowrecs=sorted([x for x in R if x["lift"]<1.5],key=lambda z:z["lift"])
seen=set();sel=[]
for x in lowrecs:
    if x["label"] not in seen: sel.append(x);seen.add(x["label"])
for i,x in enumerate(sel[:6]):
    render(x,f"lowlift_{i+1}_{x['label'].lower()}.png")
    print(f"  fig lowlift {x['img']} {x['label']} lift={x['lift']:.2f} area={x['area']:.3f}")
