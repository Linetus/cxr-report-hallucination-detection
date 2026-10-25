"""Renderiza figuras heatmap-atención + bounding-box real (NIH). PIL+numpy, sin matplotlib."""
import json,csv,sys,os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.stdout.reconfigure(encoding="utf-8")
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
NIH=Path(r"./data/raw/nih")
OUT=Path(r"./outputs/figs"); OUT.mkdir(parents=True, exist_ok=True)
D=384
R=json.load(open(SC+r"\nih_results.json"))
bbox={}
with open(NIH/"BBox_List_2017.csv") as f:
    r=csv.reader(f); next(r)
    for row in r:
        if len(row)>=6 and row[0]: bbox[(row[0],row[1])]=[float(row[2]),float(row[3]),float(row[4]),float(row[5])]

det=[x for x in R if x["detected"]]
good=sorted([x for x in det if x["pointing"]], key=lambda z:-z["lift"])
seen=set(); gsel=[]
for x in good:
    if x["label"] not in seen: gsel.append(x); seen.add(x["label"])
    if len(gsel)>=3: break
bad=sorted([x for x in det if not x["pointing"]], key=lambda z:z["energy_in_bbox"])[:2]

def jet(x):
    r=np.clip(1.5-np.abs(4*x-3),0,1); g=np.clip(1.5-np.abs(4*x-2),0,1); b=np.clip(1.5-np.abs(4*x-1),0,1)
    return np.stack([r,g,b],-1)
def render(rec, fname):
    img=Image.open(NIH/"images"/rec["img"]).convert("L").resize((D,D))
    gray=np.asarray(img,float)/255.0
    hm=np.array(rec["hm"],float).reshape(37,37); hm=(hm-hm.min())/(hm.max()-hm.min()+1e-9)
    up=np.asarray(Image.fromarray((hm*255).astype("uint8")).resize((D,D),Image.BILINEAR),float)/255.0
    a=(0.6*up)[...,None]; base=np.stack([gray]*3,-1); over=base*(1-a)+jet(up)*a
    out=Image.fromarray((np.clip(over,0,1)*255).astype("uint8"))
    dr=ImageDraw.Draw(out)
    if (rec["img"],rec["label"]) in bbox:
        x,y,w,h=bbox[(rec["img"],rec["label"])]; sc=D/1024.0
        dr.rectangle([x*sc,y*sc,(x+w)*sc,(y+h)*sc], outline=(0,255,0), width=3)
    out.save(OUT/fname); print("->",OUT/fname, rec["label"], f"lift={rec['lift']:.1f}", "point=",rec["pointing"])

for i,x in enumerate(gsel): render(x, f"nih_good_{i+1}_{x['label'].lower()}.png")
for i,x in enumerate(bad):  render(x, f"nih_bad_{i+1}_{x['label'].lower()}.png")
print("\nSELECCION_JSON:", json.dumps({
  "good":[{"f":f"nih_good_{i+1}_{x['label'].lower()}.png","label":x["label"],"lift":round(x["lift"],1),"point":x["pointing"]} for i,x in enumerate(gsel)],
  "bad":[{"f":f"nih_bad_{i+1}_{x['label'].lower()}.png","label":x["label"],"lift":round(x["lift"],1),"ein":round(x["energy_in_bbox"],2)} for i,x in enumerate(bad)]},ensure_ascii=False))
