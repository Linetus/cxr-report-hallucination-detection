"""Mosaico de los casos low-lift (heatmap+bbox+caption) para análisis visual manual."""
import json,csv,math,sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.stdout.reconfigure(encoding="utf-8")
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
NIH=Path(r"./data/raw/nih")
R={x["img"]+"|"+x["label"]:x for x in json.load(open(SC+r"\nih_results.json"))}
sel=json.load(open(SC+r"\lowlift_sel.json"))
bbox={}
with open(NIH/"BBox_List_2017.csv") as f:
    rr=csv.reader(f); next(rr)
    for row in rr:
        if len(row)>=6 and row[0]: bbox[(row[0],row[1])]=[float(x) for x in row[2:6]]
T=200; CAP=26
def jet(x):
    r=np.clip(1.5-np.abs(4*x-3),0,1);g=np.clip(1.5-np.abs(4*x-2),0,1);b=np.clip(1.5-np.abs(4*x-1),0,1);return np.stack([r,g,b],-1)
def tile(item):
    rec=R[item["img"]+"|"+item["label"]]
    img=Image.open(NIH/"images"/item["img"]).convert("L").resize((T,T));gray=np.asarray(img,float)/255
    hm=np.array(rec["hm"],float).reshape(37,37);hm=(hm-hm.min())/(hm.max()-hm.min()+1e-9)
    up=np.asarray(Image.fromarray((hm*255).astype("uint8")).resize((T,T),Image.BILINEAR),float)/255
    a=(0.6*up)[...,None];over=np.stack([gray]*3,-1)*(1-a)+jet(up)*a
    o=Image.fromarray((np.clip(over,0,1)*255).astype("uint8"));dr=ImageDraw.Draw(o)
    if (item["img"],item["label"]) in bbox:
        x,y,w,h=bbox[(item["img"],item["label"])];sc=T/1024
        dr.rectangle([x*sc,y*sc,(x+w)*sc,(y+h)*sc],outline=(0,255,0),width=2)
    cell=Image.new("RGB",(T,T+CAP),(20,20,20));cell.paste(o,(0,0))
    d=ImageDraw.Draw(cell)
    col=(120,230,120) if item["detected"] else (240,120,120)
    d.text((3,T+2),f"{item['label'][:11]} lift={item['lift']}",fill=col)
    d.text((3,T+13),f"{'ACIERTO' if item['detected'] else 'FALLO'} area={item['area']}",fill=col)
    return cell
cols=6; rows=math.ceil(len(sel)/cols)
M=Image.new("RGB",(cols*T, rows*(T+CAP)),(20,20,20))
for i,it in enumerate(sel):
    M.paste(tile(it),((i%cols)*T,(i//cols)*(T+CAP)))
OUT=Path(r"./outputs/figs/lowlift_montage.png");M.save(OUT)
print("guardado",OUT,"n=",len(sel))
