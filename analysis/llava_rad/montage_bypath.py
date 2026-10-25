"""Mosaicos por patología: TOP-5 y MEDIANA-5 de lift (con acierto/fallo). 8 filas x 5 col."""
import json,csv,math,sys
from pathlib import Path
import numpy as np
from collections import defaultdict
from PIL import Image, ImageDraw
sys.stdout.reconfigure(encoding="utf-8")
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
NIH=Path(r"./data/raw/nih"); OUTD=Path(r"./outputs/figs")
R=json.load(open(SC+r"\nih_results.json"))
bbox={}
with open(NIH/"BBox_List_2017.csv") as f:
    rr=csv.reader(f); next(rr)
    for row in rr:
        if len(row)>=6 and row[0]: bbox[(row[0],row[1])]=[float(x) for x in row[2:6]]
for r in R:
    bb=bbox.get((r["img"],r["label"])); r["area"]=round((bb[2]*bb[3])/(1024*1024),3) if bb else 0
T=200; CAP=26
def jet(x):
    r=np.clip(1.5-np.abs(4*x-3),0,1);g=np.clip(1.5-np.abs(4*x-2),0,1);b=np.clip(1.5-np.abs(4*x-1),0,1);return np.stack([r,g,b],-1)
def tile(rec):
    img=Image.open(NIH/"images"/rec["img"]).convert("L").resize((T,T));gray=np.asarray(img,float)/255
    hm=np.array(rec["hm"],float).reshape(37,37);hm=(hm-hm.min())/(hm.max()-hm.min()+1e-9)
    up=np.asarray(Image.fromarray((hm*255).astype("uint8")).resize((T,T),Image.BILINEAR),float)/255
    a=(0.6*up)[...,None];over=np.stack([gray]*3,-1)*(1-a)+jet(up)*a
    o=Image.fromarray((np.clip(over,0,1)*255).astype("uint8"));dr=ImageDraw.Draw(o)
    if (rec["img"],rec["label"]) in bbox:
        x,y,w,h=bbox[(rec["img"],rec["label"])];sc=T/1024
        dr.rectangle([x*sc,y*sc,(x+w)*sc,(y+h)*sc],outline=(0,255,0),width=2)
    cell=Image.new("RGB",(T,T+CAP),(20,20,20));cell.paste(o,(0,0));d=ImageDraw.Draw(cell)
    col=(120,230,120) if rec["detected"] else (240,120,120)
    d.text((3,T+2),f"{rec['label'][:11]} lift={rec['lift']:.1f}",fill=col)
    d.text((3,T+13),f"{'ACIERTO' if rec['detected'] else 'FALLO'} area={rec['area']}",fill=col)
    return cell
by=defaultdict(list)
for r in R: by[r["label"]].append(r)
def build(mode,fname):
    labs=sorted(by); rows=[]
    for lab in labs:
        g=sorted(by[lab],key=lambda r:r["lift"])
        if mode=="top": sel=g[-5:][::-1]
        else: mid=len(g)//2; sel=g[max(mid-2,0):mid+3]
        rows.append(sel)
    M=Image.new("RGB",(5*T, len(labs)*(T+CAP)),(20,20,20))
    for ri,sel in enumerate(rows):
        for ci,rec in enumerate(sel):
            M.paste(tile(rec),(ci*T, ri*(T+CAP)))
    M.save(OUTD/fname); print("guardado",OUTD/fname)
build("top","top5_bypath.png")
build("median","median5_bypath.png")
