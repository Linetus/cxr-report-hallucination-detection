"""NIH bbox: anclaje OBJETIVO de LLaVA-Rad. Por imagen con patología+bbox: genera informe,
   extrae heatmap de los tokens del hallazgo, mide energy-in-bbox + pointing-game + forma.
   Test: ¿la concentración predice atención FUERA de la bbox (mala localización)? GPU (llavarad)."""
import os
os.environ["HF_HOME"]=r"./hf_cache/"
import csv, json, re, time, numpy as np
from pathlib import Path
import torch
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX

NVIS=1369; G=37
MERGED=r"./hf_cache/llavarad-merged"
NIH=Path(r"./data/raw/nih")
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
OUT=SC+r"\nih_results.json"

SYN={"Atelectasis":["atelectas"],"Cardiomegaly":["cardiomeg","enlarg","cardiac silhouette"],
     "Effusion":["effusion"],"Infiltrate":["infiltrat","opacit","airspace","consolidat"],
     "Mass":["mass"],"Nodule":["nodul"],"Pneumonia":["pneumonia","consolidat"],
     "Pneumothorax":["pneumothorax"]}
NEG=re.compile(r"\b(no|without|clear|normal|free of|negative|unremarkable|within normal|intact|grossly|resolved)\b")
def pos_span(text,syns):
    t=text.lower()
    for s in syns:
        for m in re.finditer(s,t):
            if NEG.search(t[max(0,m.start()-45):m.start()]) is None: return m.start(),m.end()
    return None

tok,model,improc,_=load_pretrained_model(MERGED,None,"llavarad",load_8bit=True)
print("Modelo cargado.",flush=True)

rows=[]
with open(NIH/"BBox_List_2017.csv") as f:
    r=csv.reader(f); next(r)
    for row in r:
        if len(row)<6 or not row[0]: continue
        rows.append((row[0],row[1],float(row[2]),float(row[3]),float(row[4]),float(row[5])))

query="<image>\nDescribe the findings of the chest x-ray.\n"
def prompt_ids():
    c=conv_templates["v1"].copy(); c.append_message(c.roles[0],query); c.append_message(c.roles[1],None)
    return tokenizer_image_token(c.get_prompt(),tok,IMAGE_TOKEN_INDEX,return_tensors="pt").unsqueeze(0).cuda()
def token_spans(ids):
    spans,t="",[]; s=""
    for tid in ids.tolist():
        p=tok.decode([tid]); t.append((len(s),len(s)+len(p))); s+=p
    return t,s

SINK=np.ones((G,G),bool); SINK[:2,:2]=False   # excluir ángulo sink 2x2
ys,xs=np.mgrid[0:G,0:G]
def shape_metrics(hm):
    h=np.where(SINK,hm,0.0); m=h[SINK]
    if m.max()<=m.min(): return dict(gini=0,entropy=1,spread=0,peak=0)
    p=m/m.sum(); ent=float(-(p*np.log(p+1e-12)).sum()/np.log(len(p)))
    srt=np.sort(p);n=len(p);gini=float((2*np.arange(1,n+1)-n-1).dot(srt)/(n*srt.sum()+1e-12))
    py,px=np.unravel_index(h.argmax(),h.shape);d=np.sqrt((ys-py)**2+(xs-px)**2)
    peak=float(h[d<=3].sum()/(h.sum()+1e-12))
    w=h/(h.sum()+1e-12);cy,cx=(w*ys).sum(),(w*xs).sum();spread=float(np.sqrt((w*((ys-cy)**2+(xs-cx)**2)).sum()))
    return dict(gini=gini,entropy=ent,spread=spread,peak=peak)

results=[]; t0=time.time()
for k,(img,label,x,y,w,h) in enumerate(rows):
    p=NIH/"images"/img
    if not p.exists() or label not in SYN: continue
    try:
        image=Image.open(p).convert("RGB"); W0,H0=image.size
        it=improc.preprocess(image,return_tensors="pt")["pixel_values"][0].half().unsqueeze(0).cuda()
        pid=prompt_ids(); ipos=(pid[0]==IMAGE_TOKEN_INDEX).nonzero().item()
        stop=KeywordsStoppingCriteria(["</s>"],tok,pid)
        with torch.inference_mode():
            out=model.generate(pid,images=it,do_sample=False,max_new_tokens=60,use_cache=True,stopping_criteria=[stop])
        gen_ids=out[0,pid.shape[1]:]; n_gen=int(gen_ids.shape[0])
        if n_gen==0: continue
        spans,gen=token_spans(gen_ids)
        sp=pos_span(gen,SYN[label]); detected=sp is not None
        full=torch.cat([pid,gen_ids.unsqueeze(0)],dim=1)
        with torch.inference_mode():
            o=model(input_ids=full,images=it,output_attentions=True,return_dict=True,use_cache=False)
        seq=o.attentions[0].shape[-1]; attn=torch.stack([a[0].float().mean(0) for a in o.attentions]).mean(0)
        gs=seq-n_gen
        if detected:
            ti=[i for i,(a,b) in enumerate(spans) if a<sp[1] and b>sp[0]]
            hm=attn[[gs+i for i in ti],ipos:ipos+NVIS].mean(0)
        else:
            hm=attn[gs:gs+n_gen,ipos:ipos+NVIS].mean(0)     # global fallback
        hm=hm.cpu().numpy().reshape(G,G); hm=np.where(SINK,hm,0.0)
        # bbox (coords sobre W0xH0) -> grid 37x37
        gx0,gy0=int(x/W0*G),int(y/H0*G); gx1,gy1=int(np.ceil((x+w)/W0*G)),int(np.ceil((y+h)/H0*G))
        bm=np.zeros((G,G),bool); bm[max(gy0,0):min(gy1,G),max(gx0,0):min(gx1,G)]=True
        tot=hm.sum()+1e-12; e_in=float(hm[bm].sum()/tot); area=float(bm.sum()/(G*G))
        lift=e_in/(area+1e-9); py,px=np.unravel_index(hm.argmax(),hm.shape); point=bool(bm[py,px])
        sm=shape_metrics(hm)
        results.append(dict(img=img,label=label,detected=detected,energy_in_bbox=e_in,area=area,
                            lift=lift,pointing=point,**sm,hm=[round(float(v),4) for v in hm.flatten()]))
        del o,attn,out; torch.cuda.empty_cache()
        if k%25==0:
            json.dump(results,open(OUT,"w")); det=sum(r["detected"] for r in results)
            print(f"{k+1}/{len(rows)} ({time.time()-t0:.0f}s) n={len(results)} detect={det}",flush=True)
    except Exception as e:
        print(f"{k} ERR {type(e).__name__}: {e}",flush=True)
json.dump(results,open(OUT,"w"))
det=sum(r["detected"] for r in results)
print(f"\n✓ {len(results)} imgs, detectados={det} en {time.time()-t0:.0f}s → {OUT}")
