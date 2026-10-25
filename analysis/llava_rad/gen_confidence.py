"""Incertidumbre de generación de LLaVA-Rad: log-prob + entropía predictiva por token.
   Hipótesis: al alucinar, el modelo DUDA (menor prob, mayor entropía) en los tokens del
   hallazgo. Señal ortogonal a la forma del heatmap. GPU (llavarad)."""
import os
os.environ["HF_HOME"]=r"./hf_cache/"
import json, re, time
import xml.etree.ElementTree as ET
from pathlib import Path
import torch, numpy as np
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX

N=120; MERGED=r"./hf_cache/llavarad-merged"
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
OUT=SC+r"\gen_conf.json"
tok,model,improc,_=load_pretrained_model(MERGED,None,"llavarad",load_8bit=True)
print("Modelo cargado.",flush=True)

def get_meta(root_dir,n):
    root=Path(root_dir); meta,seen=[],set()
    for xml in sorted((root/"ecgen-radiology").glob("*.xml")):
        try:
            r=ET.parse(xml).getroot()
            fnd=r.findtext(".//AbstractText[@Label='FINDINGS']",default="") or ""
            imp=r.findtext(".//AbstractText[@Label='IMPRESSION']",default="") or ""
            rep=f"{fnd.strip()} {imp.strip()}".strip()
            for fig in r.findall(".//parentImage"):
                iid=fig.get("id",""); p=str(root/f"{iid}.png")
                if iid and iid not in seen and os.path.exists(p) and len(rep)>15:
                    seen.add(iid); meta.append((iid,p,rep))
                    if len(meta)>=n: return meta
        except Exception: continue
    return meta
meta=get_meta(r"./data/raw/open-i",N)
query="<image>\nDescribe the findings of the chest x-ray.\n"
def prompt_ids():
    c=conv_templates["v1"].copy(); c.append_message(c.roles[0],query); c.append_message(c.roles[1],None)
    return tokenizer_image_token(c.get_prompt(),tok,IMAGE_TOKEN_INDEX,return_tensors="pt").unsqueeze(0).cuda()

res=[]; t0=time.time()
for k,(iid,path,ref) in enumerate(meta):
    try:
        image=Image.open(path).convert("RGB")
        it=improc.preprocess(image,return_tensors="pt")["pixel_values"][0].half().unsqueeze(0).cuda()
        pid=prompt_ids(); stop=KeywordsStoppingCriteria(["</s>"],tok,pid)
        with torch.inference_mode():
            o=model.generate(pid,images=it,do_sample=False,max_new_tokens=60,use_cache=True,
                             stopping_criteria=[stop],output_scores=True,return_dict_in_generate=True)
        gen_ids=o.sequences[0,pid.shape[1]:]
        toks=[tok.decode([t]) for t in gen_ids.tolist()]
        probs,ents=[],[]
        for t,sc in enumerate(o.scores):
            lp=torch.log_softmax(sc[0].float(),-1); p=lp.exp()
            probs.append(float(p[gen_ids[t]].item()))
            ents.append(float(-(p*lp).sum().item()))   # entropía predictiva (nats)
        res.append({"id":iid,"gen":"".join(toks),"tokens":toks,"probs":probs,"ents":ents})
        if k%20==0:
            json.dump(res,open(OUT,"w")); print(f"{k+1}/{len(meta)} ({time.time()-t0:.0f}s)",flush=True)
    except Exception as e:
        print(f"{k} ERR {type(e).__name__}: {e}",flush=True)
json.dump(res,open(OUT,"w")); print(f"\n✓ {len(res)} imgs en {time.time()-t0:.0f}s → {OUT}")
