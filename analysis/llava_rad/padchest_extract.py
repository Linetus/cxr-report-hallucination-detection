"""Extractor LIGERO para PadChest-GR: greedy(scores) + K muestras + texto. GPU (llavarad).
   Solo lo que la fórmula de confianza necesita (min_prob, entropía, auto-consistencia)."""
import os
os.environ["HF_HOME"]=r"./hf_cache/"
import json,time
import torch,numpy as np
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX
K=4; MERGED=r"./hf_cache/llavarad-merged"
IMGDIR=r"./data/raw/padchest_gr/extract/PadChest_GR"
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
OUT=SC+r"\padchest_signals.jsonl"
sub=json.load(open(r"./data/raw/padchest_gr/subset_index.json"))
done=set()
if os.path.exists(OUT):
    for ln in open(OUT,encoding="utf-8"):
        try: done.add(json.loads(ln)["ImageID"])
        except Exception: pass
work=[s for s in sub if s["ImageID"] not in done]
tok,model,improc,_=load_pretrained_model(MERGED,None,"llavarad",load_8bit=True)
print(f"Modelo cargado. pendientes={len(work)}",flush=True)
query="<image>\nDescribe the findings of the chest x-ray.\n"
def prompt_ids():
    c=conv_templates["v1"].copy(); c.append_message(c.roles[0],query); c.append_message(c.roles[1],None)
    return tokenizer_image_token(c.get_prompt(),tok,IMAGE_TOKEN_INDEX,return_tensors="pt").unsqueeze(0).cuda()
t0=time.time()
with open(OUT,"a",encoding="utf-8") as fo:
    for k,s in enumerate(work):
        try:
            im=Image.open(os.path.join(IMGDIR,s["ImageID"])).convert("RGB")
            it=improc.preprocess(im,return_tensors="pt")["pixel_values"][0].half().unsqueeze(0).cuda()
            pid=prompt_ids(); stop=KeywordsStoppingCriteria(["</s>"],tok,pid)
            with torch.inference_mode():
                o=model.generate(pid,images=it,do_sample=False,max_new_tokens=60,use_cache=True,stopping_criteria=[stop],output_scores=True,return_dict_in_generate=True)
            g=o.sequences[0,pid.shape[1]:]; toks=[tok.decode([t]) for t in g.tolist()]
            probs,ents=[],[]
            for t,scv in enumerate(o.scores):
                lp=torch.log_softmax(scv[0].float(),-1); p=lp.exp()
                probs.append(float(p[g[t]])); ents.append(float(-(p*lp).sum()))
            samples=[]
            for _ in range(K):
                with torch.inference_mode():
                    os_=model.generate(pid,images=it,do_sample=True,temperature=0.8,top_p=0.95,max_new_tokens=48,use_cache=True,stopping_criteria=[KeywordsStoppingCriteria(["</s>"],tok,pid)])
                samples.append("".join(tok.decode([t]) for t in os_[0,pid.shape[1]:].tolist()))
            rec={"ImageID":s["ImageID"],"normal":s["normal"],"present_labels":s["present_labels"],
                 "gen":"".join(toks),"tokens":toks,"probs":[round(x,4) for x in probs],
                 "ents":[round(x,4) for x in ents],"samples":samples}
            fo.write(json.dumps(rec,ensure_ascii=False)+"\n"); fo.flush()
            if k%25==0: print(f"{k+1}/{len(work)} ({time.time()-t0:.0f}s)",flush=True)
            del o; torch.cuda.empty_cache()
        except Exception as e:
            print(f"{k} ERR {type(e).__name__}: {e}",flush=True)
print(f"\n✓ {len(work)} en {time.time()-t0:.0f}s → {OUT}",flush=True)
