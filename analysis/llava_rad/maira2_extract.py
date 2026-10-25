"""Extractor MAIRA-2 (Microsoft, Vicuna-7B, entrenado con PadChest => IN-DISTRIBUTION).
   Saca las señales AGNÓSTICAS de nuestra fórmula: min_prob/entropía (scores), K muestras
   (auto-consistencia), blank_gen (dependencia de imagen). Sobre subconjunto de PadChest.
   Resumible (JSONL, salta ImageID). 8-bit para caber en 12 GB (RTX 4070).
   REQUISITOS: aceptar licencia en https://huggingface.co/microsoft/maira-2 + HF_TOKEN con acceso."""
import os
os.environ["HF_HUB_CACHE"]=r"./hf_cache/hub"  # modelo a E: (token queda en caché por defecto)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"
import json,time,sys
import torch,numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
K=4; N=int(sys.argv[1]) if len(sys.argv)>1 else 80   # nº imágenes (pocas: es lento)
IMGDIR=r"./data/raw/padchest_gr/extract/PadChest_GR"
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
OUT=SC+r"\maira2_padchest.jsonl"
sub=json.load(open(r"./data/raw/padchest_gr/subset_index.json"))[:N]
done=set()
if os.path.exists(OUT):
    for ln in open(OUT,encoding="utf-8"):
        try: done.add(json.loads(ln)["ImageID"])
        except Exception: pass
work=[s for s in sub if s["ImageID"] not in done]
qc=BitsAndBytesConfig(load_in_8bit=True)
model=AutoModelForCausalLM.from_pretrained("microsoft/maira-2",trust_remote_code=True,quantization_config=qc,device_map={"":0})
proc=AutoProcessor.from_pretrained("microsoft/maira-2",trust_remote_code=True)
model.eval(); dev=torch.device("cuda:0")
tok=proc.tokenizer
print(f"MAIRA-2 cargado (8-bit). pendientes={len(work)}",flush=True)

def prep(img,blank=False):
    im=Image.new("RGB",img.size,0) if blank else img
    pi=proc.format_and_preprocess_reporting_input(current_frontal=im,current_lateral=None,prior_frontal=None,
        indication=None,technique=None,comparison=None,prior_report=None,return_tensors="pt",get_grounding=False)
    return pi.to(dev)

t0=time.time()
with open(OUT,"a",encoding="utf-8") as fo:
    for k,s in enumerate(work):
        try:
            img=Image.open(os.path.join(IMGDIR,s["ImageID"])).convert("RGB")
            pin=prep(img); plen=pin["input_ids"].shape[-1]
            with torch.no_grad():
                o=model.generate(**pin,max_new_tokens=120,do_sample=False,use_cache=True,
                    output_scores=True,return_dict_in_generate=True)
            g=o.sequences[0,plen:]; toks=[tok.decode([t]) for t in g.tolist()]
            probs,ents=[],[]
            for t,scv in enumerate(o.scores):
                lp=torch.log_softmax(scv[0].float(),-1); p=lp.exp()
                probs.append(float(p[g[t]])); ents.append(float(-(p*lp).sum()))
            gen=proc.decode(g,skip_special_tokens=True).lstrip()
            # K muestras (auto-consistencia)
            samples=[]
            for _ in range(K):
                with torch.no_grad():
                    os_=model.generate(**pin,max_new_tokens=100,do_sample=True,temperature=0.8,top_p=0.95,use_cache=True)
                samples.append(proc.decode(os_[0,plen:],skip_special_tokens=True).lstrip())
            # imagen en blanco (dependencia de imagen)
            pb=prep(img,blank=True); pbl=pb["input_ids"].shape[-1]
            with torch.no_grad():
                ob=model.generate(**pb,max_new_tokens=120,do_sample=False,use_cache=True)
            blank_gen=proc.decode(ob[0,pbl:],skip_special_tokens=True).lstrip()
            rec={"ImageID":s["ImageID"],"normal":s["normal"],"present_labels":s["present_labels"],
                 "gen":gen,"tokens":toks,"probs":[round(x,4) for x in probs],"ents":[round(x,4) for x in ents],
                 "samples":samples,"blank_gen":blank_gen}
            fo.write(json.dumps(rec,ensure_ascii=False)+"\n"); fo.flush()
            if k%10==0: print(f"{k+1}/{len(work)} ({time.time()-t0:.0f}s) minp={min(probs):.2f} :: {gen[:70]}",flush=True)
            del o; torch.cuda.empty_cache()
        except Exception as e:
            print(f"{k} [{s['ImageID']}] ERR {type(e).__name__}: {e}",flush=True)
print(f"\n✓ {len(work)} en {time.time()-t0:.0f}s -> {OUT}",flush=True)
