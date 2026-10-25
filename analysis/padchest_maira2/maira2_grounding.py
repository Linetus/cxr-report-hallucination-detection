"""Segunda pasada MAIRA-2: GROUNDING BOXES (get_grounding=True) sobre las imágenes PadChest.
   Guarda, por hallazgo, la frase + cajas normalizadas [x1,y1,x2,y2] (mismo formato que PadChest-GR).
   Ligero (1 generación/img). Resumible. Requiere HF_TOKEN + licencia ya aceptados."""
import os
os.environ["HF_HUB_CACHE"]=r"./hf_cache/hub"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"
import json,time,sys
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
N=int(sys.argv[1]) if len(sys.argv)>1 else 380
IMGDIR=r"./data/raw/padchest_gr/extract/PadChest_GR"
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
OUT=SC+r"\maira2_grounding.jsonl"
sub=json.load(open(r"./data/raw/padchest_gr/subset_index.json"))[:N]
done=set()
if os.path.exists(OUT):
    for ln in open(OUT,encoding="utf-8"):
        try: done.add(json.loads(ln)["ImageID"])
        except Exception: pass
work=[s for s in sub if s["ImageID"] not in done]
qc=BitsAndBytesConfig(load_in_8bit=True)
model=AutoModelForCausalLM.from_pretrained("microsoft/maira-2",trust_remote_code=True,quantization_config=qc,device_map={"":0}).eval()
proc=AutoProcessor.from_pretrained("microsoft/maira-2",trust_remote_code=True)
dev=torch.device("cuda:0")
print(f"MAIRA-2 (grounding) cargado. pendientes={len(work)}",flush=True)

def serial(seq):
    """convierte la secuencia grounded a algo JSON-serializable: [{text|phrase, boxes}]"""
    out=[]
    if seq is None: return out
    for el in seq:
        if isinstance(el,str): out.append({"text":el})
        elif isinstance(el,(list,tuple)) and len(el)==2:
            phrase,boxes=el
            out.append({"phrase":phrase,"boxes":[list(b) for b in boxes] if boxes else []})
        else: out.append({"raw":str(el)})
    return out

t0=time.time()
with open(OUT,"a",encoding="utf-8") as fo:
    for k,s in enumerate(work):
        try:
            img=Image.open(os.path.join(IMGDIR,s["ImageID"])).convert("RGB")
            pin=proc.format_and_preprocess_reporting_input(current_frontal=img,current_lateral=None,prior_frontal=None,
                indication=None,technique=None,comparison=None,prior_report=None,return_tensors="pt",get_grounding=True).to(dev)
            plen=pin["input_ids"].shape[-1]
            with torch.no_grad():
                o=model.generate(**pin,max_new_tokens=450,do_sample=False,use_cache=True)
            dec=proc.decode(o[0,plen:],skip_special_tokens=True).lstrip()
            try: parsed=serial(proc.convert_output_to_plaintext_or_grounded_sequence(dec))
            except Exception as pe: parsed=[{"parse_err":str(pe)}]
            rec={"ImageID":s["ImageID"],"normal":s["normal"],"present_labels":s["present_labels"],
                 "grounded_raw":dec,"grounded":parsed}
            fo.write(json.dumps(rec,ensure_ascii=False)+"\n"); fo.flush()
            if k%15==0:
                nb=sum(len(g.get("boxes",[])) for g in parsed if isinstance(g,dict))
                print(f"{k+1}/{len(work)} ({time.time()-t0:.0f}s) cajas={nb} :: {dec[:70]}",flush=True)
            del o; torch.cuda.empty_cache()
        except Exception as e:
            print(f"{k} [{s['ImageID']}] ERR {type(e).__name__}: {e}",flush=True)
print(f"\n✓ {len(work)} en {time.time()-t0:.0f}s -> {OUT}",flush=True)
