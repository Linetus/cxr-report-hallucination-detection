"""Batería PER-TOKEN: guarda heatmap de cada token generado (n_tok x 1369) para
   permitir ponderación + métrica compuesta de confianza offline. GPU (llavarad)."""
import os
os.environ["HF_HOME"] = r"./hf_cache/"
import glob, json, re, time
import xml.etree.ElementTree as ET
from pathlib import Path
import torch, numpy as np
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX

N = 120; NVIS = 1369; GRID = 37
MERGED = r"./hf_cache/llavarad-merged"
SC = r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
OUT_META = SC + r"\pertoken_meta.json"
OUT_NPZ  = SC + r"\pertoken_hm.npz"

# --- etiqueta de anormalidad (keyword + negación) para y ---
ABN = ["opacit","consolidation","effusion","edema","cardiomegaly","enlarg","nodul","mass",
       "atelectas","infiltrat","pneumo","fibrosis","interstitial","congestion","tortuos",
       "fracture","thickening","calcinat","granuloma","emphysema","hyperinflat","scarring","airspace"]
NEG = re.compile(r"\b(no|without|clear|normal|free of|negative|unremarkable|within normal|intact|grossly|resolved)\b")
def is_abnormal(text):
    t = text.lower()
    for s in ABN:
        for m in re.finditer(s, t):
            if NEG.search(t[max(0,m.start()-45):m.start()]) is None:
                return True
    return False

tokenizer, model, image_processor, context_len = load_pretrained_model(MERGED, None, "llavarad", load_8bit=True)
print("Modelo cargado.", flush=True)

def get_meta(root_dir, n):
    root = Path(root_dir); meta, seen = [], set()
    for xml in sorted((root / "ecgen-radiology").glob("*.xml")):
        try:
            r = ET.parse(xml).getroot()
            fnd = r.findtext(".//AbstractText[@Label='FINDINGS']", default="") or ""
            imp = r.findtext(".//AbstractText[@Label='IMPRESSION']", default="") or ""
            rep = f"{fnd.strip()} {imp.strip()}".strip()
            for fig in r.findall(".//parentImage"):
                iid = fig.get("id",""); p = str(root/f"{iid}.png")
                if iid and iid not in seen and os.path.exists(p) and len(rep) > 15:
                    seen.add(iid); meta.append((iid,p,rep))
                    if len(meta)>=n: return meta
        except Exception: continue
    return meta
meta = get_meta(r"./data/raw/open-i", N)

query = "<image>\nDescribe the findings of the chest x-ray.\n"
def prompt_ids():
    c = conv_templates["v1"].copy(); c.append_message(c.roles[0],query); c.append_message(c.roles[1],None)
    return tokenizer_image_token(c.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

records, arrays = [], {}
t0 = time.time()
for k,(iid,path,ref) in enumerate(meta):
    try:
        image = Image.open(path).convert("RGB")
        it = image_processor.preprocess(image,return_tensors="pt")["pixel_values"][0].half().unsqueeze(0).cuda()
        pid = prompt_ids(); ipos = (pid[0]==IMAGE_TOKEN_INDEX).nonzero().item()
        stop = KeywordsStoppingCriteria(["</s>"],tokenizer,pid)
        with torch.inference_mode():
            out = model.generate(pid,images=it,do_sample=False,max_new_tokens=60,use_cache=True,stopping_criteria=[stop])
        gen_ids = out[0,pid.shape[1]:]; n_gen = int(gen_ids.shape[0])
        if n_gen == 0: continue
        toks = [tokenizer.decode([t]) for t in gen_ids.tolist()]
        gen = "".join(toks)
        full = torch.cat([pid, gen_ids.unsqueeze(0)],dim=1)
        with torch.inference_mode():
            o = model(input_ids=full,images=it,output_attentions=True,return_dict=True,use_cache=False)
        seq = o.attentions[0].shape[-1]
        attn = torch.stack([a[0].float().mean(0) for a in o.attentions]).mean(0)  # (seq,seq)
        gen_start = seq - n_gen
        # heatmap CRUDO por token generado (n_gen x 1369) — SIN normalizar: preserva la masa
        # de atención (=cuánto mira ese token a la imagen), para pesar por valor offline.
        hm = attn[gen_start:gen_start+n_gen, ipos:ipos+NVIS].cpu().numpy().astype(np.float32)
        arrays[f"hm_{k}"] = hm
        records.append({"k":k,"id":iid,"gen":gen,"ref":ref,"tokens":toks,
                        "gen_ab":is_abnormal(gen),"ref_ab":is_abnormal(ref),"n_tok":n_gen})
        del o, attn, out; torch.cuda.empty_cache()
        if k%10==0:
            json.dump(records,open(OUT_META,"w")); np.savez_compressed(OUT_NPZ, **arrays)
            print(f"{k+1}/{len(meta)} ({time.time()-t0:.0f}s) recs={len(records)}", flush=True)
    except Exception as e:
        print(f"{k} ERROR {type(e).__name__}: {e}", flush=True)
json.dump(records,open(OUT_META,"w")); np.savez_compressed(OUT_NPZ, **arrays)
na=sum(r['gen_ab'] for r in records); ra=sum(r['ref_ab'] for r in records)
print(f"\n✓ {len(records)} imgs en {time.time()-t0:.0f}s | gen_ab={na} ref_ab={ra} → {OUT_META}")
