"""Extrae heatmap POR-HALLAZGO de LLaVA-Rad + etiqueta correcto/alucinado. GPU."""
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
OUT = r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad/per_finding.json"

# hallazgos localizables (con sinónimos) → clave canónica
FINDINGS = {
    "consolidation": ["consolidation","airspace"], "opacity": ["opacit"], "nodule": ["nodul"],
    "mass": ["mass"], "effusion": ["effusion"], "edema": ["edema","congestion"],
    "atelectasis": ["atelectas"], "cardiomegaly": ["cardiomegaly","enlarg","cardiac silhouette"],
    "pneumothorax": ["pneumothorax"], "infiltrate": ["infiltrat"], "tortuosity": ["tortuos"],
    "pneumonia": ["pneumonia"], "fibrosis": ["fibrosis","interstitial"], "effusion_thick": ["thickening"],
}
NEG = re.compile(r"\b(no|without|clear|normal|free of|negative|unremarkable|within normal|intact|grossly)\b")
def positive_span(text, syns):
    """Devuelve (start,end) del 1er hallazgo positivo (no negado) o None."""
    t = text.lower()
    for s in syns:
        for m in re.finditer(s, t):
            if NEG.search(t[max(0,m.start()-45):m.start()]) is None:
                return m.start(), m.end()
    return None

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
                if iid and iid not in seen and os.path.exists(p):
                    seen.add(iid); meta.append((iid,p,rep))
                    if len(meta)>=n: return meta
        except Exception: continue
    return meta
meta = get_meta(r"./data/raw/open-i", N)

query = "<image>\nDescribe the findings of the chest x-ray.\n"
def prompt_ids():
    c = conv_templates["v1"].copy(); c.append_message(c.roles[0],query); c.append_message(c.roles[1],None)
    return tokenizer_image_token(c.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

def token_char_spans(gen_ids):
    spans, text = [], ""
    for tid in gen_ids.tolist():
        piece = tokenizer.decode([tid]); spans.append((len(text), len(text)+len(piece))); text += piece
    return spans, text

results = []; t0 = time.time()
for k,(iid,path,ref) in enumerate(meta):
    try:
        image = Image.open(path).convert("RGB")
        it = image_processor.preprocess(image,return_tensors="pt")["pixel_values"][0].half().unsqueeze(0).cuda()
        pid = prompt_ids(); ipos = (pid[0]==IMAGE_TOKEN_INDEX).nonzero().item()
        stop = KeywordsStoppingCriteria(["</s>"],tokenizer,pid)
        with torch.inference_mode():
            out = model.generate(pid,images=it,do_sample=False,max_new_tokens=60,use_cache=True,stopping_criteria=[stop])
        gen_ids = out[0,pid.shape[1]:]; n_gen = gen_ids.shape[0]
        spans, gen = token_char_spans(gen_ids)
        # hallazgos POSITIVOS generados
        gen_findings = {}
        for canon, syns in FINDINGS.items():
            sp = positive_span(gen, syns)
            if sp: gen_findings[canon] = sp
        if not gen_findings: continue
        full = torch.cat([pid, gen_ids.unsqueeze(0)],dim=1)
        with torch.inference_mode():
            o = model(input_ids=full,images=it,output_attentions=True,return_dict=True,use_cache=False)
        seq = o.attentions[0].shape[-1]
        attn = torch.stack([a[0].float().mean(0) for a in o.attentions]).mean(0)  # (seq,seq)
        gen_start = seq - n_gen
        for canon,(cs,ce) in gen_findings.items():
            # tokens generados que cubren el hallazgo
            tok_idx = [i for i,(a,b) in enumerate(spans) if a < ce and b > cs]
            if not tok_idx: continue
            seq_pos = [gen_start+i for i in tok_idx]
            hm = attn[seq_pos, ipos:ipos+NVIS].mean(0).cpu().numpy()
            hm = (hm-hm.min())/(hm.max()-hm.min()+1e-8)
            label = "correcto" if positive_span(ref, FINDINGS[canon]) else "alucinado"
            results.append({"id":iid,"finding":canon,"label":label,"hm":[round(float(x),4) for x in hm.tolist()]})
        del o, attn, out; torch.cuda.empty_cache()
        if k%10==0:
            json.dump(results,open(OUT,"w"))
            print(f"{k+1}/{len(meta)} ({time.time()-t0:.0f}s) hallazgos={len(results)}", flush=True)
    except Exception as e:
        print(f"{k} ERROR {type(e).__name__}: {e}", flush=True)
json.dump(results,open(OUT,"w"))
nc = sum(r["label"]=="correcto" for r in results); na = sum(r["label"]=="alucinado" for r in results)
print(f"\n✓ {len(results)} hallazgos ({nc} correctos, {na} alucinados) en {time.time()-t0:.0f}s → {OUT}")
