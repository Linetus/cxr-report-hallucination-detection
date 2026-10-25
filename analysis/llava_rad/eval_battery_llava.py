"""Batería 100 imgs: LLaVA-Rad informe+heatmap+dispersión vs real. ¿fallo→más disperso?"""
import os
os.environ["HF_HOME"] = r"./hf_cache/"
import os, glob, json, math, re, time
import xml.etree.ElementTree as ET
from pathlib import Path
import torch, numpy as np
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX

def get_meta(root_dir, n):
    root = Path(root_dir); meta, seen = [], set()
    for xml in sorted((root / "ecgen-radiology").glob("*.xml")):
        try:
            r = ET.parse(xml).getroot()
            fnd = r.findtext(".//AbstractText[@Label='FINDINGS']", default="") or ""
            imp = r.findtext(".//AbstractText[@Label='IMPRESSION']", default="") or ""
            report = f"{fnd.strip()} {imp.strip()}".strip()
            for fig in r.findall(".//parentImage"):
                iid = fig.get("id", ""); path = str(root / f"{iid}.png")
                if iid and iid not in seen and os.path.exists(path):
                    seen.add(iid); meta.append((iid, path, report))
                    if len(meta) >= n: return meta
        except Exception:
            continue
    return meta

N = 100; NVIS = 1369; GRID = 37
MERGED = r"./hf_cache/llavarad-merged"
OUT = r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad/battery.json"

FIND = ["effusion","pneumonia","consolidation","opacit","nodule","mass","atelectas","edema","emphysema",
        "cardiomegaly","enlarged","infiltrat","pneumothorax","granuloma","scarring","scoliosis","tortuous",
        "fracture","hyperinflat","hyperaerat","pectus","calcified","prominence","effusions","thickening"]
NEG = re.compile(r"\b(no|without|clear|normal|free of|negative|unremarkable|within normal|intact|grossly)\b")
def is_abnormal(text):
    t = text.lower()
    for f in FIND:
        for m in re.finditer(f, t):
            if NEG.search(t[max(0,m.start()-45):m.start()]) is None:
                return True
    return False

tokenizer, model, image_processor, context_len = load_pretrained_model(MERGED, None, "llavarad", load_8bit=True)
print("Modelo cargado.", flush=True)

meta = get_meta(r"./data/raw/open-i", N)
print(f"{len(meta)} imágenes.", flush=True)

query = "<image>\nDescribe the findings of the chest x-ray.\n"
def prompt_ids():
    c = conv_templates["v1"].copy(); c.append_message(c.roles[0], query); c.append_message(c.roles[1], None)
    return tokenizer_image_token(c.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

results = []; t0 = time.time()
for k, (iid, path, ref) in enumerate(meta):
    try:
        image = Image.open(path).convert("RGB")
        it = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].half().unsqueeze(0).cuda()
        pid = prompt_ids(); ipos = (pid[0] == IMAGE_TOKEN_INDEX).nonzero().item()
        stop = KeywordsStoppingCriteria(["</s>"], tokenizer, pid)
        with torch.inference_mode():
            out = model.generate(pid, images=it, do_sample=False, max_new_tokens=50, use_cache=True, stopping_criteria=[stop])
        gen_ids = out[0, pid.shape[1]:]; gen = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        full = torch.cat([pid, gen_ids.unsqueeze(0)], dim=1); n_gen = gen_ids.shape[0]
        with torch.inference_mode():
            o = model(input_ids=full, images=it, output_attentions=True, return_dict=True, use_cache=False)
        seq = o.attentions[0].shape[-1]
        attn = torch.stack([a[0].float().mean(0) for a in o.attentions]).mean(0)
        hm = attn[seq-n_gen:seq, ipos:ipos+NVIS].mean(0).cpu().numpy()
        del o, attn, out; torch.cuda.empty_cache()
        p = hm / (hm.sum()+1e-12); ent = float(-(p*np.log(p+1e-12)).sum() / math.log(len(p)))
        conc = float(hm.max() / (hm.mean()+1e-12))
        hmn = (hm-hm.min())/(hm.max()-hm.min()+1e-8)
        results.append({"i": k, "id": iid, "ref": ref, "gen": gen,
                        "ref_ab": is_abnormal(ref), "gen_ab": is_abnormal(gen),
                        "ent": ent, "conc": conc, "hm": [round(float(x),4) for x in hmn.tolist()]})
        if k % 10 == 0:
            json.dump(results, open(OUT, "w"))
            print(f"{k+1}/{len(meta)} ({time.time()-t0:.0f}s) ent={ent:.3f}", flush=True)
    except Exception as e:
        print(f"{k} ERROR {type(e).__name__}: {e}", flush=True)
json.dump(results, open(OUT, "w"))

def cls(r):
    if r["ref_ab"] and not r["gen_ab"]: return "falso_normal"
    if not r["ref_ab"] and r["gen_ab"]: return "falso_positivo"
    return "acierto"
print(f"\n=== RESULTADOS (n={len(results)}, {time.time()-t0:.0f}s) ===")
for c in ["acierto", "falso_normal", "falso_positivo"]:
    g = [r for r in results if cls(r) == c]
    if g: print(f"{c:15s} n={len(g):3d}  entropia={np.mean([r['ent'] for r in g]):.3f}  conc={np.mean([r['conc'] for r in g]):.1f}")
fallos = [r for r in results if cls(r) != "acierto"]; aciertos = [r for r in results if cls(r) == "acierto"]
if fallos and aciertos:
    print(f"\n→ Entropía FALLOS {np.mean([r['ent'] for r in fallos]):.3f} vs ACIERTOS {np.mean([r['ent'] for r in aciertos]):.3f}")
    print("  (si fallos > aciertos → el heatmap se dispersa en los fallos = valor de fiabilidad)")
