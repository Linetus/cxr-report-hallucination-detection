"""EXTRACTOR CONSOLIDADO — todas las señales XAI en UNA pasada por imagen. Resumible (JSONL).
   Uso: python extract_all.py <openi|nih> <N|all> [K_consistency=4]
   Señales por imagen:
     uncertainty : probs + entropía predictiva por token (min_prob, etc.)
     image_dep   : masa de atención visual vs texto vs sink por token
     heatmap     : mapa visual agregado 37x37 (forma)
     faithfulness: caída de logprob de los tokens del hallazgo al ocluir la región top (comprehensiveness)
     ablation    : texto generado con imagen EN BLANCO (¿persiste el hallazgo? => prior)
     consistency : K generaciones con sampling (¿varía el hallazgo? => alucinación)
"""
import os,sys,json,re,time
os.environ["HF_HOME"]=r"./hf_cache/"
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, torch
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX

DATASET=sys.argv[1] if len(sys.argv)>1 else "openi"
NARG=sys.argv[2] if len(sys.argv)>2 else "all"
K=int(sys.argv[3]) if len(sys.argv)>3 else 4
NVIS=1369; G=37
MERGED=r"./hf_cache/llavarad-merged"
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
OUT=SC+rf"\allsignals_{DATASET}.jsonl"

FINDS={"consolidation":["consolidat","airspace"],"opacity":["opacit"],"nodule":["nodul"],"mass":["mass"],
 "effusion":["effusion"],"edema":["edema","congestion"],"atelectasis":["atelectas"],
 "cardiomegaly":["cardiomeg","enlarg"],"pneumothorax":["pneumothorax"],"infiltrate":["infiltrat"],
 "pneumonia":["pneumonia"],"fibrosis":["fibrosis","interstitial"],"tortuosity":["tortuos"],"hernia":["hernia"]}
NEG=re.compile(r"\b(no|without|clear|normal|free of|negative|unremarkable|within normal|intact|grossly|resolved)\b")
def pos_span(t,syns):
    tl=t.lower()
    for s in syns:
        for m in re.finditer(s,tl):
            if NEG.search(tl[max(0,m.start()-45):m.start()]) is None: return m.start(),m.end()
    return None
def findings_present(text):
    return set(c for c,syns in FINDS.items() if pos_span(text,syns))

tok,model,improc,_=load_pretrained_model(MERGED,None,"llavarad",load_8bit=True)
print(f"Modelo cargado. dataset={DATASET} N={NARG} K={K}",flush=True)

# --- worklist ---
def openi_list(n):
    root=Path(r"./data/raw/open-i"); out,seen=[],set()
    for xml in sorted((root/"ecgen-radiology").glob("*.xml")):
        try:
            r=ET.parse(xml).getroot()
            fnd=r.findtext(".//AbstractText[@Label='FINDINGS']",default="") or ""
            imp=r.findtext(".//AbstractText[@Label='IMPRESSION']",default="") or ""
            rep=f"{fnd.strip()} {imp.strip()}".strip()
            for fig in r.findall(".//parentImage"):
                iid=fig.get("id",""); p=str(root/f"{iid}.png")
                if iid and iid not in seen and os.path.exists(p) and len(rep)>15:
                    seen.add(iid); out.append({"id":iid,"path":p,"ref":rep})
                    if n!="all" and len(out)>=int(n): return out
        except Exception: continue
    return out
def nih_list(n):
    root=Path(r"./data/raw/nih"); out=[]
    import csv
    with open(root/"BBox_List_2017.csv") as f:
        r=csv.reader(f); next(r)
        for row in r:
            if len(row)<6 or not row[0]: continue
            p=str(root/"images"/row[0])
            if os.path.exists(p):
                out.append({"id":row[0]+"|"+row[1],"path":p,"label":row[1],
                            "bbox":[float(row[2]),float(row[3]),float(row[4]),float(row[5])]})
                if n!="all" and len(out)>=int(n): return out
    return out
work=(openi_list(NARG) if DATASET=="openi" else nih_list(NARG))
done=set()
if os.path.exists(OUT):
    for ln in open(OUT,encoding="utf-8"):
        try: done.add(json.loads(ln)["id"])
        except Exception: pass
work=[w for w in work if w["id"] not in done]
print(f"pendientes={len(work)} (ya hechos={len(done)})",flush=True)

query="<image>\nDescribe the findings of the chest x-ray.\n"
def prompt_ids():
    c=conv_templates["v1"].copy(); c.append_message(c.roles[0],query); c.append_message(c.roles[1],None)
    return tokenizer_image_token(c.get_prompt(),tok,IMAGE_TOKEN_INDEX,return_tensors="pt").unsqueeze(0).cuda()
SINK=np.ones((G,G),bool); SINK[:2,:2]=False; sink_flat=(~SINK).flatten()

def process(w):
    image=Image.open(w["path"]).convert("RGB")
    it=improc.preprocess(image,return_tensors="pt")["pixel_values"][0].half().unsqueeze(0).cuda()
    pid=prompt_ids(); ipos=(pid[0]==IMAGE_TOKEN_INDEX).nonzero().item()
    stop=KeywordsStoppingCriteria(["</s>"],tok,pid)
    # 1) greedy + scores
    with torch.inference_mode():
        o=model.generate(pid,images=it,do_sample=False,max_new_tokens=60,use_cache=True,
                         stopping_criteria=[stop],output_scores=True,return_dict_in_generate=True)
    gen_ids=o.sequences[0,pid.shape[1]:]; n_gen=int(gen_ids.shape[0])
    if n_gen==0: return None
    toks=[tok.decode([t]) for t in gen_ids.tolist()]; gen="".join(toks)
    probs,ents,base_lp=[],[],[]
    for t,sc in enumerate(o.scores):
        lp=torch.log_softmax(sc[0].float(),-1); p=lp.exp()
        probs.append(float(p[gen_ids[t]])); ents.append(float(-(p*lp).sum()))
        base_lp.append(float(lp[gen_ids[t]]))
    # 2) forward atención (teacher-forced) -> image_dep + heatmap
    full=torch.cat([pid,gen_ids.unsqueeze(0)],1)
    with torch.inference_mode():
        att=model(input_ids=full,images=it,output_attentions=True,return_dict=True,use_cache=False)
    seq=att.attentions[0].shape[-1]; A=torch.stack([a[0].float().mean(0) for a in att.attentions]).mean(0)
    gs=seq-n_gen
    visblock=A[gs:gs+n_gen, ipos:ipos+NVIS]                    # (n_gen,1369)
    vis_mass=visblock.sum(1).cpu().numpy()                     # atención al bloque visual por token
    sink_mass=visblock[:, sink_flat].sum(1).cpu().numpy()
    hm=visblock.mean(0).cpu().numpy(); hm=np.where(SINK.flatten(),hm,0.0)
    del att,A
    # tokens del hallazgo objetivo (para faithfulness): 1er hallazgo focal generado
    spans=[];s=""
    for tk in toks: spans.append((len(s),len(s)+len(tk))); s+=tk
    tgt=None
    for syns in FINDS.values():
        sp=pos_span(gen,syns)
        if sp: tgt=sp if tgt is None else (min(tgt[0],sp[0]),max(tgt[1],sp[1]))
    fi=[i for i,(a,b) in enumerate(spans) if tgt and a<tgt[1] and b>tgt[0]]
    # 3) faithfulness: ocluir región top del heatmap del hallazgo, re-forward, caída de logprob
    occ_drop=None
    if fi:
        H2=hm.reshape(G,G); py,px=np.unravel_index(H2.argmax(),H2.shape)
        Hp=it.shape[-1]; pt=Hp//G
        it2=it.clone(); y0,x0=max(py-1,0)*pt,max(px-1,0)*pt; y1,x1=min(py+2,G)*pt,min(px+2,G)*pt
        it2[...,y0:y1,x0:x1]=0.0
        with torch.inference_mode():
            lo=model(input_ids=full,images=it2,return_dict=True,use_cache=False).logits[0].float()
        drops=[]
        for i in fi:
            lp=torch.log_softmax(lo[gs+i-1],-1)[gen_ids[i]].item(); drops.append(base_lp[i]-lp)
        occ_drop=float(np.mean(drops))
    # 4) ablación imagen en blanco
    blank=torch.zeros_like(it)
    with torch.inference_mode():
        ob=model.generate(pid,images=blank,do_sample=False,max_new_tokens=60,use_cache=True,stopping_criteria=[KeywordsStoppingCriteria(["</s>"],tok,pid)])
    blank_gen="".join(tok.decode([t]) for t in ob[0,pid.shape[1]:].tolist())
    # 5) auto-consistencia (K muestras; LLaVA-Rad solo admite batch 1 => secuencial)
    samples=[]
    for _ in range(K):
        with torch.inference_mode():
            os_=model.generate(pid,images=it,do_sample=True,temperature=0.8,top_p=0.95,max_new_tokens=48,use_cache=True,stopping_criteria=[KeywordsStoppingCriteria(["</s>"],tok,pid)])
        samples.append("".join(tok.decode([t]) for t in os_[0,pid.shape[1]:].tolist()))
    torch.cuda.empty_cache()
    rec={"id":w["id"],"gen":gen,"tokens":toks,"probs":[round(x,4) for x in probs],
         "ents":[round(x,4) for x in ents],"vis_mass":[round(float(x),4) for x in vis_mass],
         "sink_mass":[round(float(x),4) for x in sink_mass],"hm":[round(float(x),4) for x in hm.tolist()],
         "occ_drop":occ_drop,"blank_gen":blank_gen,"samples":samples,
         "gen_finds":sorted(findings_present(gen)),"blank_finds":sorted(findings_present(blank_gen))}
    if DATASET=="openi": rec["ref"]=w["ref"]; rec["ref_finds"]=sorted(findings_present(w["ref"]))
    else: rec["label"]=w["label"]; rec["bbox"]=w["bbox"]
    return rec

t0=time.time(); n=0
with open(OUT,"a",encoding="utf-8") as f:
    for k,w in enumerate(work):
        try:
            rec=process(w)
            if rec: f.write(json.dumps(rec,ensure_ascii=False)+"\n"); f.flush(); n+=1
            if k%20==0: print(f"{k+1}/{len(work)} ({time.time()-t0:.0f}s) escritos={n}",flush=True)
        except Exception as e:
            print(f"{k} [{w['id']}] ERR {type(e).__name__}: {e}",flush=True)
print(f"\n✓ {n} nuevos en {time.time()-t0:.0f}s → {OUT}",flush=True)
