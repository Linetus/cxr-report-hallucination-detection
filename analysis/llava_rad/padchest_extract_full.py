"""Extractor COMPLETO PadChest-GR: TODAS las métricas en las 382; HEATMAP solo en 50.
   Resumible (JSONL, salta ImageID hechos). GPU (llavarad)."""
import os
os.environ["HF_HOME"]=r"./hf_cache/"
import json,re,time
import torch,numpy as np
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX
K=4; NVIS=1369; G=37; MERGED=r"./hf_cache/llavarad-merged"
IMGDIR=r"./data/raw/padchest_gr/extract/PadChest_GR"
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
OUT=SC+r"\padchest_full.jsonl"
sub=json.load(open(r"./data/raw/padchest_gr/subset_index.json"))
HM50=set(s["ImageID"] for s in sub[:50])   # heatmap solo para las primeras 50
FINDS={"consolidation":["consolidat","airspace","alveolar"],"opacity":["opacit"],"nodule":["nodul"],
 "mass":["mass"],"effusion":["effusion"],"edema":["edema","congestion"],"atelectasis":["atelectas"],
 "cardiomegaly":["cardiomeg","enlarg"],"pneumothorax":["pneumothorax"],"infiltrate":["infiltrat"],
 "pneumonia":["pneumonia"],"fibrosis":["fibros","interstitial"]}
NEG=re.compile(r"\b(no|without|clear|normal|free of|negative|unremarkable|within normal|intact|grossly|resolved)\b")
def pos_span(t,syns):
    tl=t.lower()
    for s in syns:
        for m in re.finditer(s,tl):
            if NEG.search(tl[max(0,m.start()-45):m.start()]) is None: return m.start(),m.end()
    return None
done=set()
if os.path.exists(OUT):
    for ln in open(OUT,encoding="utf-8"):
        try: done.add(json.loads(ln)["ImageID"])
        except Exception: pass
work=[s for s in sub if s["ImageID"] not in done]
tok,model,improc,_=load_pretrained_model(MERGED,None,"llavarad",load_8bit=True)
print(f"Modelo cargado. pendientes={len(work)}  (hm en {len(HM50)})",flush=True)
query="<image>\nDescribe the findings of the chest x-ray.\n"
def prompt_ids():
    c=conv_templates["v1"].copy(); c.append_message(c.roles[0],query); c.append_message(c.roles[1],None)
    return tokenizer_image_token(c.get_prompt(),tok,IMAGE_TOKEN_INDEX,return_tensors="pt").unsqueeze(0).cuda()
def tspans(toks):
    sp=[];s=""
    for t in toks: sp.append((len(s),len(s)+len(t))); s+=t
    return sp
SINK=np.ones((G,G),bool); SINK[:2,:2]=False
t0=time.time()
with open(OUT,"a",encoding="utf-8") as fo:
    for k,s in enumerate(work):
        try:
            im=Image.open(os.path.join(IMGDIR,s["ImageID"])).convert("RGB")
            it=improc.preprocess(im,return_tensors="pt")["pixel_values"][0].half().unsqueeze(0).cuda()
            pid=prompt_ids(); ipos=(pid[0]==IMAGE_TOKEN_INDEX).nonzero().item(); stop=KeywordsStoppingCriteria(["</s>"],tok,pid)
            with torch.inference_mode():
                o=model.generate(pid,images=it,do_sample=False,max_new_tokens=60,use_cache=True,stopping_criteria=[stop],output_scores=True,return_dict_in_generate=True)
            g=o.sequences[0,pid.shape[1]:]; n_gen=int(g.shape[0]); toks=[tok.decode([t]) for t in g.tolist()]; gen="".join(toks)
            probs,ents,base_lp=[],[],[]
            for t,scv in enumerate(o.scores):
                lp=torch.log_softmax(scv[0].float(),-1); p=lp.exp()
                probs.append(float(p[g[t]])); ents.append(float(-(p*lp).sum())); base_lp.append(float(lp[g[t]]))
            full=torch.cat([pid,g.unsqueeze(0)],1)
            with torch.inference_mode():
                att=model(input_ids=full,images=it,output_attentions=True,return_dict=True,use_cache=False)
            seq=att.attentions[0].shape[-1]; A=torch.stack([a[0].float().mean(0) for a in att.attentions]).mean(0)
            gs=seq-n_gen; vb=A[gs:gs+n_gen,ipos:ipos+NVIS]
            vis_mass=vb.sum(1).cpu().numpy(); sink_mass=vb[:,(~SINK).flatten()].sum(1).cpu().numpy()
            hm=vb.mean(0).cpu().numpy(); hm=np.where(SINK.flatten(),hm,0.0); del att,A
            spans=tspans(toks); tgt=None
            for syns in FINDS.values():
                sp=pos_span(gen,syns)
                if sp: tgt=sp if tgt is None else (min(tgt[0],sp[0]),max(tgt[1],sp[1]))
            fi=[i for i,(a,b) in enumerate(spans) if tgt and a<tgt[1] and b>tgt[0]]
            occ=None
            if fi:
                H2=hm.reshape(G,G); py,px=np.unravel_index(H2.argmax(),H2.shape); Hp=it.shape[-1]; pt=Hp//G
                it2=it.clone(); it2[...,max(py-1,0)*pt:min(py+2,G)*pt,max(px-1,0)*pt:min(px+2,G)*pt]=0.0
                with torch.inference_mode():
                    lo=model(input_ids=full,images=it2,return_dict=True,use_cache=False).logits[0].float()
                occ=float(np.mean([base_lp[i]-torch.log_softmax(lo[gs+i-1],-1)[g[i]].item() for i in fi]))
            blank=torch.zeros_like(it)
            with torch.inference_mode():
                ob=model.generate(pid,images=blank,do_sample=False,max_new_tokens=60,use_cache=True,stopping_criteria=[KeywordsStoppingCriteria(["</s>"],tok,pid)])
            blank_gen="".join(tok.decode([t]) for t in ob[0,pid.shape[1]:].tolist())
            samples=[]
            for _ in range(K):
                with torch.inference_mode():
                    os_=model.generate(pid,images=it,do_sample=True,temperature=0.8,top_p=0.95,max_new_tokens=48,use_cache=True,stopping_criteria=[KeywordsStoppingCriteria(["</s>"],tok,pid)])
                samples.append("".join(tok.decode([t]) for t in os_[0,pid.shape[1]:].tolist()))
            rec={"ImageID":s["ImageID"],"normal":s["normal"],"present_labels":s["present_labels"],
                 "gen":gen,"tokens":toks,"probs":[round(x,4) for x in probs],"ents":[round(x,4) for x in ents],
                 "vis_mass":[round(float(x),4) for x in vis_mass],"sink_mass":[round(float(x),4) for x in sink_mass],
                 "occ_drop":occ,"blank_gen":blank_gen,"samples":samples,
                 "hm":[round(float(x),4) for x in hm.tolist()] if s["ImageID"] in HM50 else []}
            fo.write(json.dumps(rec,ensure_ascii=False)+"\n"); fo.flush()
            if k%20==0: print(f"{k+1}/{len(work)} ({time.time()-t0:.0f}s)",flush=True)
            del o; torch.cuda.empty_cache()
        except Exception as e:
            print(f"{k} [{s['ImageID']}] ERR {type(e).__name__}: {e}",flush=True)
print(f"\n✓ {len(work)} en {time.time()-t0:.0f}s → {OUT}",flush=True)
