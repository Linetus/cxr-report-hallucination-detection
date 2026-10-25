"""¿POR QUÉ falla PadChest? Diagnóstico:
   1) ¿Dónde cae el token min_prob (contenido vs estructura) en Open-I vs PadChest?
   2) Descomponer alucinaciones PadChest: cardiomegalia-overcall vs fabricación focal → min_prob de cada una
   3) Auditar el mapeo de etiquetas PadChest (¿'fabricado' contaminado por hallazgos no mapeados?)"""
import os,sys,json,re
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from collections import Counter
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
oi=[json.loads(l) for l in open(SC+r"\allsignals_openi.jsonl",encoding="utf-8")]
pc=[json.loads(l) for l in open(SC+r"\padchest_full.jsonl",encoding="utf-8")]
rows=json.load(open(SC+r"\padchest_rows.json",encoding="utf-8"))
byid={x["id"]:x for x in rows}

def argmin_tok(r):
    p=np.array(r["probs"]); i=int(p.argmin()); toks=r["tokens"]
    return toks[i].strip() or "·", i/max(len(p),1), float(p[i])

print("=== 1) ¿QUÉ token lleva el min_prob? (contenido vs estructura) ===")
for name,data in [("Open-I",oi),("PadChest",pc)]:
    toks=Counter(); poss=[]
    for r in data:
        t,pos,_=argmin_tok(r); toks[t]+=1; poss.append(pos)
    print(f"\n {name} (n={len(data)}): posición media del min = {np.mean(poss):.2f} (0=inicio,1=final)")
    print("   tokens min más frecuentes:", toks.most_common(10))

print("\n=== 2) Alucinaciones PadChest: cardiomegalia vs focal ===")
CARD={"cardiomegaly"}
h=[x for x in rows if x["typ"]=="halluc"]
card=[x for x in h if set(x["hcats"])<=CARD]           # solo cardiomegalia
foc =[x for x in h if not (set(x["hcats"])<=CARD)]      # incluye algo focal
corr=[x for x in rows if x["typ"]=="correct"]
def mps(g): return np.array([x["min_prob"] for x in g])
print(f" correcto      n={len(corr):3d}  min_prob medio={mps(corr).mean():.3f}")
print(f" halluc CARDIO n={len(card):3d}  min_prob medio={mps(card).mean():.3f}  (sobre-diagnóstico confiado)")
print(f" halluc FOCAL  n={len(foc):3d}  min_prob medio={mps(foc).mean():.3f}  (fabricación focal)")
# ¿la fabricación focal sí baja la confianza como en Open-I?
if len(foc)>=4:
    def rankauc(a,b):
        s=np.r_[-mps(a),-mps(b)]; y=np.r_[np.zeros(len(a)),np.ones(len(b))]
        o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
        np_=len(b); return (r[y==1].sum()-np_*(np_+1)/2)/(len(a)*np_)
    print(f"  AUC(focal-fab vs correcto por min_prob)={rankauc(corr,foc):.3f}")
    print(f"  focal fabricados: {[(x['hcats'],round(x['min_prob'],2)) for x in foc]}")

print("\n=== 3) Auditar mapeo etiquetas PadChest ===")
alllabels=Counter()
for r in pc:
    for l in r.get("present_labels",[]): alllabels[l]+=1
print(" present_labels más comunes (¿se mapean a focal?):")
for l,c in alllabels.most_common(25): print(f"   {c:3d}  {l}")
