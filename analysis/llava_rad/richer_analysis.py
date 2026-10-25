"""Métricas ricas de heatmap (clusters/formas/dispersión), sink-excluido, acierto vs fallo."""
import json, math, numpy as np
from scipy import ndimage
from scipy.stats import kurtosis as kurt

B = json.load(open(r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad/battery.json"))
G = 37
def cls(r):
    if r["ref_ab"] and not r["gen_ab"]: return "falso_normal"
    if not r["ref_ab"] and r["gen_ab"]: return "falso_positivo"
    return "acierto"

hms = [np.array(r["hm"], dtype=float).reshape(G, G) for r in B]

# 1) DETECTAR SINK data-driven: posiciones que son argmax en >15% de las imágenes
argmax_pos = [np.unravel_index(h.argmax(), h.shape) for h in hms]
from collections import Counter
freq = Counter(argmax_pos)
sinks = [p for p, c in freq.items() if c > 0.15*len(hms)]
print(f"Sinks detectados (argmax en >15% imgs): {sinks}  frecuencias: {[freq[p] for p in sinks]}")
sink_mask = np.ones((G, G), bool)
for p in sinks: sink_mask[p] = False   # excluir

def metrics(h):
    hm = np.where(sink_mask, h, 0.0)            # excluir sink
    hm = (hm - hm.min())/(hm.max()-hm.min()+1e-9)
    p = hm.flatten()/(hm.sum()+1e-12)
    ent = -(p*np.log(p+1e-12)).sum()/math.log(len(p))          # dispersión
    srt = np.sort(p); n=len(p); gini = (2*np.arange(1,n+1)-n-1).dot(srt)/(n*srt.sum()+1e-12)
    # clusters en el top-15%
    thr = np.percentile(hm, 85); mask = hm >= thr
    lab, nclus = ndimage.label(mask)
    sizes = ndimage.sum(mask, lab, range(1, nclus+1)) if nclus else [0]
    max_clus = max(sizes)/max(mask.sum(),1)
    # dispersión espacial: std de posiciones ponderado por atención
    ys, xs = np.mgrid[0:G,0:G]; w = hm/(hm.sum()+1e-12)
    cy, cx = (w*ys).sum(), (w*xs).sum()
    spread = math.sqrt((w*((ys-cy)**2+(xs-cx)**2)).sum())
    # nº de picos (máximos locales)
    mx = ndimage.maximum_filter(hm, size=3); npk = int(((hm==mx)&(hm>np.percentile(hm,80))).sum())
    return dict(ent=ent, gini=gini, nclus=nclus, max_clus=max_clus, spread=spread, npk=npk, kurt=float(kurt(p)))

for r, h in zip(B, hms): r["_m"] = metrics(h)
aci = [r for r in B if cls(r)=="acierto"]; fal = [r for r in B if cls(r)!="acierto"]
keys = ["ent","gini","nclus","max_clus","spread","npk","kurt"]
print(f"\n=== ACIERTO (n={len(aci)}) vs FALLO (n={len(fal)}) — sink excluido ===")
print(f"{'métrica':10}{'acierto':>10}{'fallo':>10}{'Δ%':>9}")
for k in keys:
    a = np.mean([r['_m'][k] for r in aci]); f = np.mean([r['_m'][k] for r in fal])
    print(f"{k:10}{a:>10.3f}{f:>10.3f}{100*(f-a)/(abs(a)+1e-9):>8.1f}%")
