"""Métricas de coherencia por-hallazgo: ¿separan correcto de alucinado? (scipy env)."""
import json, math, numpy as np
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

R = json.load(open(r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad/per_finding.json"))
G = 37
hms = [np.array(r["hm"], float).reshape(G, G) for r in R]

# sink data-driven
from collections import Counter
freq = Counter(np.unravel_index(h.argmax(), h.shape) for h in hms)
sinks = [p for p, c in freq.items() if c > 0.10*len(hms)]
print(f"Sinks: {sinks}")
smask = np.ones((G, G), bool)
for p in sinks: smask[p] = False

ys, xs = np.mgrid[0:G, 0:G]
def coherence(h):
    hm = np.where(smask, h, 0.0); hm = (hm-hm.min())/(hm.max()-hm.min()+1e-9)
    flat = hm.flatten()
    # pico y concentración radial
    py, px = np.unravel_index(hm.argmax(), hm.shape)
    dist = np.sqrt((ys-py)**2 + (xs-px)**2)
    peak_conc = float(hm[dist <= 3].sum() / (hm.sum()+1e-9))         # masa cerca del pico
    radial = -spearmanr(flat, dist.flatten())[0]                    # + = decae con la distancia (coherente)
    # mayor cluster (top-15%)
    m = hm >= np.percentile(hm, 85); lab, n = ndimage.label(m)
    sizes = ndimage.sum(m, lab, range(1, n+1)) if n else [0]
    max_clus = float(max(sizes)/max(m.sum(), 1))
    # Moran's I (rook)
    x = flat - flat.mean(); W = 0.0; num = 0.0
    hg = hm
    num = (x*np.roll(x.reshape(G,G),1,0).flatten()).sum() + (x*np.roll(x.reshape(G,G),-1,0).flatten()).sum() \
        + (x*np.roll(x.reshape(G,G),1,1).flatten()).sum() + (x*np.roll(x.reshape(G,G),-1,1).flatten()).sum()
    moran = float((len(x)/(4*len(x))) * num/((x**2).sum()+1e-9))
    return dict(peak_conc=peak_conc, radial=float(radial), max_clus=max_clus, moran=moran)

for r, h in zip(R, hms): r["_m"] = coherence(h)
cor = [r for r in R if r["label"]=="correcto"]; alu = [r for r in R if r["label"]=="alucinado"]
print(f"\n=== HEATMAP POR-HALLAZGO — CORRECTO (n={len(cor)}) vs ALUCINADO (n={len(alu)}) ===")
print(f"{'métrica':12}{'correcto':>10}{'alucinado':>11}{'Δ%':>8}{'AUC':>7}")
y = np.array([1]*len(cor)+[0]*len(alu))
for k in ["peak_conc","radial","max_clus","moran"]:
    c = np.mean([r['_m'][k] for r in cor]); a = np.mean([r['_m'][k] for r in alu])
    scores = np.array([r['_m'][k] for r in cor]+[r['_m'][k] for r in alu])
    auc = roc_auc_score(y, scores) if len(cor) and len(alu) else float('nan')
    print(f"{k:12}{c:>10.3f}{a:>11.3f}{100*(c-a)/(abs(a)+1e-9):>7.1f}%{auc:>7.2f}")
print("\nAUC ~0.5 = no separa | AUC lejos de 0.5 = la coherencia distingue correcto/alucinado (=> métrica de fiabilidad)")
