"""Sonda empírica: ¿qué definición data-driven de 'valor por token' recupera tokens
   clínicos (reveladores) SIN keywords? Compara varios esquemas mostrando sus top-tokens."""
import faulthandler; faulthandler.enable()
import json, numpy as np
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
META=json.load(open(SC+r"\pertoken_meta.json")); NPZ=np.load(SC+r"\pertoken_hm.npz")
G=37

# sink data-driven
rows=np.concatenate([NPZ[f"hm_{r['k']}"].astype(np.float64) for r in META if f"hm_{r['k']}" in NPZ.files],0)
posmean=rows.mean(0); thr=posmean.mean()+6*posmean.std()
sink=np.where(posmean>thr)[0]; nonsink=np.ones(G*G,bool); nonsink[sink]=False
print("sink:",[tuple(np.unravel_index(int(s),(G,G))) for s in sink])

# patrón medio (sink excluido, normalizado) sobre TODOS los tokens = 'atención genérica'
def norm_ns(v):
    a=v[nonsink]; a=a-a.min(); s=a.sum(); return a/s if s>0 else np.ones_like(a)/len(a)
mean_pat = norm_ns(rows.mean(0))

def schemes(hm_row):
    a = hm_row[nonsink]                       # atención sobre anatomía (sin sink)
    mass = a.sum()                            # 1) masa visual no-sink
    an = norm_ns(hm_row)                      # distribución normalizada sin sink
    conc = (an.max()/(an.mean()+1e-12))       # 2) concentración (pico/media)
    negent = np.log(len(an)) + (an*np.log(an+1e-12)).sum()   # 3) neg-entropía (localización)
    dist = np.sqrt(((an-mean_pat)**2).sum())  # 4) distintividad vs patrón genérico
    return dict(mass=mass, conc=conc, negent=negent, dist=dist, distconc=dist*conc)

# top tokens por esquema, agregando 5 imágenes de ejemplo
import collections
examples = META[:5]
for scheme in ["mass","conc","negent","dist","distconc"]:
    print(f"\n=== esquema '{scheme}' — top tokens (5 imgs) ===")
    for r in examples:
        hm=NPZ[f"hm_{r['k']}"].astype(np.float64); toks=r["tokens"]
        w=np.array([schemes(hm[i])[scheme] for i in range(len(toks))])
        top=sorted(zip(toks,w),key=lambda z:-z[1])[:7]
        print("   "+" | ".join(f"'{t.strip()}':{v:.2f}" for t,v in top))
