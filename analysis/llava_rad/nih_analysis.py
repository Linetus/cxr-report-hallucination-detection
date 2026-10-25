"""Anclaje OBJETIVO NIH (todas las patologías). Valida: (1) ¿atención en bbox real?
   (2) ¿la forma del heatmap (concentración/dispersión) predice el anclaje? (3) resuelve
   la tensión con el test focal Open-I (dos modos de fallo)."""
import os,sys,json,math,numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
sys.stdout.reconfigure(encoding="utf-8")
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from collections import defaultdict
def spear(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if np.ptp(a)==0 or np.ptp(b)==0:return 0.0
    ra=rankdata(a);rb=rankdata(b);ra=ra-ra.mean();rb=rb-rb.mean()
    d=math.sqrt((ra*ra).sum()*(rb*rb).sum());return float((ra*rb).sum()/d) if d>0 else 0.0
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
R=json.load(open(SC+r"\nih_results.json"))
det=[r for r in R if r["detected"]]
print(f"registros={len(R)}  detección global={len(det)}/{len(R)} ({len(det)/max(len(R),1):.2f})")

# --- por patología: detección + anclaje ---
print(f"\n{'patología':13}{'n':>4}{'det':>5}{'det%':>6}{'lift':>7}{'point':>7}")
by=defaultdict(list)
for r in R: by[r["label"]].append(r)
for lab in sorted(by):
    g=by[lab]; d=[r for r in g if r["detected"]]
    lift=np.mean([r["lift"] for r in d]) if d else 0; pt=np.mean([r["pointing"] for r in d]) if d else 0
    print(f"{lab:13}{len(g):>4}{len(d):>5}{len(d)/len(g):>6.2f}{lift:>7.2f}{pt:>7.2f}")

# --- anclaje global (detectados) ---
lift=np.array([r["lift"] for r in det]); ein=np.array([r["energy_in_bbox"] for r in det])
point=np.array([r["pointing"] for r in det],int); area=np.array([r["area"] for r in det])
gini=np.array([r["gini"] for r in det]); ent=np.array([r["entropy"] for r in det]); spr=np.array([r["spread"] for r in det])
print(f"\n=== ANCLAJE GLOBAL (detectados n={len(det)}) ===")
print(f"energy-in-bbox={ein.mean():.3f}  area={area.mean():.3f}  LIFT={lift.mean():.2f} (>1=mejor que azar)")
print(f"pointing-game={point.mean():.2f}  (azar≈{area.mean():.2f})  → x{point.mean()/max(area.mean(),1e-9):.1f}")

# --- ¿la forma predice el anclaje? ---
print(f"\n=== FORMA → ANCLAJE (Spearman con energy-in-bbox) ===")
for nm,v in [("gini(concentr.)",gini),("entropy(disper.)",ent),("spread",spr)]:
    print(f"  {nm:18} {spear(v,ein):+.3f}")
ung=(point==0).astype(int)
print(f"\n=== detector objetivo: forma → MAL-anclado (pico fuera de bbox), n_mal={ung.sum()}/{len(ung)} ===")
for nm,v in [("gini",gini),("entropy",ent),("spread",spr)]:
    a=roc_auc_score(ung,v)
    print(f"  AUC {nm:8} = {max(a,1-a):.3f}   (anclado {v[point==1].mean():.3f} vs mal {v[point==0].mean():.3f})")

# --- síntesis tensión ---
print("\n=== SÍNTESIS ===")
sg=spear(gini,ein)
print(f"NIH (lesión REAL): concentración {'↑'if sg>0 else '↓'} anclaje (gini~ein={sg:+.2f}).")
print("Open-I focal (posible lesión AUSENTE): concentración↑ alucinación (AUC 0.74).")
print("=> el MISMO patrón de forma significa cosas opuestas según exista lesión real:")
print("   con lesión, concentrarse = acertar; sin lesión, concentrarse = inventar.")
print("   La señal fiable NO es la forma sola, es el ANCLAJE (energy-in-bbox), medible aquí.")
