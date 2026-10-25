"""¿Cuántos casos podemos marcar como alta-prob-error SIN falsos positivos, solo con min_prob?
   = máximo recall a precisión 100% (ningún correcto marcado). Full y focal."""
import os,sys,json,numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; sys.stdout.reconfigure(encoding="utf-8")
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
GC={r["id"]:r for r in json.load(open(SC+r"\gen_conf.json"))}; LAB=json.load(open(SC+r"\labels.json"))
FOC={"CXR1001_IM-0004-1002","CXR1016_IM-0013-2001","CXR1019_IM-0015-2001","CXR103_IM-0023-2001","CXR1030_IM-0024-1001","CXR1030_IM-0024-2001","CXR1035_IM-0028-2001","CXR1044_IM-0036-1001","CXR1047_IM-0036-1001","CXR1047_IM-0036-2001","CXR1000_IM-0003-1001","CXR1000_IM-0003-3001","CXR1001_IM-0004-1001","CXR1003_IM-0005-2002","CXR101_IM-0011-2001","CXR101_IM-0011-4004","CXR1012_IM-0013-1001","CXR1014_IM-0013-1001","CXR1014_IM-0013-2001","CXR1015_IM-0001-1001","CXR1015_IM-0001-2001","CXR1015_IM-0013-1001","CXR1015_IM-0013-2001","CXR1016_IM-0013-1001","CXR1021_IM-0017-1001-0001","CXR1034_IM-0028-1001","CXR1037_IM-0029-1001","CXR1043_IM-0035-0001-0001","CXR1043_IM-0035-0001-0002","CXR1050_IM-0038-4004","CXR1053_IM-0040-1001","CXR1053_IM-0040-3003","CXR105_IM-0037-1001","CXR1022_IM-0017-2001"}
def analyze(ids,tag):
    mp=np.array([min(GC[i]["probs"]) for i in ids]); y=np.array([LAB[i] for i in ids])
    npos=int(y.sum())
    order=np.argsort(mp)  # menor min_prob = más sospechoso primero
    # máximo nº de marcados empezando por los más bajos, todos error (precisión 100%)
    clean=0
    for k in order:
        if y[k]==1: clean+=1
        else: break
    thr_clean = mp[order[clean-1]] if clean>0 else None
    print(f"\n=== {tag} (n={len(ids)}, errores={npos}) ===")
    print(f"marcando por min_prob más bajo, SIN falso positivo: {clean} casos (todos error) "
          f"→ recall {clean}/{npos}={clean/npos:.0%}" + (f", umbral min_prob<{thr_clean:.3f}" if thr_clean else ""))
    # a distintas precisiones
    for prec_target in [1.0,0.9,0.8,0.7]:
        best=0
        for thr in sorted(mp):
            flag=mp<=thr; tp=int((flag&(y==1)).sum()); fp=int((flag&(y==0)).sum())
            if flag.sum()>0 and tp/flag.sum()>=prec_target: best=max(best,tp)
        print(f"  precisión≥{prec_target:.0%}: hasta {best} errores capturados (recall {best/npos:.0%})")
analyze([i for i in GC if i in LAB],"FULL 120")
analyze([i for i in GC if i in FOC],"FOCAL 34")
