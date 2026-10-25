import json
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
M=json.load(open(SC+r"\pertoken_meta.json"))
# etiquetado manual (Claude-como-juez): 1 = poco fiable (halucina patología o miss grave)
UNRELIABLE = {
 "CXR1000_IM-0003-2001",  # miss: masa/opacidad LSD
 "CXR1001_IM-0004-1002",  # halluc: edema (real=fibrosis intersticial)
 "CXR1013_IM-0013-1001",  # miss: cardiomegalia leve
 "CXR1013_IM-0013-2001",  # miss: cardiomegalia leve
 "CXR1016_IM-0013-2001",  # halluc: opacidad LSI (real=derrame derecho)
 "CXR1019_IM-0015-2001",  # halluc: proceso infeccioso LSI (real=embotamiento CF derecho)
 "CXR103_IM-0023-2001",   # halluc: consolidación LID (real=pulmones limpios)
 "CXR1030_IM-0024-1001",  # halluc: densidades nodulares LSD (real=normal)
 "CXR1030_IM-0024-2001",  # halluc: neumonía LM izq (real=normal)
 "CXR1035_IM-0028-2001",  # halluc/lado: atelectasia/cicatriz LSI (real=nodular derecho)
 "CXR1021_IM-0017-1001-0002",  # miss GRAVE: neumotórax derecho grande
 "CXR1044_IM-0036-1001",  # halluc: derrame izq pequeño (real=sin derrame)
 "CXR1047_IM-0036-1001",  # halluc: opacidad LSD infección (real=limpio)
 "CXR1047_IM-0036-2001",  # halluc: atelectasia bibasilar (real=limpio)
}
lab = {r["id"]: (1 if r["id"] in UNRELIABLE else 0) for r in M}
json.dump(lab, open(SC+r"\labels.json","w"))
missing = UNRELIABLE - set(r["id"] for r in M)
print(f"total={len(lab)}  unreliable={sum(lab.values())}  reliable={len(lab)-sum(lab.values())}")
if missing: print("WARN ids no encontrados:", missing)
