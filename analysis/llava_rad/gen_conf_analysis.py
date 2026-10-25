"""¿La incertidumbre de generación separa alucinado de correcto? Global y en los tokens
   del hallazgo afirmado. Contra etiquetas gold (full + focal)."""
import os,sys,json,math,re,numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.metrics import roc_auc_score
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
G=json.load(open(SC+r"\gen_conf.json")); LAB=json.load(open(SC+r"\labels.json"))
FIND={"consolidation":["consolidat","airspace"],"opacity":["opacit"],"nodule":["nodul"],"mass":["mass"],
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
def finding_tok_idx(rec):
    # char spans de cada token
    spans=[];s=""
    for tk in rec["tokens"]: spans.append((len(s),len(s)+len(tk))); s+=tk
    best=None
    for syns in FIND.values():
        sp=pos_span(rec["gen"],syns)
        if sp: best=sp if best is None else (min(best[0],sp[0]),max(best[1],sp[1])); # cualquier hallazgo focal
    if best is None: return None
    return [i for i,(a,b) in enumerate(spans) if a<best[1] and b>best[0]]

rows=[]
for r in G:
    if r["id"] not in LAB: continue
    p=np.array(r["probs"]); e=np.array(r["ents"])
    fi=finding_tok_idx(r)
    fp=p[fi] if fi else p; fe=e[fi] if fi else e
    rows.append(dict(id=r["id"],y=LAB[r["id"]],
        mean_prob=p.mean(),min_prob=p.min(),mean_ent=e.mean(),max_ent=e.max(),
        low=float((p<0.5).mean()),
        f_meanprob=float(fp.mean()),f_minprob=float(fp.min()),f_maxent=float(fe.max()),f_meanent=float(fe.mean())))
ids=[r["id"] for r in rows]; y=np.array([r["y"] for r in rows])
FEATS=["mean_prob","min_prob","mean_ent","max_ent","low","f_meanprob","f_minprob","f_maxent","f_meanent"]
def show(mask,tag):
    yy=y[mask]; print(f"\n=== {tag} (n={mask.sum()}, alucina={int(yy.sum())}) ===")
    print(f"{'señal':11}{'fiable':>9}{'alucina':>9}{'AUC':>7}")
    arr={f:np.array([r[f] for r in rows])[mask] for f in FEATS}
    for f in FEATS:
        v=arr[f]; a=roc_auc_score(yy,v); a=max(a,1-a)
        print(f"{f:11}{v[yy==0].mean():>9.3f}{v[yy==1].mean():>9.3f}{a:>7.2f}")
show(np.ones(len(y),bool),"FULL")
FOC={"CXR1001_IM-0004-1002","CXR1016_IM-0013-2001","CXR1019_IM-0015-2001","CXR103_IM-0023-2001","CXR1030_IM-0024-1001","CXR1030_IM-0024-2001","CXR1035_IM-0028-2001","CXR1044_IM-0036-1001","CXR1047_IM-0036-1001","CXR1047_IM-0036-2001","CXR1000_IM-0003-1001","CXR1000_IM-0003-3001","CXR1001_IM-0004-1001","CXR1003_IM-0005-2002","CXR101_IM-0011-2001","CXR101_IM-0011-4004","CXR1012_IM-0013-1001","CXR1014_IM-0013-1001","CXR1014_IM-0013-2001","CXR1015_IM-0001-1001","CXR1015_IM-0001-2001","CXR1015_IM-0013-1001","CXR1015_IM-0013-2001","CXR1016_IM-0013-1001","CXR1021_IM-0017-1001-0001","CXR1034_IM-0028-1001","CXR1037_IM-0029-1001","CXR1043_IM-0035-0001-0001","CXR1043_IM-0035-0001-0002","CXR1050_IM-0038-4004","CXR1053_IM-0040-1001","CXR1053_IM-0040-3003","CXR105_IM-0037-1001","CXR1022_IM-0017-2001"}
show(np.array([i in FOC for i in ids]),"FOCAL")
