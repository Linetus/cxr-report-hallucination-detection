"""VISTAZO PRELIMINAR a la fórmula multi-señal con lo ya generado (parcial).
   Usa las señales nuevas del extractor consolidado + etiquetas gold (primeras ~120)."""
import os,sys,json,math,re
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
SC=r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad"
LAB=json.load(open(SC+r"\labels.json"))
recs=[json.loads(l) for l in open(SC+r"\allsignals_openi.jsonl",encoding="utf-8")]
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
def finds_present(t): return set(c for c,syns in FINDS.items() if pos_span(t,syns))

rows=[]
for r in recs:
    if r["id"] not in LAB: continue
    p=np.array(r["probs"]); e=np.array(r["ents"]); vm=np.array(r["vis_mass"]); toks=r["tokens"]
    gen=r["gen"]; gf=set(r.get("gen_finds",[])); bf=set(r.get("blank_finds",[]))
    # tokens del hallazgo -> dependencia de imagen del hallazgo
    spans=[];s=""
    for tk in toks: spans.append((len(s),len(s)+len(tk))); s+=tk
    fi=[]
    for syns in FINDS.values():
        sp=pos_span(gen,syns)
        if sp: fi+=[i for i,(a,b) in enumerate(spans) if a<sp[1] and b>sp[0]]
    imgdep_find=float(vm[fi].mean()) if fi else float(vm.mean())
    # auto-consistencia: ¿los hallazgos generados se repiten en las K muestras?
    sm=[finds_present(s) for s in r.get("samples",[])]
    if gf and sm: consist=float(np.mean([np.mean([c in s for s in sm]) for c in gf]))
    else: consist=1.0
    # ablación en blanco: fracción de hallazgos que persisten sin imagen (prior de lenguaje)
    blank_persist=float(len(gf & bf)/len(gf)) if gf else 0.0
    rows.append(dict(y=LAB[r["id"]],
        min_prob=float(p.min()), max_ent=float(e.max()), mean_ent=float(e.mean()),
        imgdep_all=float(vm.mean()), imgdep_find=imgdep_find,
        occ_drop=(r["occ_drop"] if r["occ_drop"] is not None else np.nan),
        blank_persist=blank_persist, consist=consist, n_finds=len(gf)))
y=np.array([r["y"] for r in rows])
# imputar occ_drop faltante con la mediana
od=np.array([r["occ_drop"] for r in rows]); md=np.nanmedian(od); od=np.where(np.isnan(od),md,od)
for i,r in enumerate(rows): r["occ_drop"]=float(od[i])
print(f"n emparejados={len(rows)}  alucina={int(y.sum())}  (extracción parcial)")
FEATS=["min_prob","max_ent","mean_ent","imgdep_all","imgdep_find","occ_drop","blank_persist","consist"]
def col(f): return np.array([r[f] for r in rows])
print("\n=== UNIVARIANTE (|AUC|) — señal por señal ===")
print(f"{'señal':13}{'fiable':>9}{'alucina':>9}{'AUC':>7}")
for f in FEATS:
    v=col(f); a=roc_auc_score(y,v); a=max(a,1-a)
    print(f"{f:13}{v[y==0].mean():>9.3f}{v[y==1].mean():>9.3f}{a:>7.2f}")
# composite
X=np.column_stack([col(f) for f in FEATS]); cv=StratifiedKFold(5,shuffle=True,random_state=0)
lg=make_pipeline(StandardScaler(),LogisticRegression(C=0.5,penalty="l1",solver="liblinear",class_weight="balanced"))
pr=cross_val_predict(lg,X,y,cv=cv,method="predict_proba")[:,1]; a0=roc_auc_score(y,pr)
rng=np.random.default_rng(0); null=[]
for _ in range(300):
    yp=rng.permutation(y)
    try: null.append(roc_auc_score(yp,cross_val_predict(lg,X,yp,cv=StratifiedKFold(5,shuffle=True,random_state=1),method="predict_proba")[:,1]))
    except Exception: pass
null=np.array(null)
print(f"\n=== COMPOSITE multi-señal (L1) OOF AUC={a0:.3f}  p={(1+np.sum(null>=a0))/(1+len(null)):.3f} ===")
clf=LogisticRegression(C=0.5,penalty="l1",solver="liblinear",class_weight="balanced").fit(StandardScaler().fit_transform(X),y)
nz=[(FEATS[j],clf.coef_[0][j]) for j in range(len(FEATS)) if abs(clf.coef_[0][j])>1e-6]
print("señales retenidas (coef, + = ALUCINA):", ", ".join(f"{n}:{c:+.2f}" for n,c in sorted(nz,key=lambda z:-abs(z[1]))))

# --- TEST FOCAL (controla régimen) ---
FOC={"CXR1001_IM-0004-1002","CXR1016_IM-0013-2001","CXR1019_IM-0015-2001","CXR103_IM-0023-2001","CXR1030_IM-0024-1001","CXR1030_IM-0024-2001","CXR1035_IM-0028-2001","CXR1044_IM-0036-1001","CXR1047_IM-0036-1001","CXR1047_IM-0036-2001","CXR1000_IM-0003-1001","CXR1000_IM-0003-3001","CXR1001_IM-0004-1001","CXR1003_IM-0005-2002","CXR101_IM-0011-2001","CXR101_IM-0011-4004","CXR1012_IM-0013-1001","CXR1014_IM-0013-1001","CXR1014_IM-0013-2001","CXR1015_IM-0001-1001","CXR1015_IM-0001-2001","CXR1015_IM-0013-1001","CXR1015_IM-0013-2001","CXR1016_IM-0013-1001","CXR1021_IM-0017-1001-0001","CXR1034_IM-0028-1001","CXR1037_IM-0029-1001","CXR1043_IM-0035-0001-0001","CXR1043_IM-0035-0001-0002","CXR1050_IM-0038-4004","CXR1053_IM-0040-1001","CXR1053_IM-0040-3003","CXR105_IM-0037-1001","CXR1022_IM-0017-2001"}
ids=[r["id"] for r in recs if r["id"] in LAB]
fm=np.array([i in FOC for i in ids])
if fm.sum()>10:
    Xf,yf=X[fm],y[fm]
    prf=cross_val_predict(lg,Xf,yf,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1]; af=roc_auc_score(yf,prf)
    nullf=[]
    for _ in range(300):
        yp=rng.permutation(yf)
        try: nullf.append(roc_auc_score(yp,cross_val_predict(lg,Xf,yp,cv=StratifiedKFold(5,shuffle=True,random_state=1),method="predict_proba")[:,1]))
        except Exception: pass
    nullf=np.array(nullf)
    print(f"\n=== FOCAL (n={fm.sum()}, alucina={int(yf.sum())}) COMPOSITE OOF AUC={af:.3f}  p={(1+np.sum(nullf>=af))/(1+len(nullf)):.3f} ===")
    print("univar focal:", ", ".join(f"{f}:{max(roc_auc_score(yf,col(f)[fm]),1-roc_auc_score(yf,col(f)[fm])):.2f}" for f in FEATS))
    # combinación FOCADA: solo min_prob + occ_drop (2 señales robustas)
    X2=np.column_stack([col("min_prob")[fm],col("occ_drop")[fm]])
    lg2=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight="balanced",max_iter=2000))
    pr2=cross_val_predict(lg2,X2,yf,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1]; a2=roc_auc_score(yf,pr2)
    n2=[]
    for _ in range(2000):
        yp=rng.permutation(yf)
        try: n2.append(roc_auc_score(yp,cross_val_predict(lg2,X2,yp,cv=StratifiedKFold(5,shuffle=True,random_state=1),method="predict_proba")[:,1]))
        except Exception: pass
    n2=np.array(n2); print(f"\n>>> FOCAL min_prob+occ_drop (2 señales): OOF AUC={a2:.3f}  p={(1+np.sum(n2>=a2))/(1+len(n2)):.3f}")
    from scipy.stats import rankdata
    def sp(a,b):
        a=np.asarray(a,float);b=np.asarray(b,float);ra=rankdata(a);rb=rankdata(b);ra-=ra.mean();rb-=rb.mean()
        return float((ra*rb).sum()/math.sqrt((ra*ra).sum()*(rb*rb).sum()+1e-9))
    print(f"    corr(min_prob, occ_drop) focal = {sp(col('min_prob')[fm],col('occ_drop')[fm]):+.2f}  (~0 => complementarias)")
