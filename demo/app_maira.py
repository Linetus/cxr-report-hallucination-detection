"""Demo TFM (MAIRA-2) — Fiabilidad de informes radiológicos IA.
   3 paneles: imagen + informe | cajas de anclaje (grounding propio de MAIRA) | semáforo de confianza
   calibrado sobre min_prob-contenido (MAIRA). Ejemplos con VEREDICTO REAL del radiólogo para comparar.
   Lanzar:  (env maira2)  python demo/app_maira.py    ->  http://127.0.0.1:7862"""
import os, json
for _k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY","all_proxy","ALL_PROXY"]:
    os.environ.pop(_k, None)
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"; os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"
os.environ.setdefault("HF_HUB_CACHE", r"./hf_cache/hub"); os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB = json.load(open(os.path.join(HERE, "maira_semaforo_calib.json"), encoding="utf-8"))
EXDIR = os.path.join(HERE, "examples_maira")
MANIFEST = {m["file"]: m for m in json.load(open(os.path.join(EXDIR, "manifest.json"), encoding="utf-8"))}

print("Cargando MAIRA-2 (8-bit)…", flush=True)
qc = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/maira-2", trust_remote_code=True,
                                             quantization_config=qc, device_map={"": 0}).eval()
proc = AutoProcessor.from_pretrained("microsoft/maira-2", trust_remote_code=True)
DEV = torch.device("cuda:0")

def _prep(im, grounding):
    return proc.format_and_preprocess_reporting_input(
        current_frontal=im, current_lateral=None, prior_frontal=None, indication=None,
        technique=None, comparison=None, prior_report=None, return_tensors="pt",
        get_grounding=grounding).to(DEV)

def _boxes_overlay(im, grounded):
    out = im.convert("RGB").copy(); dr = ImageDraw.Draw(out); W, H = out.size
    try: font = ImageFont.truetype("arial.ttf", max(14, W // 45))
    except Exception: font = ImageFont.load_default()
    n = 0
    for el in grounded:
        if not isinstance(el, dict): continue
        for b in el.get("boxes", []):
            x1, y1, x2, y2 = b[0] * W, b[1] * H, b[2] * W, b[3] * H
            dr.rectangle([x1, y1, x2, y2], outline=(229, 72, 77), width=max(3, W // 300))
            lab = el.get("phrase", "")[:38]
            dr.rectangle([x1, max(0, y1 - 18), x1 + 8 * len(lab), y1], fill=(229, 72, 77))
            dr.text((x1 + 2, max(0, y1 - 18)), lab, fill="white", font=font)
            n += 1
    if n == 0:
        dr.text((10, 10), "sin hallazgos localizados", fill=(120, 200, 120), font=font)
    return out

def _zone(mp):
    if mp <= CALIB["P25"]: return "#e5484d", "Revisar prioritario", CALIB["zones"]["rojo"]["halluc_rate"]
    if mp <= CALIB["P50"]: return "#f2b134", "Precaución", CALIB["zones"]["amarillo"]["halluc_rate"]
    if mp <= CALIB["P75"]: return "#7cc576", "Fiable", CALIB["zones"]["verde_claro"]["halluc_rate"]
    return "#2e8b57", "Muy fiable", CALIB["zones"]["verde_oscuro"]["halluc_rate"]

def _semaforo_html(mp):
    color, label, rate = _zone(mp)
    lo, hi = CALIB["score_min"], CALIB["score_max"]; z = CALIB["zones"]
    pos = float(np.clip((mp - lo) / (hi - lo + 1e-9), 0, 1)) * 100
    # Nota: Gradio aplica su propio color oscuro por defecto a span/b/td/div (mayor
    # prioridad que la herencia desde un contenedor padre), por lo que cada elemento de
    # texto necesita su color explícito en línea, no basta con fijarlo en el contenedor.
    c_title, c_sec, c_ter = "#f0f0f2", "#b8bcc0", "#9aa0a6"
    return f"""
<div style="font-family:system-ui,sans-serif;background:#1e1e22;padding:14px 12px;border-radius:10px;">
  <div style="font-weight:700;font-size:15px;color:{c_title};">Confianza del veredicto (MAIRA-2)</div>
  <div style="font-size:12px;color:{c_sec};margin-bottom:12px;">min_prob de contenido = incertidumbre de la decisión</div>
  <div style="position:relative;height:32px;border-radius:8px;
       background:linear-gradient(90deg,#b42318,#e5484d 20%,#f2b134 42%,#e8d64b 55%,#7cc576 72%,#2e8b57);">
    <div style="position:absolute;left:{pos:.1f}%;top:-6px;transform:translateX(-50%);width:0;height:0;
         border-left:7px solid transparent;border-right:7px solid transparent;border-top:11px solid #fff;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:10px;margin-top:4px;">
    <span style="color:{c_ter};">incierto (riesgo)</span><span style="color:{c_ter};">seguro</span></div>
  <div style="margin-top:12px;padding:10px;border-radius:8px;background:{color}22;border:1px solid {color};">
    <span style="font-weight:800;color:{color};font-size:16px;">● {label}</span>
    <span style="color:{c_sec};"> — min_prob {mp:.2f} · P(alucinación) ≈ <b style="color:{c_title};">{rate:.0%}</b></span></div>
  <table style="width:100%;margin-top:10px;font-size:11px;border-collapse:collapse;">
    <tr><td style="color:{c_ter};">banda</td><td style="color:{c_ter};">min_prob</td><td style="color:{c_ter};">P(alucina)</td></tr>
    <tr><td style="color:{c_sec};">🔴 Revisar</td><td style="color:{c_sec};">≤ {CALIB['P25']:.2f}</td><td style="color:{c_sec};">{z['rojo']['halluc_rate']:.0%}</td></tr>
    <tr><td style="color:{c_sec};">🟡 Precaución</td><td style="color:{c_sec};">{CALIB['P25']:.2f}–{CALIB['P50']:.2f}</td><td style="color:{c_sec};">{z['amarillo']['halluc_rate']:.0%}</td></tr>
    <tr><td style="color:{c_sec};">🟩 Fiable</td><td style="color:{c_sec};">{CALIB['P50']:.2f}–{CALIB['P75']:.2f}</td><td style="color:{c_sec};">{z['verde_claro']['halluc_rate']:.0%}</td></tr>
    <tr><td style="color:{c_sec};">🟢 Muy fiable</td><td style="color:{c_sec};">&gt; {CALIB['P75']:.2f}</td><td style="color:{c_sec};">{z['verde_oscuro']['halluc_rate']:.0%}</td></tr></table>
  <div style="font-size:10px;color:{c_ter};margin-top:8px;line-height:1.4;">
    Validado en 2 modelos × 2 dominios (AUC 0.80–0.90). El color mide la <b style="color:{c_sec};">duda del modelo</b>: verde = baja
    probabilidad de <i style="color:{c_sec};">alucinación</i>, NO garantiza ausencia de <i style="color:{c_sec};">omisiones</i>. Las cajas indican <b style="color:{c_sec};">dónde</b> ancla.</div>
</div>"""

@torch.inference_mode()
def analyze(pil_img):
    if pil_img is None: return None, "Sube una radiografía.", None, ""
    im = pil_img.convert("RGB")
    pin = _prep(im, False); plen = pin["input_ids"].shape[-1]
    o = model.generate(**pin, max_new_tokens=120, do_sample=False, use_cache=True,
                       output_scores=True, return_dict_in_generate=True)
    g = o.sequences[0, plen:]
    probs = [float(torch.log_softmax(s[0].float(), -1).exp()[g[t]]) for t, s in enumerate(o.scores)]
    mp = float(min(probs[1:])) if len(probs) > 3 else float(min(probs))
    report = proc.decode(g, skip_special_tokens=True).strip()
    ping = _prep(im, True); pleng = ping["input_ids"].shape[-1]
    og = model.generate(**ping, max_new_tokens=450, do_sample=False, use_cache=True)
    dec = proc.decode(og[0, pleng:], skip_special_tokens=True).strip()
    try:
        seq = proc.convert_output_to_plaintext_or_grounded_sequence(dec)
        grounded = [{"phrase": e[0], "boxes": [list(b) for b in e[1]]} if isinstance(e, (list, tuple)) and len(e) == 2
                    else {"text": e} for e in (seq or [])]
    except Exception: grounded = []
    return im, report, _boxes_overlay(im, grounded), _semaforo_html(mp)

def load_example(path):
    fn = os.path.basename(path); m = MANIFEST.get(fn, {})
    v = m.get("real", "—"); g = m.get("geval", "?")
    tag = {"correct": "✅ el modelo acertó", "halluc": "❌ el modelo ALUCINÓ", "miss": "⚠️ el modelo OMITIÓ"}.get(g, "")
    return f"Radiólogo (verdad): {v}\n→ En este caso, {tag} (etiqueta G-Eval)."

with gr.Blocks(title="Fiabilidad de informes radiológicos IA — MAIRA-2") as demo:
    gr.Markdown("## Informe radiológico IA (MAIRA-2) + anclaje + confianza calibrada\n"
                "Sube una radiografía o elige un ejemplo. Se muestra el **informe**, **dónde ancla** (cajas de MAIRA) "
                "y **cuánto fiarse** (semáforo calibrado). Los ejemplos incluyen el **veredicto real del radiólogo**.")
    inp = gr.Image(type="pil", label="Radiografía de tórax", height=280)
    real_box = gr.Textbox(label="Veredicto real del radiólogo (solo en ejemplos)", lines=2, interactive=False)
    if os.path.isdir(EXDIR):
        exs = [[os.path.join(EXDIR, f)] for f in sorted(os.listdir(EXDIR)) if f.endswith(".png")]
        ex_widget = gr.Examples(examples=exs, inputs=inp, label="Ejemplos (clic para cargar; incluyen la verdad)")
        ex_widget.dataset.click(lambda idx: load_example(exs[int(idx)][0]),
                                 inputs=[ex_widget.dataset], outputs=real_box)
    btn = gr.Button("Analizar", variant="primary")
    with gr.Row():
        with gr.Column():
            out_img = gr.Image(label="Imagen", height=330)
            out_rep = gr.Textbox(label="Informe generado por MAIRA-2", lines=5)
        with gr.Column():
            out_hm = gr.Image(label="Anclaje — cajas propias de MAIRA (DÓNDE)", height=330)
        with gr.Column():
            out_sem = gr.HTML()
    btn.click(analyze, inputs=inp, outputs=[out_img, out_rep, out_hm, out_sem])

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"), server_port=7862,
                inbrowser=False, allowed_paths=[HERE])
