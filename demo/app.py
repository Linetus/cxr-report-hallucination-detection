"""Demo TFM — Fiabilidad visual de un informe radiológico generado por IA (LLaVA-Rad).
   3 paneles: (1) imagen + informe generado, (2) mapa de calor (DÓNDE mira, lift≈3.0),
   (3) semáforo-gradiente de confianza (CUÁNTO fiarse) calibrado sobre min_prob de contenido.
   Lanzar:  python demo/app.py     (env llavarad; carga el modelo una vez)"""
import os, json, io
os.environ.setdefault("HF_HOME", r"./hf_cache/")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
from PIL import Image
import gradio as gr
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX

MERGED = os.environ.get("LLAVARAD_MODEL_PATH", r"./hf_cache/llavarad-merged")
NVIS, G = 1369, 37
HERE = os.path.dirname(os.path.abspath(__file__))
CALIB = json.load(open(os.path.join(HERE, "semaforo_calib.json"), encoding="utf-8"))
EXDIR = os.path.join(HERE, "examples")
_mpath = os.path.join(EXDIR, "manifest.json")
MANIFEST = {m["file"]: m for m in json.load(open(_mpath, encoding="utf-8"))} if os.path.exists(_mpath) else {}

print("Cargando LLaVA-Rad (8-bit)…", flush=True)
tok, model, improc, _ = load_pretrained_model(MERGED, None, "llavarad", load_8bit=True)
model.eval()
QUERY = "<image>\nDescribe the findings of the chest x-ray.\n"
SINK = np.ones((G, G), bool); SINK[:2, :2] = False

def _prompt_ids():
    c = conv_templates["v1"].copy(); c.append_message(c.roles[0], QUERY); c.append_message(c.roles[1], None)
    return tokenizer_image_token(c.get_prompt(), tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

def _heat_overlay(pil_img, hm):
    """superpone el heatmap (37x37) sobre la imagen en escala de grises, colormap rojo-amarillo."""
    h = hm.copy(); h = (h - h.min()) / (h.ptp() + 1e-9)
    hm_img = Image.fromarray((h * 255).astype(np.uint8)).resize(pil_img.size, Image.BICUBIC)
    hn = np.asarray(hm_img, float) / 255.0
    base = np.asarray(pil_img.convert("L"), float)[..., None] / 255.0
    # colormap: azul(0) -> rojo -> amarillo(1)
    r = np.clip(1.5 * hn, 0, 1); g = np.clip(1.5 * hn - 0.5, 0, 1); b = np.clip(0.6 - hn, 0, 1)
    heat = np.stack([r, g, b], -1)
    a = (hn ** 1.3)[..., None] * 0.6
    out = (1 - a) * np.repeat(base, 3, -1) + a * heat
    return Image.fromarray((out * 255).astype(np.uint8))

def _zone(mp):
    if mp <= CALIB["P25"]: return "rojo", "#e5484d", "Revisar prioritario", CALIB["zones"]["rojo"]["halluc_rate"]
    if mp <= CALIB["P50"]: return "amarillo", "#f2b134", "Precaución", CALIB["zones"]["amarillo"]["halluc_rate"]
    if mp <= CALIB["P75"]: return "verde_claro", "#7cc576", "Fiable", CALIB["zones"]["verde_claro"]["halluc_rate"]
    return "verde_oscuro", "#2e8b57", "Muy fiable", CALIB["zones"]["verde_oscuro"]["halluc_rate"]

def _semaforo_html(mp):
    name, color, label, rate = _zone(mp)
    lo, hi = CALIB["score_min"], CALIB["score_max"]
    pos = float(np.clip((mp - lo) / (hi - lo + 1e-9), 0, 1)) * 100  # 0=izq(rojo) 100=der(verde)
    p25 = (CALIB["P25"] - lo) / (hi - lo) * 100
    p50 = (CALIB["P50"] - lo) / (hi - lo) * 100
    p75 = (CALIB["P75"] - lo) / (hi - lo) * 100
    z = CALIB["zones"]
    # Nota: Gradio aplica su propio color oscuro por defecto a span/b/td/div (mayor
    # prioridad que la herencia desde un contenedor padre), por lo que cada elemento de
    # texto necesita su color explícito en línea, no basta con fijarlo en el contenedor.
    c_title, c_sec, c_ter = "#f0f0f2", "#b8bcc0", "#9aa0a6"
    return f"""
<div style="font-family:system-ui,sans-serif;background:#1e1e22;padding:14px 12px;border-radius:10px;">
  <div style="font-weight:700;font-size:15px;margin-bottom:2px;color:{c_title};">Confianza del veredicto</div>
  <div style="font-size:12px;color:{c_sec};margin-bottom:12px;">min_prob de contenido = incertidumbre de la decisión del modelo</div>
  <div style="position:relative;height:34px;border-radius:8px;
       background:linear-gradient(90deg,#b42318 0%,#e5484d 18%,#f2b134 42%,#e8d64b 55%,#7cc576 72%,#2e8b57 100%);">
    <div style="position:absolute;left:{pos:.1f}%;top:-6px;transform:translateX(-50%);
         width:0;height:0;border-left:7px solid transparent;border-right:7px solid transparent;border-top:11px solid #fff;"></div>
    <div style="position:absolute;left:{pos:.1f}%;top:34px;transform:translateX(-50%);color:#fff;font-size:11px;">min_prob {mp:.2f}</div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:10px;margin-top:22px;">
    <span style="color:{c_ter};">más incierto (riesgo)</span><span style="color:{c_ter};">más seguro</span>
  </div>
  <div style="margin-top:14px;padding:10px;border-radius:8px;background:{color}22;border:1px solid {color};">
    <span style="font-weight:800;color:{color};font-size:16px;">● {label}</span>
    <span style="color:{c_sec};"> — P(alucinación) ≈ <b style="color:{c_title};">{rate:.0%}</b> en esta banda</span>
  </div>
  <table style="width:100%;margin-top:12px;font-size:11.5px;border-collapse:collapse;">
    <tr><td style="color:{c_ter};">banda</td><td style="color:{c_ter};">umbral min_prob</td><td style="color:{c_ter};">P(alucinación)</td></tr>
    <tr><td style="color:{c_sec};">🔴 Revisar</td><td style="color:{c_sec};">≤ {CALIB['P25']:.2f}</td><td style="color:{c_sec};">{z['rojo']['halluc_rate']:.0%}</td></tr>
    <tr><td style="color:{c_sec};">🟡 Precaución</td><td style="color:{c_sec};">{CALIB['P25']:.2f}–{CALIB['P50']:.2f}</td><td style="color:{c_sec};">{z['amarillo']['halluc_rate']:.0%}</td></tr>
    <tr><td style="color:{c_sec};">🟩 Fiable</td><td style="color:{c_sec};">{CALIB['P50']:.2f}–{CALIB['P75']:.2f}</td><td style="color:{c_sec};">{z['verde_claro']['halluc_rate']:.0%}</td></tr>
    <tr><td style="color:{c_sec};">🟢 Muy fiable</td><td style="color:{c_sec};">&gt; {CALIB['P75']:.2f}</td><td style="color:{c_sec};">{z['verde_oscuro']['halluc_rate']:.0%}</td></tr>
  </table>
  <div style="font-size:10.5px;color:{c_ter};margin-top:10px;line-height:1.4;">
    El color mide la <b style="color:{c_sec};">duda del modelo</b> (validado en Open-I, AUC 0.90). Verde = baja probabilidad de
    <i style="color:{c_sec};">alucinación</i>, pero NO garantiza ausencia de <i style="color:{c_sec};">omisiones</i> (un fallo confiado puede caer en verde).
    El mapa de calor indica <b style="color:{c_sec};">dónde</b> mira (anclaje lift≈3.0), no cuánto fiarse.
  </div>
</div>"""

@torch.inference_mode()
def analyze(pil_img):
    if pil_img is None: return None, "Sube una radiografía de tórax.", None, ""
    im = pil_img.convert("RGB")
    it = improc.preprocess(im, return_tensors="pt")["pixel_values"][0].half().unsqueeze(0).cuda()
    pid = _prompt_ids(); ipos = (pid[0] == IMAGE_TOKEN_INDEX).nonzero().item()
    stop = KeywordsStoppingCriteria(["</s>"], tok, pid)
    o = model.generate(pid, images=it, do_sample=False, max_new_tokens=70, use_cache=True,
                       stopping_criteria=[stop], output_scores=True, return_dict_in_generate=True)
    g = o.sequences[0, pid.shape[1]:]; n_gen = int(g.shape[0])
    report = tok.decode(g, skip_special_tokens=True).strip()
    probs = [float(torch.log_softmax(s[0].float(), -1).exp()[g[t]]) for t, s in enumerate(o.scores)]
    mp = float(min(probs[1:])) if len(probs) > 3 else float(min(probs))  # min_prob de CONTENIDO
    # heatmap por atención visual
    full = torch.cat([pid, g.unsqueeze(0)], 1)
    att = model(input_ids=full, images=it, output_attentions=True, return_dict=True, use_cache=False)
    seq = att.attentions[0].shape[-1]
    A = torch.stack([a[0].float().mean(0) for a in att.attentions]).mean(0)
    gs = seq - n_gen
    vb = A[gs:gs + n_gen, ipos:ipos + NVIS].mean(0).cpu().numpy()
    vb = np.where(SINK.flatten(), vb, 0.0).reshape(G, G)
    overlay = _heat_overlay(im, vb)
    return im, report, overlay, _semaforo_html(mp)

def load_example(path):
    fn = os.path.basename(path); m = MANIFEST.get(fn, {})
    v = m.get("real", "—"); g = m.get("geval", "?")
    tag = {"correct": "✅ el modelo acertó", "halluc": "❌ el modelo ALUCINÓ", "miss": "⚠️ el modelo OMITIÓ"}.get(g, "")
    return f"Radiólogo (verdad): {v}\n→ En este caso, {tag} (etiqueta G-Eval)."

with gr.Blocks(title="Fiabilidad de informes radiológicos IA") as demo:
    gr.Markdown("## Informe radiológico IA + explicabilidad visual + confianza calibrada\n"
                "Sube una radiografía de tórax. El sistema genera el informe (LLaVA-Rad), muestra **dónde mira** "
                "(mapa de calor) y **cuánto fiarse** (semáforo calibrado sobre la incertidumbre de la decisión).")
    inp = gr.Image(type="pil", label="Radiografía de tórax", height=300)
    real_box = gr.Textbox(label="Veredicto real del radiólogo (solo en ejemplos)", lines=2, interactive=False)
    if os.path.isdir(EXDIR):
        exs = [[os.path.join(EXDIR, f)] for f in sorted(os.listdir(EXDIR)) if f.endswith(".png")]
        ex_widget = gr.Examples(examples=exs, inputs=inp, label="Ejemplos (clic para cargar; incluyen la verdad)")
        ex_widget.dataset.click(lambda idx: load_example(exs[int(idx)][0]),
                                 inputs=[ex_widget.dataset], outputs=real_box)
    btn = gr.Button("Analizar", variant="primary")
    with gr.Row():
        with gr.Column():
            out_img = gr.Image(label="Imagen", height=340)
            out_rep = gr.Textbox(label="Informe generado", lines=5)
        with gr.Column():
            out_hm = gr.Image(label="Mapa de calor — DÓNDE mira (anclaje lift≈3.0)", height=340)
        with gr.Column():
            out_sem = gr.HTML()
    btn.click(analyze, inputs=inp, outputs=[out_img, out_rep, out_hm, out_sem])

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"), server_port=7860,
                inbrowser=False, allowed_paths=[HERE])
