"""
Genera el SEGUNDO informe de seguimiento del TFM en PDF.
Continúa el primer seguimiento: foco en el avance del entrenamiento y en los
retos técnicos surgidos (ideas no plasmadas en el primer documento).
Uso: python scripts/generate_report_2.py
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

# ─── COLORES ─────────────────────────────────────────────────────────────────
AZUL_OSC  = colors.HexColor("#1a2e4a")
AZUL_MED  = colors.HexColor("#2d5986")
AZUL_CLA  = colors.HexColor("#4a90d9")
GRIS_CLA  = colors.HexColor("#f4f6f9")
GRIS_TXT  = colors.HexColor("#666666")
GRID_COL  = colors.HexColor("#d0d8e4")
VERDE     = colors.HexColor("#27ae60")
NARANJA   = colors.HexColor("#e67e22")

FECHA_STR = "Julio 2026"
FECHA_LARGA = "17 de julio de 2026"

OUTPUT_PATH = Path("outputs/tfm_seguimiento_fase2.pdf")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

W = A4[0] - 3.6 * cm

# ─── ESTILOS ─────────────────────────────────────────────────────────────────
def make(name, **kw):
    return ParagraphStyle(name, **kw)

S_TITLE  = make("Title2",  fontSize=20, fontName="Helvetica-Bold",
                textColor=AZUL_OSC, alignment=TA_CENTER, spaceAfter=6, leading=26)
S_SUB    = make("Sub",     fontSize=12, fontName="Helvetica",
                textColor=AZUL_MED, alignment=TA_CENTER, spaceAfter=3)
S_META   = make("Meta",    fontSize=9,  fontName="Helvetica",
                textColor=GRIS_TXT, alignment=TA_CENTER, spaceAfter=2)
S_H1     = make("H1",      fontSize=13, fontName="Helvetica-Bold",
                textColor=AZUL_OSC, spaceBefore=16, spaceAfter=5)
S_H2     = make("H2",      fontSize=10, fontName="Helvetica-Bold",
                textColor=AZUL_MED, spaceBefore=9, spaceAfter=3)
S_BODY   = make("Body",    fontSize=9.5, fontName="Helvetica",
                textColor=colors.black, leading=14, alignment=TA_JUSTIFY, spaceAfter=5)
S_BULL   = make("Bull",    fontSize=9.5, fontName="Helvetica",
                textColor=colors.black, leading=14, leftIndent=14, spaceAfter=3)
S_CAP    = make("Cap",     fontSize=8,  fontName="Helvetica-Oblique",
                textColor=GRIS_TXT, alignment=TA_CENTER, spaceAfter=4)
S_CELL   = make("Cell",    fontSize=8.5, fontName="Helvetica",
                textColor=colors.black, leading=12, spaceAfter=0, spaceBefore=0)
S_CELL_B = make("CellB",   fontSize=8.5, fontName="Helvetica-Bold",
                textColor=colors.black, leading=12, spaceAfter=0, spaceBefore=0)
S_CELL_H = make("CellH",   fontSize=8.5, fontName="Helvetica-Bold",
                textColor=colors.white, leading=12, spaceAfter=0, spaceBefore=0,
                alignment=TA_CENTER)
S_CELL_C = make("CellC",   fontSize=8.5, fontName="Helvetica",
                textColor=colors.black, leading=12, spaceAfter=0, spaceBefore=0,
                alignment=TA_CENTER)

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def hr(color=AZUL_CLA, t=0.7):
    return HRFlowable(width="100%", thickness=t, color=color, spaceAfter=7)

def sp(n=1):
    return Spacer(1, n * 0.32 * cm)

def b(txt):   return Paragraph(txt, S_BODY)
def h1(txt):  return Paragraph(txt, S_H1)
def h2(txt):  return Paragraph(txt, S_H2)
def bul(txt): return Paragraph(f"• {txt}", S_BULL)

def pc(txt, bold=False):
    return Paragraph(txt, S_CELL_B if bold else S_CELL)

def ph(txt):
    return Paragraph(txt, S_CELL_H)

def pcc(txt):
    return Paragraph(txt, S_CELL_C)

BASE_TABLE = TableStyle([
    ("BACKGROUND",    (0, 0), (-1,  0), AZUL_MED),
    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, GRIS_CLA]),
    ("GRID",          (0, 0), (-1, -1), 0.4, GRID_COL),
    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING",    (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING",   (0, 0), (-1, -1), 7),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
])

# ─── CABECERAS / PIES DE PÁGINA ───────────────────────────────────────────────
def on_first_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(AZUL_OSC)
    canvas.rect(0, A4[1] - 3.2*cm, A4[0], 3.2*cm, fill=1, stroke=0)
    canvas.setFillColor(AZUL_CLA)
    canvas.rect(0, A4[1] - 3.4*cm, A4[0], 0.22*cm, fill=1, stroke=0)
    _footer(canvas, doc)
    canvas.restoreState()

def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(AZUL_OSC)
    canvas.rect(0, A4[1] - 1.0*cm, A4[0], 1.0*cm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(1.8*cm, A4[1] - 0.68*cm,
                      "TFM — Generación de Informes Radiológicos con XAI")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(A4[0] - 1.8*cm, A4[1] - 0.68*cm,
                           f"Daniel Gómez Arranz · {FECHA_STR}")
    _footer(canvas, doc)
    canvas.restoreState()

def _footer(canvas, doc):
    canvas.setFillColor(GRIS_CLA)
    canvas.rect(0, 0, A4[0], 0.8*cm, fill=1, stroke=0)
    canvas.setFillColor(GRIS_TXT)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawCentredString(A4[0] / 2, 0.27*cm, f"Página {doc.page}")

# ─── TABLA GENÉRICA ──────────────────────────────────────────────────────────
def tabla(data, col_widths, extra_styles=None):
    ts = list(BASE_TABLE._cmds)
    if extra_styles:
        ts += extra_styles
    return Table(data, colWidths=col_widths, style=TableStyle(ts),
                 repeatRows=1)

# ─── BADGE HELPER ────────────────────────────────────────────────────────────
def _badge(label, color):
    return Table([[Paragraph(label, make(
        "_bl", fontSize=7.5, fontName="Helvetica-Bold",
        textColor=colors.white, alignment=TA_CENTER,
        leading=10, spaceAfter=0, spaceBefore=0))]],
        colWidths=[2.6*cm],
        style=TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), color),
            ("TOPPADDING",    (0,0),(-1,-1), 3),
            ("BOTTOMPADDING", (0,0),(-1,-1), 3),
            ("LEFTPADDING",   (0,0),(-1,-1), 2),
            ("RIGHTPADDING",  (0,0),(-1,-1), 2),
        ]))

# ─── CONTENIDO ───────────────────────────────────────────────────────────────
def story():
    s = []

    # ══ PORTADA ══════════════════════════════════════════════════════════════
    s += [
        Spacer(1, 3.5*cm),
        Paragraph("SEGUNDO INFORME DE SEGUIMIENTO — TFM", make(
            "_tag", fontSize=9, fontName="Helvetica-Bold",
            textColor=AZUL_CLA, alignment=TA_CENTER, spaceAfter=10)),
        Paragraph(
            "Generación de Informes Radiológicos<br/>"
            "mediante Arquitecturas Encoder-Decoder<br/>"
            "Multimodales con Explicabilidad Visual",
            S_TITLE),
        sp(0.4),
        hr(AZUL_CLA, 1.4),
        sp(0.4),
        Paragraph("Máster Universitario en Inteligencia Artificial", S_SUB),
        sp(0.2),
        Paragraph("Autor: Daniel Gómez Arranz · dagoar@outlook.es", S_META),
        Paragraph("Convocatoria: Septiembre 2026", S_META),
        Paragraph(f"Fecha: {FECHA_LARGA}", S_META),
        sp(1.6),
        Paragraph(
            "Este segundo informe da continuidad al primer seguimiento. No reintroduce la "
            "arquitectura, el dataset ni el plan de trabajo ya descritos: se centra en el "
            "<b>avance del entrenamiento de los modelos</b>, actualmente en curso, y en los "
            "<b>retos técnicos surgidos</b> durante su implementación y ejecución que no "
            "estaban recogidos en el documento anterior.",
            make("_lead", fontSize=9.5, fontName="Helvetica", textColor=GRIS_TXT,
                 alignment=TA_CENTER, leading=14, leftIndent=1.2*cm, rightIndent=1.2*cm)),
        sp(1.0),
    ]

    avance = [
        [ph("HITO"), ph("SITUACIÓN EN ESTE SEGUIMIENTO")],
        [pc("Stage 1 — contrastivo"),   _badge("COMPLETADO", VERDE)],
        [pc("Stage 2 — generación"),    _badge("EN CURSO", NARANJA)],
        [pc("Evaluación / Fase XAI"),   _badge("SIGUIENTE", AZUL_MED)],
    ]
    s.append(tabla(avance, [7.5*cm, 8.5*cm]))
    s.append(PageBreak())

    # ══ 1. OBJETO DE ESTE SEGUIMIENTO ═════════════════════════════════════════
    s += [
        h1("1. Objeto de este segundo seguimiento"),
        hr(),
        b("El primer informe de seguimiento dejó definidos la arquitectura encoder-decoder, "
          "el pipeline de datos sobre Open-I y el plan de análisis de explicabilidad. Desde "
          "entonces el trabajo se ha centrado en la <b>ejecución del entrenamiento</b>: el "
          "alineamiento visual-textual (Stage 1) ha concluido y el ajuste generativo (Stage 2) "
          "se encuentra en marcha."),
        b("El propósito de este documento es doble: (1) confirmar que <b>se continúa con el "
          "entrenamiento de los modelos</b> según el plan previsto, y (2) documentar los "
          "<b>retos técnicos</b> que han aparecido al pasar del diseño a la ejecución real. "
          "Son cuestiones de implementación que no eran visibles en la fase de diseño y que, "
          "por su valor metodológico, conviene dejar registradas."),
    ]

    # ══ 2. ESTADO DEL ENTRENAMIENTO ═══════════════════════════════════════════
    s += [
        sp(0.4),
        h1("2. Estado del entrenamiento"),
        hr(),
        b("El entrenamiento sigue el esquema en dos etapas previsto. <b>Stage 1</b> "
          "(alineamiento contrastivo del encoder visual con el texto) ha completado su "
          "convergencia y su checkpoint se utiliza como punto de partida del encoder en "
          "Stage 2. <b>Stage 2</b> (ajuste del decoder para la generación de informes) está "
          "actualmente en ejecución sobre el hardware local."),
        b("El seguimiento del entrenamiento se realiza mediante <b>val/rouge_l por epoch</b>, "
          "que refleja la calidad del texto generado mejor que la pérdida de entrenamiento. "
          "El proceso avanza de forma progresiva epoch a epoch; los resultados finales sobre "
          "el conjunto de test (BLEU-1/4, ROUGE-L, METEOR) se reportarán una vez alcanzada la "
          "convergencia."),
        sp(0.3),
    ]

    estado = [
        [ph("ETAPA"), ph("OBJETIVO"), ph("ESTADO")],
        [pc("Stage 1 — contrastivo (InfoNCE)"),
         pc("Alinear imagen e informe en el espacio latente"),
         _badge("COMPLETADO", VERDE)],
        [pc("Stage 2 — generación (LM)"),
         pc("Generar el informe token a token condicionado en la imagen"),
         _badge("EN CURSO", NARANJA)],
        [pc("Evaluación en test"),
         pc("Métricas NLP sobre partición reservada"),
         _badge("PENDIENTE", AZUL_MED)],
    ]
    s.append(tabla(estado, [5.0*cm, 8.2*cm, 3.0*cm]))
    s += [sp(0.3), Paragraph("Tabla 1. Situación de las etapas de entrenamiento.", S_CAP)]

    # ══ 3. RETOS TÉCNICOS SURGIDOS ════════════════════════════════════════════
    s += [
        sp(0.5),
        h1("3. Retos técnicos surgidos durante la ejecución"),
        hr(),
        b("El paso del diseño a la ejecución hizo aflorar varias dificultades no anticipadas "
          "en el primer informe. Todas han sido resueltas y el entrenamiento prosigue con "
          "normalidad; se documentan por su interés metodológico."),
        sp(0.2),

        h2("3.1 Inyección de cross-attention en un decoder preentrenado"),
        b("GPT-2 es un modelo puramente auto-regresivo: sus bloques no contemplan atención "
          "cruzada hacia una modalidad externa. Para condicionar la generación en la imagen "
          "hubo que <b>insertar capas de cross-attention dentro de cada bloque</b> del decoder "
          "preservando sus pesos preentrenados. La primera aproximación —sustituir dinámicamente "
          "el método de forward de cada bloque— provocó una <b>recursión infinita</b> al "
          "encadenarse las llamadas. Se rediseñó con un patrón más limpio en el que cada bloque "
          "recibe las características visuales a través de un atributo propio, eliminando la "
          "recursión y manteniendo el flujo de generación intacto."),

        h2("3.2 Expansión de batch en la búsqueda por haces (beam search)"),
        b("Durante la generación con beam search, la librería expande internamente el batch "
          "(por ejemplo, multiplicándolo por el número de haces). Las características visuales, "
          "sin embargo, conservaban su tamaño original, lo que producía un <b>desajuste de "
          "dimensiones</b> entre texto e imagen. Se resolvió replicando las características "
          "visuales tantas veces como haces, de modo que cada hipótesis de la búsqueda atienda "
          "a la imagen correcta."),

        h2("3.3 Sentido del padding en un modelo decoder-only"),
        b("Los modelos decoder-only requieren <b>relleno por la izquierda</b> durante la "
          "generación, para que las secuencias terminen en un límite derecho limpio desde el "
          "que continuar la predicción. El valor por defecto (relleno por la derecha) degradaba "
          "la calidad del texto generado. Se ajustó explícitamente el sentido del padding en el "
          "tokenizador."),

        h2("3.4 Interrupciones de entrenamiento y reanudación por checkpoint"),
        b("Al ejecutarse sobre una estación de trabajo local, la <b>suspensión del sistema por "
          "gestión de energía</b> interrumpía el proceso en GPU y detenía el entrenamiento. Se "
          "adoptaron dos medidas: (1) deshabilitar la suspensión durante las sesiones de "
          "entrenamiento, y (2) guardar un checkpoint por epoch y <b>reanudar desde el estado "
          "completo</b> —pesos, optimizador, planificador de learning rate y contador de epoch— "
          "en lugar de reentrenar desde cero. Esto convierte una interrupción en una pérdida de, "
          "como máximo, la epoch en curso."),
    ]

    retos = [
        [ph("RETO"), ph("SÍNTOMA"), ph("RESOLUCIÓN")],
        [pc("Cross-attention en GPT-2"),
         pc("Recursión infinita al inyectar la atención"),
         pc("Rediseño basado en atributo por bloque")],
        [pc("Beam search"),
         pc("Desajuste de dimensiones imagen-texto"),
         pc("Replicación de features por haz")],
        [pc("Padding decoder-only"),
         pc("Texto generado degradado"),
         pc("Relleno por la izquierda")],
        [pc("Suspensión del sistema"),
         pc("Proceso GPU detenido a mitad de epoch"),
         pc("Deshabilitar sleep + reanudar por checkpoint")],
    ]
    s += [sp(0.3),
          tabla(retos, [4.2*cm, 6.0*cm, 6.0*cm]),
          sp(0.3),
          Paragraph("Tabla 2. Resumen de retos técnicos y su resolución.", S_CAP)]

    # ══ 4. INTERPRETACIÓN DE MÉTRICAS ═════════════════════════════════════════
    s += [
        PageBreak(),
        h1("4. Interpretación de las métricas durante el entrenamiento"),
        hr(),
        b("Un aspecto no tratado en el primer informe es cómo leer correctamente el progreso "
          "del entrenamiento generativo. La pérdida mostrada por batch en la barra de progreso "
          "aparenta ser <b>casi estática</b>, lo que puede inducir a pensar erróneamente que el "
          "modelo no aprende. En realidad se trata de un valor suavizado y de escala reducida: "
          "la señal relevante es la <b>métrica de validación por epoch</b>."),
        b("Por ello el criterio de seguimiento y de parada temprana es <b>val/rouge_l</b>, "
          "evaluada al final de cada epoch sobre la partición de validación. Esta métrica sí "
          "progresa de forma apreciable a medida que el decoder aprende a redactar informes "
          "coherentes con la imagen. La evaluación cuantitativa completa se realizará sobre el "
          "conjunto de test reservado una vez alcanzada la convergencia."),
        sp(0.3),
        h2("Lección metodológica"),
        b("La distinción entre <i>pérdida de entrenamiento por batch</i> y <i>métrica de "
          "validación por epoch</i> es esencial para no tomar decisiones prematuras (detener o "
          "reconfigurar el entrenamiento) basadas en una señal ruidosa. El seguimiento debe "
          "anclarse siempre en la métrica que mide el objetivo real —la calidad del texto "
          "generado— y no en la pérdida intermedia de optimización."),
    ]

    # ══ 5. PRÓXIMOS PASOS ═════════════════════════════════════════════════════
    s += [
        sp(0.5),
        h1("5. Próximos pasos"),
        hr(),
        b("La línea de trabajo se mantiene sin cambios respecto al plan del primer informe. "
          "El foco inmediato es <b>continuar el entrenamiento de Stage 2 hasta la convergencia</b> "
          "y, a continuación, avanzar hacia la evaluación y el análisis de explicabilidad."),
        bul("<b>Continuar Stage 2</b> hasta convergencia de val/rouge_l, reanudando desde "
            "checkpoint tras cualquier interrupción."),
        bul("<b>Evaluación en test</b>: BLEU-1/4, ROUGE-L y METEOR sobre la partición reservada, "
            "más una muestra de informes generados para análisis cualitativo."),
        bul("<b>Fase XAI</b>: implementación de los heatmaps por frase y comparación entre "
            "Cross-Attention Rollout e Integrated Gradients, tal como se detalló en el primer "
            "informe."),
        sp(0.4),
        b("<b>Conclusión.</b> El proyecto avanza conforme a lo planificado. El entrenamiento de "
          "los modelos está en marcha, los retos técnicos surgidos en la ejecución se han "
          "resuelto sin alterar el diseño, y se continúa hacia la evaluación y el análisis de "
          "explicabilidad que constituyen la aportación central del TFM."),
    ]

    return s


# ─── BUILD ────────────────────────────────────────────────────────────────────
def main():
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH), pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.8*cm,  bottomMargin=1.4*cm,
        title="Segundo Informe de Seguimiento TFM — Generación de Informes Radiológicos",
        author="Daniel Gómez Arranz",
    )
    doc.build(story(), onFirstPage=on_first_page, onLaterPages=on_page)
    print(f"PDF generado: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
