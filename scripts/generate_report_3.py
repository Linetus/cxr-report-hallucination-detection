"""
Genera el TERCER informe de seguimiento del TFM en PDF (mismo estilo que los anteriores).
Foco: cierre de Stage 2, evolución del enfoque hacia el análisis de fiabilidad/explicabilidad,
elevación a un MLLM del estado del arte (LLaVA-Rad), anclaje objetivo con bounding boxes NIH,
y el hallazgo central sobre detección de alucinaciones.
Uso: python scripts/generate_report_3.py
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak,
    Image as RLImage,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

AZUL_OSC  = colors.HexColor("#1a2e4a"); AZUL_MED = colors.HexColor("#2d5986")
AZUL_CLA  = colors.HexColor("#4a90d9"); GRIS_CLA = colors.HexColor("#f4f6f9")
GRIS_TXT  = colors.HexColor("#666666"); GRID_COL = colors.HexColor("#d0d8e4")
VERDE     = colors.HexColor("#27ae60"); NARANJA  = colors.HexColor("#e67e22")

FECHA_STR = "Agosto 2026"; FECHA_LARGA = "7 de agosto de 2026"
OUTPUT_PATH = Path("outputs/tfm_seguimiento_fase3.pdf")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
FIG = Path("outputs/figs")

def make(name, **kw): return ParagraphStyle(name, **kw)
S_TITLE = make("Title3", fontSize=20, fontName="Helvetica-Bold", textColor=AZUL_OSC, alignment=TA_CENTER, spaceAfter=6, leading=26)
S_SUB   = make("Sub", fontSize=12, fontName="Helvetica", textColor=AZUL_MED, alignment=TA_CENTER, spaceAfter=3)
S_META  = make("Meta", fontSize=9, fontName="Helvetica", textColor=GRIS_TXT, alignment=TA_CENTER, spaceAfter=2)
S_H1    = make("H1", fontSize=13, fontName="Helvetica-Bold", textColor=AZUL_OSC, spaceBefore=16, spaceAfter=5)
S_H2    = make("H2", fontSize=10, fontName="Helvetica-Bold", textColor=AZUL_MED, spaceBefore=9, spaceAfter=3)
S_BODY  = make("Body", fontSize=9.5, fontName="Helvetica", textColor=colors.black, leading=14, alignment=TA_JUSTIFY, spaceAfter=5)
S_BULL  = make("Bull", fontSize=9.5, fontName="Helvetica", textColor=colors.black, leading=14, leftIndent=14, spaceAfter=3)
S_CAP   = make("Cap", fontSize=8, fontName="Helvetica-Oblique", textColor=GRIS_TXT, alignment=TA_CENTER, spaceAfter=4)
S_CELL  = make("Cell", fontSize=8.5, fontName="Helvetica", textColor=colors.black, leading=12)
S_CELL_B= make("CellB", fontSize=8.5, fontName="Helvetica-Bold", textColor=colors.black, leading=12)
S_CELL_H= make("CellH", fontSize=8.5, fontName="Helvetica-Bold", textColor=colors.white, leading=12, alignment=TA_CENTER)
S_CELL_C= make("CellC", fontSize=8.5, fontName="Helvetica", textColor=colors.black, leading=12, alignment=TA_CENTER)

def hr(color=AZUL_CLA, t=0.7): return HRFlowable(width="100%", thickness=t, color=color, spaceAfter=7)
def sp(n=1): return Spacer(1, n*0.32*cm)
def b(txt): return Paragraph(txt, S_BODY)
def h1(txt): return Paragraph(txt, S_H1)
def h2(txt): return Paragraph(txt, S_H2)
def bul(txt): return Paragraph(f"• {txt}", S_BULL)
def pc(txt, bold=False): return Paragraph(txt, S_CELL_B if bold else S_CELL)
def ph(txt): return Paragraph(txt, S_CELL_H)
def pcc(txt): return Paragraph(txt, S_CELL_C)

BASE_TABLE = TableStyle([
    ("BACKGROUND",(0,0),(-1,0),AZUL_MED), ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,GRIS_CLA]),
    ("GRID",(0,0),(-1,-1),0.4,GRID_COL), ("VALIGN",(0,0),(-1,-1),"TOP"),
    ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),7), ("RIGHTPADDING",(0,0),(-1,-1),7)])

def on_first_page(canvas, doc):
    canvas.saveState(); canvas.setFillColor(AZUL_OSC); canvas.rect(0, A4[1]-3.2*cm, A4[0], 3.2*cm, fill=1, stroke=0)
    canvas.setFillColor(AZUL_CLA); canvas.rect(0, A4[1]-3.4*cm, A4[0], 0.22*cm, fill=1, stroke=0)
    _footer(canvas, doc); canvas.restoreState()
def on_page(canvas, doc):
    canvas.saveState(); canvas.setFillColor(AZUL_OSC); canvas.rect(0, A4[1]-1.0*cm, A4[0], 1.0*cm, fill=1, stroke=0)
    canvas.setFillColor(colors.white); canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(1.8*cm, A4[1]-0.68*cm, "TFM — Generación de Informes Radiológicos con XAI")
    canvas.setFont("Helvetica", 8); canvas.drawRightString(A4[0]-1.8*cm, A4[1]-0.68*cm, f"Daniel Gómez Arranz · {FECHA_STR}")
    _footer(canvas, doc); canvas.restoreState()
def _footer(canvas, doc):
    canvas.setFillColor(GRIS_CLA); canvas.rect(0, 0, A4[0], 0.8*cm, fill=1, stroke=0)
    canvas.setFillColor(GRIS_TXT); canvas.setFont("Helvetica", 7.5)
    canvas.drawCentredString(A4[0]/2, 0.27*cm, f"Página {doc.page}")

def tabla(data, col_widths, extra=None):
    ts = list(BASE_TABLE._cmds); ts += (extra or [])
    return Table(data, colWidths=col_widths, style=TableStyle(ts), repeatRows=1)
def _badge(label, color):
    return Table([[Paragraph(label, make("_bl", fontSize=7.5, fontName="Helvetica-Bold", textColor=colors.white, alignment=TA_CENTER, leading=10))]],
        colWidths=[2.6*cm], style=TableStyle([("BACKGROUND",(0,0),(-1,-1),color),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2)]))

def fig_row(items, wcm=4.7):
    imgs = [RLImage(str(FIG/it[0]), width=wcm*cm, height=wcm*cm) for it in items]
    caps = [Paragraph(it[1], make("_fc", fontSize=7.5, fontName="Helvetica",
            textColor=GRIS_TXT, alignment=TA_CENTER, leading=9)) for it in items]
    t = Table([imgs, caps], colWidths=[wcm*cm]*len(items))
    t.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    return t

def story():
    s = []
    # ── PORTADA ──
    s += [
        Spacer(1, 3.5*cm),
        Paragraph("TERCER INFORME DE SEGUIMIENTO — TFM", make("_tag", fontSize=9, fontName="Helvetica-Bold", textColor=AZUL_CLA, alignment=TA_CENTER, spaceAfter=10)),
        Paragraph("Generación de Informes Radiológicos<br/>mediante Arquitecturas Encoder-Decoder<br/>Multimodales con Explicabilidad Visual", S_TITLE),
        sp(0.4), hr(AZUL_CLA, 1.4), sp(0.4),
        Paragraph("Máster Universitario en Inteligencia Artificial", S_SUB), sp(0.2),
        Paragraph("Autor: Daniel Gómez Arranz · dagoar@outlook.es", S_META),
        Paragraph("Convocatoria: Septiembre 2026", S_META),
        Paragraph(f"Fecha: {FECHA_LARGA}", S_META), sp(1.6),
        Paragraph(
            "Este tercer informe da continuidad a los anteriores. Documenta el <b>cierre del "
            "entrenamiento (Stage 2)</b> y una <b>evolución justificada del enfoque</b>: al aplicar "
            "el marco de explicabilidad ya previsto, se constató que el generador propio, limitado "
            "por el tamaño del conjunto de datos, produce explicaciones débiles. Para que el análisis "
            "—que constituye la aportación central del TFM— sea significativo, se ha <b>elevado el "
            "estudio a un modelo multimodal del estado del arte</b> y se ha validado de forma "
            "objetiva contra anotaciones reales de localización.",
            make("_lead", fontSize=9.5, fontName="Helvetica", textColor=GRIS_TXT, alignment=TA_CENTER, leading=14, leftIndent=1.2*cm, rightIndent=1.2*cm)),
        sp(1.0),
    ]
    avance = [
        [ph("HITO"), ph("SITUACIÓN EN ESTE SEGUIMIENTO")],
        [pc("Stage 2 — generación"), _badge("COMPLETADO", VERDE)],
        [pc("Framework XAI (fidelidad + anclaje)"), _badge("COMPLETADO", VERDE)],
        [pc("Elevación a LLaVA-Rad + anclaje NIH"), _badge("COMPLETADO", VERDE)],
        [pc("Detección de alucinaciones (multi-señal)"), _badge("EN CURSO", NARANJA)],
        [pc("Redacción de la memoria"), _badge("EN CURSO", NARANJA)],
    ]
    s.append(tabla(avance, [7.5*cm, 8.5*cm])); s.append(PageBreak())

    # ── 1 ──
    s += [h1("1. Objeto de este tercer seguimiento"), hr(),
        b("Los informes anteriores dejaron definidos la arquitectura encoder-decoder, el pipeline de "
          "datos sobre Open-I, el plan de explicabilidad y el avance del entrenamiento. Desde entonces "
          "el <b>ajuste generativo (Stage 2) ha concluido</b> y se ha ejecutado el análisis de "
          "explicabilidad previsto."),
        b("Este documento persigue dos objetivos: (1) reportar el cierre del entrenamiento y sus "
          "conclusiones, y (2) motivar y describir una <b>evolución del enfoque de la aportación</b>. "
          "Dicha evolución no altera los objetivos del TFM —la fidelidad y el anclaje de las "
          "explicaciones—, sino que refuerza su relevancia: traslada el análisis a un modelo que "
          "funciona a escala clínica y lo valida con una referencia objetiva.")]

    # ── 2 ──
    s += [sp(0.3), h1("2. Cierre del entrenamiento y hallazgo determinante"), hr(),
        b("El entrenamiento en dos etapas se ha completado. El modelo propio genera texto radiológico "
          "coherente; sin embargo, un <b>test de sensibilidad a la imagen</b> reveló que, por el tamaño "
          "limitado de Open-I (~4.000 estudios), tiende a producir un <b>informe-plantilla</b> similar "
          "para imágenes distintas, memorizando la mayoría de casos normales del dataset."),
        b("Este hallazgo tiene una implicación metodológica publicable: la métrica <b>ROUGE puede "
          "engañar</b> en este dominio. Una variante mejor anclada a la imagen obtuvo una ROUGE "
          "<i>inferior</i> a la del modelo ciego a la imagen, porque premia la reproducción de la "
          "plantilla. Se incorpora, por ello, un eje de evaluación de <b>uso real de la imagen</b> "
          "junto a las métricas de texto. La conclusión operativa es doble: (a) la <b>destilación de "
          "conocimiento</b> desde modelos preentrenados libres se consolida como columna metodológica "
          "ante la restricción de datos; y (b) el análisis de explicabilidad debe apoyarse en un modelo "
          "suficientemente funcional para ser significativo.")]

    # ── 3 ──
    s += [sp(0.3), h1("3. Evolución del enfoque: del generador al análisis de fiabilidad"), hr(),
        b("La generación de informes es una línea saturada en la literatura y un modelo propio de "
          "escala reducida queda por debajo del estado del arte. La <b>contribución diferencial</b> del "
          "TFM no es, por tanto, el generador, sino el <b>análisis de explicabilidad y fiabilidad</b>: "
          "determinar si las explicaciones visuales son <i>fieles</i> al razonamiento del modelo, si "
          "están <i>ancladas</i> a la anatomía correcta, y si permiten <b>detectar alucinaciones</b>."),
        b("Para que ese análisis sea relevante se ha elevado a <b>LLaVA-Rad</b> (modelo multimodal de "
          "radiografía de tórax, decoder de 7.000 millones de parámetros con encoder visual biomédico, "
          "entrenado sobre MIMIC-CXR). Se ejecuta en el hardware local mediante cuantización a 8 bits. "
          "El marco de explicabilidad diseñado para el modelo propio se ha <b>adaptado</b> a su "
          "arquitectura y se ha aplicado tanto a este modelo como al propio.")]

    # ── 4 ──
    s += [PageBreak(), h1("4. Análisis de explicabilidad: metodología y resultados"), hr(),
        h2("4.1 Cómo se obtienen los mapas de calor"),
        b("El mapa de calor de una afirmación se extrae de la <b>propia atención del modelo</b> durante "
          "la generación, sin coste ni entrenamiento adicional. La imagen se codifica en una rejilla de "
          "<b>37×37 = 1.369 parches visuales</b>. Tras generar el informe, se realiza una pasada con la "
          "secuencia completa registrando los pesos de atención de todas las capas y cabezas. Para cada "
          "token generado se toma su atención hacia esos 1.369 parches, se <b>promedia sobre capas y "
          "cabezas</b> y se reordena en una matriz 37×37. Al agregar los tokens que componen una frase o "
          "un hallazgo concreto se obtiene el mapa que señala qué región de la radiografía sustentó esa "
          "afirmación."),
        b("Para visualizarlo se reescala el mapa a la resolución de la imagen y se superpone (tonos "
          "cálidos = mayor atención). Se descarta un artefacto conocido —el <i>attention sink</i>, una "
          "esquina que acumula atención de forma no informativa— antes de calcular cualquier métrica. "
          "En el modelo propio la explicación proviene de la cross-attention inyectada; en LLaVA-Rad, de "
          "la atención del texto hacia los tokens visuales antepuestos a la secuencia."),
        h2("4.2 Métricas empleadas"),
        b("Se definen a continuación todas las métricas utilizadas en el análisis.")]
    gloss = [
        [ph("Métrica"), ph("Qué mide")],
        [pc("Energy-in-bbox"), pc("Fracción de la masa de atención que cae dentro de la caja real (0–1).")],
        [pc("Área de la caja"), pc("Proporción de la imagen que ocupa la caja; es la línea base de azar.")],
        [pc("Lift"), pc("Energy-in-bbox ÷ área. >1 = mejor que el azar; cuánto se concentra en la lesión.")],
        [pc("Pointing-game"), pc("Vale 1 si el pico de atención cae dentro de la caja; mide acierto de localización.")],
        [pc("AUC"), pc("Capacidad de una señal para separar dos clases (0,5 = azar; 1 = separación perfecta).")],
        [pc("p (permutación)"), pc("Probabilidad de obtener ese AUC por azar barajando las etiquetas; <0,05 = significativo.")],
        [pc("min_prob"), pc("Probabilidad del token menos seguro del informe: la incertidumbre del modelo al generar.")],
        [pc("Entropía predictiva"), pc("Dispersión de la distribución de salida por token; mayor valor = más duda.")],
        [pc("Forma del heatmap"), pc("Gini, entropía y dispersión del mapa: concentración frente a difusión de la atención.")],
    ]
    s += [sp(0.2), tabla(gloss, [4.6*cm, 11.8*cm]),
          sp(0.2), Paragraph("Tabla 1. Glosario de métricas del análisis.", S_CAP),
          sp(0.3), h2("4.3 Anclaje clínico objetivo (bounding boxes NIH)"),
        b("Se ha validado el anclaje de la atención contra <b>anotaciones reales de localización</b> "
          "(NIH ChestX-ray8, 8 patologías con caja delimitadora). La métrica <i>lift</i> mide cuánto se "
          "concentra la atención dentro de la lesión real frente al azar. El resultado confirma que la "
          "atención <b>está clínicamente anclada</b> (lift global 3,03; el pico de atención cae en la "
          "lesión 3,7 veces más que el azar), variando con sentido clínico: los hallazgos focales "
          "anclan con fuerza y la cardiomegalia —difusa y central— lo hace de forma más débil.")]
    nih = [
        [ph("Patología"), ph("Detección"), ph("Lift"), ph("Pointing-game")],
        [pc("Nódulo"), pcc("0,22"), pcc("7,91"), pcc("0,47")],
        [pc("Masa"), pcc("0,18"), pcc("5,10"), pcc("0,87")],
        [pc("Atelectasia"), pcc("0,29"), pcc("3,45"), pcc("0,30")],
        [pc("Derrame pleural"), pcc("0,58"), pcc("3,23"), pcc("0,36")],
        [pc("Neumotórax"), pcc("0,74"), pcc("2,81"), pcc("0,37")],
        [pc("Neumonía"), pcc("0,29"), pcc("2,67"), pcc("0,23")],
        [pc("Infiltrado"), pcc("0,19"), pcc("2,62"), pcc("0,48")],
        [pc("Cardiomegalia"), pcc("0,58"), pcc("1,63"), pcc("0,39")],
        [pc("Global", bold=True), pcc("0,39"), pcc("3,03"), pcc("0,38")],
    ]
    s += [sp(0.2), tabla(nih, [5.4*cm, 3.2*cm, 3.2*cm, 4.2*cm]),
          sp(0.2), Paragraph("Tabla 2. Anclaje objetivo de LLaVA-Rad sobre NIH (n = 388 hallazgos detectados).", S_CAP),
          sp(0.35),
          fig_row([("nih_good_1_nodule.png","Nódulo · lift 22"),
                   ("nih_good_2_pneumothorax.png","Neumotórax · lift 17"),
                   ("nih_good_3_mass.png","Masa · lift 16")]),
          sp(0.12),
          fig_row([("nih_bad_1_pneumothorax.png","Neumotórax · atención dispersa"),
                   ("nih_bad_2_pneumothorax.png","Neumotórax · atención fuera de la lesión")]),
          sp(0.2),
          Paragraph("Figura 1. Mapa de atención (tonos cálidos = mayor atención) superpuesto a la "
                    "radiografía; en verde, la caja real de la lesión. En los cinco casos el modelo "
                    "<b>detectó correctamente</b> la patología: la diferencia no está en el acierto sino "
                    "en el <b>anclaje</b>. Arriba, la atención cae sobre la lesión (bien anclada); abajo, "
                    "se dispersa fuera de ella (mal anclada). Un diagnóstico correcto puede, por tanto, no "
                    "ir acompañado de una atención anclada.", S_CAP)]

    s += [sp(0.3), h2("4.4 El anclaje acompaña al acierto, pero no es una señal de inferencia"),
        b("Podría pensarse que el grado de anclaje predice el acierto; sin embargo, un control cuidadoso "
          "lo desmiente. Al comparar con un <b>mapa de calor homogéneo</b> (el mismo tipo para aciertos y "
          "omisiones), la relación entre anclaje (<i>lift</i>) y acierto <b>desaparece</b> (AUC 0,52): la "
          "aparente asociación era un artefacto de contrastar mapas por-hallazgo —más concentrados— con "
          "mapas globales —más difusos—. El anclaje, por tanto, <b>no predice la correctitud</b>, en "
          "coherencia con que la fiabilidad no reside en la atención; y además solo puede medirse con la "
          "caja de referencia, no disponible en inferencia."),
        b("El estudio revela dos <b>disociaciones</b> entre lo que el modelo mira y lo que dice: casos en "
          "que <b>afirma un hallazgo sin mirarlo</b> (alucinación) y casos en que <b>lo mira pero no lo "
          "reporta</b> (omisión de lesiones focales pequeñas, como masas o nódulos). Esto sitúa el fallo en "
          "la decisión de lenguaje, no en la visión. De hecho, ninguna propiedad morfológica de la imagen "
          "—contraste o conspicuidad de la lesión— predice el acierto; el único factor con valor "
          "predictivo es el <b>tamaño</b> del hallazgo."),
    ]
    s += [sp(0.3), h2("4.5 Detección de alucinaciones: la forma no basta, la incertidumbre sí"),
        b("A diferencia del anclaje —que mide <i>dónde</i> mira el modelo—, aquí se estudia si la "
          "explicación permite saber <i>cuándo fiarse</i>. Sobre Open-I, cada informe generado se compara "
          "con el informe real completo para etiquetarlo como correcto o alucinado. "),
        b("La hipótesis inicial —que un mapa de atención más disperso delata una alucinación— se ha "
          "<b>refutado con rigor</b>: ninguna característica geométrica del heatmap (concentración, "
          "dispersión, agrupamiento) separa de forma robusta los hallazgos correctos de los alucinados. "
          "Las señales aparentemente fuertes resultaron artefactos, detectados al validar contra una "
          "referencia objetiva. Por el contrario, la <b>incertidumbre del propio modelo al generar</b> "
          "(la probabilidad del token menos seguro del informe) sí discrimina de forma significativa. "
          "El etiquetado de correcto/alucinado se realiza comparando la <b>narrativa completa</b> "
          "generada con el informe real, mediante un juez basado en un modelo de lenguaje (metodología "
          "tipo G-Eval).")]
    sig = [
        [ph("Señal / objetivo (agnóstico, en inferencia)"), ph("AUC")],
        [pc("Forma del heatmap (morfología)"), pcc("0,54  (no signif.)")],
        [pc("Fórmula de confianza — detectar ALUCINACIONES"), pcc("0,87")],
        [pc("Fórmula de confianza — detectar MISSES (omisiones)"), pcc("0,61  (≈ azar)")],
    ]
    s += [sp(0.2), tabla(sig, [11.4*cm, 4.6*cm]),
          sp(0.2), Paragraph("Tabla 3. Capacidad de las señales para detectar errores (n = 800 informes "
                    "etiquetados; 42 alucinaciones, 32 omisiones; validación out-of-fold y permutación). La "
                    "fórmula combina incertidumbre y auto-consistencia, controlando la longitud del informe.", S_CAP),
          sp(0.2),
          b("Estas señales son <b>agnósticas al modelo</b> (probabilidades de salida y consistencia entre "
            "muestras) y están disponibles <b>en inferencia</b>. Un hallazgo determinante es que la fórmula "
            "distingue dos modos de error: detecta muy bien las <b>alucinaciones</b> —el modelo <i>titubea</i> "
            "al fabricar un hallazgo (AUC 0,87)— pero casi no detecta las <b>omisiones</b>, porque estas se "
            "expresan como un informe ``normal'' emitido con plena confianza, indistinguible de un normal "
            "correcto. Como <b>priorizador de revisión</b>, ordenar los informes por esta confianza permitiría "
            "capturar el 88&#37; de las alucinaciones revisando una fracción de los estudios. Combinada y "
            "<b>calibrada</b>, la fórmula expresa la fiabilidad de cada informe como una probabilidad con su "
            "<b>intervalo de confianza</b> —más estrecho para las alucinaciones que para las omisiones—, con "
            "el radiólogo como decisor.")]

    # ── 5 retos ──
    s += [PageBreak(), h1("5. Retos técnicos surgidos en esta fase"), hr(),
        b("Como en la fase anterior, el paso a la ejecución hizo aflorar dificultades que se documentan "
          "por su interés metodológico. Todas han sido resueltas."),
        h2("5.1 Ejecución de un modelo de 7B en hardware local"),
        b("Cargar un modelo multimodal de 7.000 millones de parámetros en una GPU de 12 GB no era "
          "directo: las vías habituales de cuantización fallaban. Se resolvió <b>fusionando los pesos "
          "en CPU</b> y cargando después el modelo fusionado en <b>8 bits</b>, lo que lo hace caber en "
          "memoria manteniendo la capacidad de generar y de exponer sus mapas de atención."),
        h2("5.2 Adaptación del marco XAI a otra arquitectura"),
        b("A diferencia del modelo propio (cross-attention inyectada), este MLLM antepone los tokens "
          "visuales a la secuencia de texto. El extractor de explicabilidad se <b>reescribió</b> para "
          "leer la atención del texto hacia esos tokens visuales y reconstruir el mapa sobre la imagen."),
        h2("5.3 Detección de artefactos metodológicos"),
        b("Varias señales parecían separar correctos de alucinados con métricas altas, pero al "
          "<b>validarlas contra una referencia objetiva</b> se revelaron como artefactos (por ejemplo, "
          "efectos del máximo de atención o de la media de un estadístico sensible a valores extremos). "
          "La lección —validar siempre frente a una etiqueta objetiva o continua— es en sí una "
          "aportación del trabajo.")]
    retos = [
        [ph("RETO"), ph("SÍNTOMA"), ph("RESOLUCIÓN")],
        [pc("Modelo 7B en GPU 12 GB"), pc("Cuantización directa inviable"), pc("Fusión de pesos en CPU + carga 8 bits")],
        [pc("XAI en nueva arquitectura"), pc("Atención con tokens visuales prefijo"), pc("Extractor reescrito texto→visual")],
        [pc("Señales espejismo"), pc("AUC alto por artefacto"), pc("Validación contra referencia objetiva")],
        [pc("Librerías numéricas (Windows)"), pc("Fallo de BLAS en correlaciones"), pc("Implementación manual robusta")],
    ]
    s += [sp(0.3), tabla(retos, [4.2*cm, 6.0*cm, 6.0*cm]),
          sp(0.2), Paragraph("Tabla 4. Retos técnicos de esta fase y su resolución.", S_CAP)]

    # ── 6 próximos pasos ──
    s += [sp(0.4), h1("6. Trabajo en curso y próximos pasos"), hr(),
        b("El foco actual es consolidar la detección de fiabilidad y redactar la memoria dentro del "
          "plazo previsto."),
        bul("<b>Extracción consolidada multi-señal</b> (en ejecución) sobre Open-I y sobre el conjunto "
            "con anotaciones NIH: en una sola pasada se obtienen incertidumbre, dependencia de la "
            "imagen, fidelidad por perturbación, ablación y auto-consistencia."),
        bul("<b>Fórmula de confianza calibrada</b>: combinar las señales significativas para estimar la "
            "probabilidad de alucinación con su intervalo de confianza."),
        bul("<b>Validación en un segundo modelo</b> para comprobar la generalización de las conclusiones."),
        bul("<b>Redacción de la memoria</b> siguiendo la plantilla oficial del Máster."),
        sp(0.4),
        b("<b>Conclusión.</b> El entrenamiento se ha completado y el análisis de explicabilidad —núcleo "
          "del TFM— ha producido resultados sólidos: la atención de estos modelos está clínicamente "
          "anclada, pero la fiabilidad no se lee en la forma de la explicación sino en la incertidumbre "
          "del modelo. La evolución del enfoque refuerza la aportación sin desviarse de los objetivos, y "
          "el trabajo avanza conforme al calendario hacia la fórmula de confianza y la memoria."),
        b("El producto que se persigue combina, para cada estudio, el <b>informe generado</b>, el "
          "<b>mapa de calor</b> (dónde ha mirado el modelo) y una <b>confianza calibrada con intervalo</b> "
          "(cuánto fiarse), con el radiólogo como decisor. El mapa orienta la mirada; la confianza, no su "
          "forma, indica la fiabilidad. Su beneficio clínico deberá confirmarse con un estudio de lectura.")]
    return s

def main():
    doc = SimpleDocTemplate(str(OUTPUT_PATH), pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm, topMargin=1.8*cm, bottomMargin=1.4*cm,
        title="Tercer Informe de Seguimiento TFM — Generación de Informes Radiológicos",
        author="Daniel Gómez Arranz")
    doc.build(story(), onFirstPage=on_first_page, onLaterPages=on_page)
    print(f"PDF generado: {OUTPUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
