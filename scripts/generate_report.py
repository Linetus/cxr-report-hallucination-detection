"""
Genera el informe de seguimiento del TFM en PDF.
Uso: python scripts/generate_report.py
"""
from pathlib import Path
from datetime import date

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

# ─── COLORES ─────────────────────────────────────────────────────────────────
AZUL_OSC  = colors.HexColor("#1a2e4a")
AZUL_MED  = colors.HexColor("#2d5986")
AZUL_CLA  = colors.HexColor("#4a90d9")
GRIS_CLA  = colors.HexColor("#f4f6f9")
GRIS_TXT  = colors.HexColor("#666666")
GRID_COL  = colors.HexColor("#d0d8e4")
VERDE     = colors.HexColor("#27ae60")
NARANJA   = colors.HexColor("#e67e22")

OUTPUT_PATH = Path("outputs/tfm_seguimiento_fase1.pdf")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

W = A4[0] - 3.6 * cm   # ancho útil (márgenes 1.8 cm c/lado)

# ─── ESTILOS ─────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

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
# Estilos para interior de celdas (sin márgenes propios)
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
    """Paragraph para celda de tabla."""
    return Paragraph(txt, S_CELL_B if bold else S_CELL)

def ph(txt):
    """Paragraph cabecera de tabla (blanco, centrado)."""
    return Paragraph(txt, S_CELL_H)

def pcc(txt):
    """Paragraph celda centrada."""
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
                           "Daniel Gómez Arranz · Mayo 2026")
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

# ─── CONTENIDO ───────────────────────────────────────────────────────────────
def story():
    s = []

    # ══ PORTADA ══════════════════════════════════════════════════════════════
    s += [
        Spacer(1, 3.5*cm),
        Paragraph("INFORME DE SEGUIMIENTO — TFM", make(
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
        Paragraph(f"Fecha: {date.today().strftime('%d de %B de %Y')}", S_META),
        sp(1.8),
    ]

    # Tabla estado portada
    estado = [
        [ph("FASE"), ph("DESCRIPCIÓN"), ph("ESTADO")],
        [pc("Fase 1"), pc("Pipeline de datos + EDA Open-I"),                    _badge("COMPLETADA", VERDE)],
        [pc("Fase 2"), pc("Arquitectura encoder-decoder + módulo XAI"),          _badge("COMPLETADA", VERDE)],
        [pc("Fase 3"), pc("Entrenamiento Stage 1 (contrastivo) + Stage 2 (gen.)"),_badge("EN CURSO",  NARANJA)],
        [pc("Fase 4"), pc("Análisis XAI — Cross-Attn Rollout vs Integrated Gradients"), _badge("PENDIENTE", AZUL_MED)],
        [pc("Fase 5"), pc("Evaluación NLP (BLEU / ROUGE / METEOR) + análisis cualitativo"), _badge("PENDIENTE", AZUL_MED)],
        [pc("Fase 6"), pc("Demo interactiva + Redacción memoria TFM"),           _badge("PENDIENTE", AZUL_MED)],
    ]
    s.append(tabla(estado, [2.5*cm, 10.5*cm, 3.0*cm]))
    s.append(PageBreak())

    # ══ 1. RESUMEN EJECUTIVO ══════════════════════════════════════════════════
    s += [
        h1("1. Resumen Ejecutivo"),
        hr(),
        b("Este TFM desarrolla un sistema de <b>generación automática de informes radiológicos</b> "
          "a partir de imágenes de radiografía de tórax, con énfasis en la <b>trazabilidad visual</b>: "
          "cada afirmación del informe generado queda anclada a la región de la imagen que la motivó."),
        b("La motivación clínica es directa. Los sistemas actuales de IA para diagnóstico producen "
          "predicciones sin razonamiento. Un radiólogo no dice <i>'consolidación: 87%'</i>; describe "
          "lo que ve, dónde lo ve y por qué concluye lo que concluye. Este proyecto replica ese proceso "
          "de forma trazable y verificable por el clínico."),
        sp(0.3),
        h2("Contribución Principal"),
        b("La arquitectura implementa cross-attention entre el decoder de lenguaje y los patches visuales "
          "del encoder. Este mecanismo, inherente al proceso de generación, permite extraer sin coste "
          "adicional qué región de la imagen activó cada token generado. La contribución diferencial del "
          "TFM es la <b>comparación sistemática de dos métodos XAI</b> aplicados a un modelo generativo:"),
        bul("<b>Cross-Attention Rollout</b>: acumula pesos de atención capa a capa durante la generación. "
            "Instantáneo, sin coste adicional de inferencia."),
        bul("<b>Integrated Gradients (Captum)</b>: mide la contribución causal de cada patch integrando "
            "gradientes a lo largo de un camino imagen-negra → imagen-real. Más riguroso matemáticamente."),
        b("Ambos métodos se evalúan sobre las mismas muestras y se compara su concordancia mediante "
          "correlación de Spearman, analizando cuál produce heatmaps más alineados con el criterio clínico."),
        sp(0.3),
        h2("Contexto y Motivación"),
        b("La interpretabilidad de los modelos de IA en entornos clínicos no es una característica "
          "opcional: es un requisito regulatorio y ético. Este proyecto aborda esa necesidad en el "
          "dominio radiológico, desarrollando un sistema cuyas decisiones son auditables por el "
          "clínico frase a frase. El perfil investigador del autor se orienta hacia <b>IA aplicada "
          "en entornos de alta responsabilidad</b> donde la trazabilidad de las predicciones "
          "es imprescindible."),
    ]

    # ══ 2. ARQUITECTURA ══════════════════════════════════════════════════════
    s += [
        sp(0.5),
        h1("2. Arquitectura del Sistema"),
        hr(),
        b("La arquitectura combina un encoder visual basado en Vision Transformer con un decoder de "
          "lenguaje GPT-2 ampliado con capas de cross-attention. El diseño sigue el patrón "
          "encoder-decoder de Vaswani et al. (2017), adaptado para generación condicional imagen→texto."),
        sp(0.3),
    ]

    arch = [
        [ph("COMPONENTE"), ph("IMPLEMENTACIÓN"), ph("PARÁMS."), ph("ROL EN EL SISTEMA")],
        [pc("Visual Encoder"),   pc("ViT-Base/16 (timm, ImageNet)"),         pcc("86 M"),  pc("14×14 patches → 196 embeddings de 768D")],
        [pc("Cross-Attention"),  pc("8 cabezas × 12 capas (custom)"),         pcc("~5 M"),  pc("Cada token atiende los 196 patches visuales")],
        [pc("Language Decoder"), pc("GPT-2 + capas cross-attn inyectadas"),   pcc("117 M"), pc("Generación token a token condicionada en imagen")],
        [pc("Módulo XAI"),       pc("CrossAttnExtractor + Captum IG"),         pcc("—"),     pc("Heatmaps 14×14 por frase generada")],
        [pc("TOTAL", bold=True), pc("—"),                                     pcc("~208 M"),pc("—")],
    ]
    s.append(tabla(arch, [3.5*cm, 5.0*cm, 2.2*cm, 5.9*cm],
        extra_styles=[
            ("BACKGROUND", (0,-1),(-1,-1), colors.HexColor("#e8f0fb")),
            ("FONTNAME",   (0,-1),(-1,-1), "Helvetica-Bold"),
        ]))
    s += [sp(0.3), Paragraph("Tabla 1. Componentes de la arquitectura.", S_CAP), sp(0.3)]

    s += [
        h2("Flujo de Inferencia"),
        Paragraph(
            "Imagen RX (224×224)  →  ViT-Base/16  →  196 patches (768D)  →  "
            "GPT-2 + Cross-Attention  →  Informe texto  →  XAI: heatmap por frase",
            make("_code", fontSize=8, fontName="Courier", textColor=AZUL_OSC,
                 backColor=GRIS_CLA, leftIndent=10, rightIndent=10,
                 leading=12, spaceAfter=5, spaceBefore=3)),
        b("El decoder genera cada token mirando simultáneamente el texto previo (self-attention) "
          "y los 196 patches de la imagen (cross-attention). Los pesos de esta segunda atención "
          "son la materia prima del módulo XAI: agrupados por frase, producen un mapa 14×14 "
          "que señala qué zona de la RX activó cada hallazgo clínico."),
        sp(0.3),
        h2("Comparación con Enfoques Alternativos"),
    ]

    comp = [
        [ph("ENFOQUE"),          ph("Explicabilidad"), ph("Generación texto"), ph("Coste XAI")],
        [pc("Clasificador CNN"),         pcc("Post-hoc (GradCAM)"),    pcc("No"),          pcc("Moderado")],
        [pc("LLM multimodal (LLaVA)"),   pcc("Opaca"),                 pcc("Sí"),           pcc("Alto")],
        [pc("Este TFM (Cross-Attn)"),    pcc("Inherente"),             pcc("Sí"),           pcc("Nulo")],
    ]
    s.append(tabla(comp, [5.5*cm, 3.5*cm, 3.5*cm, 4.1*cm],
        extra_styles=[("ALIGN",(1,0),(-1,-1),"CENTER")]))
    s += [sp(0.3), Paragraph("Tabla 2. Comparativa de enfoques.", S_CAP)]

    # ══ 3. PIPELINE DE DATOS ══════════════════════════════════════════════════
    s += [
        PageBreak(),
        h1("3. Pipeline de Datos — Fase 1 (Completada)"),
        hr(),
        h2("Dataset: Open-I Indiana University CXR"),
        b("Dataset de acceso libre del National Library of Medicine (NLM). Contiene radiografías "
          "de tórax en PNG con informes clínicos estructurados en XML (secciones FINDINGS e IMPRESSION). "
          "Se utilizó como dataset principal por su disponibilidad sin restricciones de acceso, "
          "a diferencia de MIMIC-CXR (requiere credenciales PhysioNet)."),
        sp(0.3),
    ]

    data = [
        [ph("MÉTRICA"), ph("VALOR"), ph("OBSERVACIÓN")],
        [pc("Total estudios"),        pcc("3.955"),  pc("Cada estudio = 1 paciente / visita")],
        [pc("Total imágenes (PNG)"),  pcc("7.470"),  pc("Vista frontal (PA) y lateral por estudio")],
        [pc("Split train"),           pcc("5.231"),  pc("70% de estudios — ambas vistas incluidas")],
        [pc("Split validación"),      pcc("1.120"),  pc("15% de estudios")],
        [pc("Split test"),            pcc("1.119"),  pc("15% de estudios — reservado para evaluación final")],
        [pc("Criterio de split"),     pcc("UID"),    pc("Por estudio, no por imagen — evita data leakage entre vistas")],
        [pc("Longitud media informe"), pcc("~50 tok."), pc("Tokenización GPT-2, máx. 256 tokens")],
    ]
    s.append(tabla(data, [4.5*cm, 2.5*cm, 9.6*cm]))
    s += [sp(0.3), Paragraph("Tabla 3. Estadísticas del dataset Open-I.", S_CAP), sp(0.3)]

    s += [
        h2("Preprocesamiento Implementado"),
        bul("<b>CXRPreprocessor</b> (Albumentations): normalización de histograma (CLAHE), "
            "resize a 224×224, conversión a 3 canales. En modo train: RandomBrightnessContrast, "
            "HorizontalFlip, ShiftScaleRotate. En eval: solo normalización."),
        bul("<b>ReportPreprocessor</b>: limpieza de texto clínico (eliminación de artefactos XML, "
            "normalización de abreviaturas médicas), extracción de secciones FINDINGS/IMPRESSION, "
            "tokenización GPT-2, padding/truncation a 256 tokens."),
        bul("<b>Split por UID de estudio</b>: la vista frontal y lateral del mismo paciente siempre "
            "quedan en la misma partición, garantizando que el modelo no 'vea' durante validación "
            "imágenes del mismo caso que en entrenamiento."),
        sp(0.3),
        h2("Consideración sobre vistas múltiples"),
        b("Cada estudio radiológico suele incluir dos vistas: posteroanterior (PA, frontal) y lateral. "
          "Ambas vistas comparten el mismo informe. El modelo las trata como pares independientes "
          "imagen-texto, lo que duplica los ejemplos de entrenamiento sin filtrado adicional. "
          "El split por UID garantiza que ambas vistas del mismo estudio estén siempre en "
          "el mismo conjunto."),
    ]

    # ══ 4. ENTRENAMIENTO ══════════════════════════════════════════════════════
    s += [
        sp(0.5),
        h1("4. Entrenamiento — Fase 3 (En Curso)"),
        hr(),
        h2("Stage 1: Alineamiento Visual-Textual (Completado)"),
        b("Entrenamiento contrastivo estilo CLIP con pérdida InfoNCE. El encoder aprende que "
          "la imagen y el informe de un mismo caso son similares en el espacio latente, "
          "mientras que pares imagen-informe de distintos casos son disimilares. "
          "Este stage proporciona un encoder visual con semántica clínica que Stage 2 hereda."),
        sp(0.3),
    ]

    s1 = [
        [ph("PARÁMETRO"), ph("VALOR")],
        [pc("Dataset / splits"),       pc("Open-I: 5.231 train / 1.120 val")],
        [pc("Arquitectura"),           pc("ViT-Base/16 + projection heads (256D)")],
        [pc("Objetivo de pérdida"),    pc("InfoNCE contrastiva — batch size efectivo 64")],
        [pc("Learning rate"),          pc("1e-4 con scheduler coseno y weight decay 1e-4")],
        [pc("Hardware / velocidad"),   pc("RTX 4070 12 GB · fp16-mixed · ~16 it/s")],
        [pc("Criterio de parada"),     pc("Early stopping — 5 epochs sin mejora de val/loss")],
        [pc("Epochs hasta convergencia"), pc("13 epochs")],
        [pc("Mejor val/loss"),         pc("2.383 (epoch 7)  frente a 2.760 inicial (azar con batch 16)")],
        [pc("Checkpoint guardado"),    pc("stage1_epoch=07  — usado como punto de partida de Stage 2")],
    ]
    s.append(tabla(s1, [5.5*cm, 11.1*cm]))
    s += [sp(0.3), Paragraph("Tabla 4. Configuración y resultados Stage 1.", S_CAP), sp(0.3)]

    s += [
        h2("Stage 2: Generación de Informes (En Curso)"),
        b("Fine-tuning del decoder GPT-2 con cross-attention sobre los patch features del encoder "
          "Stage 1. El modelo aprende a generar texto condicionado en la imagen mediante "
          "language modelling (cross-entropy sobre next-token prediction). "
          "La evaluación se realiza con BLEU-1/4, ROUGE-L y METEOR comparando el informe "
          "generado contra el informe de referencia del radiólogo."),
        sp(0.3),
    ]

    s2 = [
        [ph("PARÁMETRO"), ph("VALOR")],
        [pc("Arquitectura"),         pc("ViT-Base/16 (encoder Stage 1) + GPT-2 + Cross-Attention (12 capas)")],
        [pc("Objetivo de pérdida"),  pc("Cross-entropy (language modelling)")],
        [pc("Learning rate"),        pc("5e-5 con warmup coseno (500 steps) y weight decay 1e-4")],
        [pc("Label smoothing"),      pc("0.1  — regularización para evitar overfitting en dataset pequeño")],
        [pc("Precision / hardware"), pc("fp16-mixed · RTX 4070 12 GB · ~0.09 it/s (generación beam)")],
        [pc("Epochs planificados"),  pc("Máx. 30 con early stopping (5 epochs sin mejora ROUGE-L)")],
        [pc("Métrica de seguimiento"),pc("val/rouge_l  — refleja mejor calidad de texto generado que val/loss")],
        [pc("Métricas de evaluación"),pc("BLEU-1, BLEU-4, ROUGE-L, METEOR (sobre conjunto test al finalizar)")],
        [pc("Estado"),               pc("En entrenamiento — resultados finales pendientes")],
    ]
    s.append(tabla(s2, [5.5*cm, 11.1*cm]))
    s += [sp(0.3), Paragraph("Tabla 5. Configuración Stage 2.", S_CAP)]

    # ══ 5. INFRAESTRUCTURA ════════════════════════════════════════════════════
    s += [
        PageBreak(),
        h1("5. Infraestructura Técnica"),
        hr(),
        sp(0.2),
    ]

    infra = [
        [ph("CATEGORÍA"),      ph("COMPONENTE"),                                 ph("VERSIÓN")],
        [pc("Hardware"),       pc("NVIDIA RTX 4070 (Ada Lovelace) — 12 GB VRAM"), pc("CUDA 12.1 · driver 13.2")],
        [pc("Framework"),      pc("PyTorch + PyTorch Lightning"),                  pc("2.2.0 / 2.2.0")],
        [pc("Encoder"),        pc("timm ViT-Base/16 (pretrained ImageNet-21k)"),   pc("timm 0.9.16")],
        [pc("Decoder"),        pc("HuggingFace GPT-2"),                            pc("transformers 4.40.0")],
        [pc("XAI"),            pc("Captum — Integrated Gradients + Layer Attn"),   pc("captum 0.7.0")],
        [pc("Augmentación"),   pc("Albumentations"),                               pc("1.4.3")],
        [pc("Métricas NLP"),   pc("evaluate + sacrebleu + nltk"),                  pc("0.4.1 / 2.4.0 / 3.8.1")],
        [pc("Config"),         pc("OmegaConf / Hydra"),                            pc("—")],
        [pc("Logging"),        pc("TensorBoard"),                                   pc("2.16.2")],
        [pc("Precision"),      pc("16-mixed — Tensor Cores Ada Lovelace"),          pc("—")],
        [pc("Batch efectivo"), pc("64 = batch 16 x grad_accumulation 4"),           pc("—")],
    ]
    s.append(tabla(infra, [3.2*cm, 8.4*cm, 5.0*cm]))
    s += [sp(0.3), Paragraph("Tabla 6. Stack tecnológico.", S_CAP)]

    # ══ 6. PLAN XAI ══════════════════════════════════════════════════════════
    s += [
        sp(0.5),
        h1("6. Plan de Análisis XAI — Fase 4"),
        hr(),
        b("El módulo XAI es la contribución central del TFM. Una vez completado Stage 2, "
          "se implementará el pipeline de generación de heatmaps y la comparación sistemática "
          "entre ambos métodos. El objetivo es responder: <i>¿cuál de los dos métodos produce "
          "heatmaps más alineados con el criterio clínico de un radiólogo?</i>"),
        sp(0.3),
        h2("Método 1 — Cross-Attention Rollout"),
        bul("Extracción de los pesos de cross-attention de las 12 capas del decoder durante la generación."),
        bul("Acumulación por rollout (producto matricial capa a capa) para obtener la atención efectiva."),
        bul("Agrupación de tokens por frase y promediado para obtener un mapa 14×14 por frase."),
        bul("Coste: nulo (los pesos ya se calculan durante la generación normal)."),
        sp(0.2),
        h2("Método 2 — Integrated Gradients (Captum)"),
        bul("Interpolación lineal desde imagen negra (baseline) hasta imagen real en 50 pasos."),
        bul("Cálculo del gradiente de cada token generado respecto a los patch features en cada paso."),
        bul("Integración numérica (regla del trapecio) sobre los 50 pasos → atribución por patch."),
        bul("Coste: ~50x el coste de inferencia — evaluación offline sobre muestra del test set."),
        sp(0.2),
        h2("Métricas de Evaluación XAI"),
        bul("<b>Correlación de Spearman</b> entre los mapas de ambos métodos: concordancia global."),
        bul("<b>Análisis cualitativo</b>: inspección visual sobre 20-30 casos seleccionados — "
            "¿señala el heatmap la región correcta según el hallazgo textual?"),
        bul("<b>Coherencia clínica</b>: para frases con hallazgos localizables (opacidad, derrame, "
            "cardiomegalia), verificar que el heatmap activa la zona anatómica correspondiente."),
    ]

    # ══ 7. PRÓXIMOS PASOS Y CRONOGRAMA ════════════════════════════════════════
    s += [
        PageBreak(),
        h1("7. Próximos Pasos y Cronograma"),
        hr(),
        h2("Inmediato — Completar Stage 2"),
        bul("Finalizar entrenamiento Stage 2 en hardware local (RTX 4070). "
            "Estimación: 6-10 epochs adicionales, ~4-8 horas de GPU."),
        bul("Evaluar modelo en test set: BLEU-1/4, ROUGE-L, METEOR."),
        bul("Generar 20-30 informes de ejemplo y análisis cualitativo inicial."),
        sp(0.3),
    ]

    crono = [
        [ph("FASE"),      ph("DESCRIPCIÓN"),                              ph("PLAZO"),           ph("ESTADO")],
        [pc("Fase 1"),   pc("Pipeline datos + EDA"),                      pcc("Completada"),     _badge("DONE",      VERDE)],
        [pc("Fase 2"),   pc("Arquitectura encoder-decoder + XAI"),        pcc("Completada"),     _badge("DONE",      VERDE)],
        [pc("Fase 3"),   pc("Entrenamiento Stage 1 + Stage 2"),           pcc("Mayo 2026"),      _badge("EN CURSO",  NARANJA)],
        [pc("Fase 4"),   pc("Módulo XAI — heatmaps + comparativa"),       pcc("Jun 2026"),       _badge("SIGUIENTE", AZUL_MED)],
        [pc("Fase 5"),   pc("Evaluación NLP + análisis cualitativo"),     pcc("Jun–Jul 2026"),   _badge("PENDIENTE", GRIS_TXT)],
        [pc("Fase 6"),   pc("Demo interactiva + Memoria TFM"),            pcc("Jul–Sep 2026"),   _badge("PENDIENTE", GRIS_TXT)],
        [pc("Defensa"),  pc("Presentación oral"),                         pcc("Sep 2026"),       _badge("PENDIENTE", GRIS_TXT)],
    ]
    s.append(tabla(crono, [2.0*cm, 7.0*cm, 3.2*cm, 2.8*cm]))
    s += [sp(0.3), Paragraph("Tabla 7. Cronograma del proyecto.", S_CAP), sp(0.8)]

    s += [
        h2("Riesgos y Mitigaciones"),
    ]

    riesgos = [
        [ph("RIESGO"),                                   ph("PROBABILIDAD"), ph("MITIGACIÓN")],
        [pc("Stage 2 no converge (ROUGE-L < 0.15)"),     pcc("Baja"),   pc("Ajustar LR, añadir label smoothing, probar BioGPT")],
        [pc("Heatmaps XAI no son clínicamente coherentes"), pcc("Media"), pc("Comparar ambos métodos; el análisis comparativo es la aportación aunque los resultados sean modestos")],
        [pc("Tiempo de GPU insuficiente"),               pcc("Baja"),   pc("MIMIC-CXR como dataset adicional; reducir epochs Stage 2")],
        [pc("Dataset Open-I demasiado pequeño"),         pcc("Media"),  pc("Data augmentation agresiva; Stage 1 + pretraining ImageNet mitigan la escasez")],
    ]
    s.append(tabla(riesgos, [5.5*cm, 2.5*cm, 8.6*cm]))
    s += [sp(0.3), Paragraph("Tabla 8. Análisis de riesgos.", S_CAP)]

    # ══ 8. REFERENCIAS ════════════════════════════════════════════════════════
    s += [
        sp(0.5),
        h1("8. Referencias Principales"),
        hr(),
        bul("Johnson et al. (2019). MIMIC-CXR: a de-identified publicly available database "
            "of chest radiographs with free-text reports. <i>Scientific Data.</i>"),
        bul("Vaswani et al. (2017). Attention Is All You Need. <i>NeurIPS.</i>"),
        bul("Dosovitskiy et al. (2021). An Image is Worth 16x16 Words: Transformers for "
            "Image Recognition at Scale. <i>ICLR 2021.</i>"),
        bul("Chefer et al. (2021). Transformer Interpretability Beyond Attention Visualization. "
            "<i>CVPR 2021.</i>"),
        bul("Sundararajan et al. (2017). Axiomatic Attribution for Deep Networks "
            "(Integrated Gradients). <i>ICML 2017.</i>"),
        bul("Zhang et al. (2023). Radiology Report Generation with Cross-Modal Memory. "
            "<i>IEEE TMI.</i>"),
        bul("OpenAI (2019). GPT-2: Language Models are Unsupervised Multitask Learners."),
    ]

    return s


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

# ─── BUILD ────────────────────────────────────────────────────────────────────
def main():
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH), pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.8*cm,  bottomMargin=1.4*cm,
        title="Informe de Seguimiento TFM — Generación de Informes Radiológicos",
        author="Daniel Gómez Arranz",
    )
    doc.build(story(), onFirstPage=on_first_page, onLaterPages=on_page)
    print(f"PDF generado: {OUTPUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
