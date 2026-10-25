"""
Evaluación de anclaje clínico (clinical grounding) de los heatmaps XAI contra
bounding boxes ground-truth de NIH ChestX-ray8.

Pregunta: cuando el modelo genera un hallazgo (p.ej. "cardiomegaly"), ¿el heatmap
cae sobre la región anatómica correcta según el bbox del radiólogo?

Métricas (estándar en localización débilmente supervisada de CXR):
  - pointing_game : ¿el pico del heatmap cae dentro del bbox? (hit/miss)
  - energy_in_bbox: fracción de la energía del heatmap dentro del bbox
  - iobb          : Intersection-over-Bbox del heatmap umbralizado (top-k%)
  - baseline aleatorio: fracción de área del bbox (referencia para energy)

Los bboxes de NIH están en coordenadas de la imagen original 1024×1024; el heatmap
del modelo se produce a 14×14 y se re-escala a la resolución de entrada (224).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# ─── PRIMITIVAS ──────────────────────────────────────────────────────────────

def scale_bbox(bbox, from_size: int, to_size: int) -> tuple[float, float, float, float]:
    """Escala (x, y, w, h) de una resolución cuadrada a otra."""
    s = to_size / from_size
    x, y, w, h = bbox
    return (x * s, y * s, w * s, h * s)


def bbox_mask(bbox, shape: tuple[int, int]) -> np.ndarray:
    """Máscara booleana (H, W) del bbox (x, y, w, h) en coords de `shape`."""
    x, y, w, h = bbox
    m = np.zeros(shape, dtype=bool)
    x0, y0 = int(round(x)), int(round(y))
    x1, y1 = int(round(x + w)), int(round(y + h))
    m[max(0, y0):min(shape[0], y1), max(0, x0):min(shape[1], x1)] = True
    return m


def pointing_game(heatmap: np.ndarray, bbox) -> bool:
    """¿El píxel de máxima activación cae dentro del bbox? (bbox en coords del heatmap)."""
    yx = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    return bool(bbox_mask(bbox, heatmap.shape)[yx])


def energy_in_bbox(heatmap: np.ndarray, bbox) -> float:
    """Fracción de la energía (suma) del heatmap contenida en el bbox."""
    m = bbox_mask(bbox, heatmap.shape)
    hm = np.clip(heatmap, 0, None)
    return float(hm[m].sum() / (hm.sum() + 1e-8))


def iobb(heatmap: np.ndarray, bbox, percentile: float = 80.0) -> float:
    """Intersection-over-Bbox del heatmap umbralizado al percentil `percentile`."""
    thr = np.percentile(heatmap, percentile)
    pred = heatmap >= thr
    gt = bbox_mask(bbox, heatmap.shape)
    inter = np.logical_and(pred, gt).sum()
    return float(inter / (gt.sum() + 1e-8))


def bbox_area_fraction(bbox, shape: tuple[int, int]) -> float:
    """Fracción de área del bbox — baseline aleatorio para energy_in_bbox."""
    return float(bbox_mask(bbox, shape).sum() / (shape[0] * shape[1]))


# ─── EVALUACIÓN DE UN CASO ────────────────────────────────────────────────────

@dataclass
class GroundingScore:
    pointing_hit: bool
    energy: float
    iobb: float
    baseline: float          # fracción de área (energy esperada si fuese aleatorio)
    lift: float              # energy / baseline (>1 = mejor que azar)


def evaluate_localization(
    heatmap: np.ndarray,          # (h, w) del modelo (p.ej. 14×14), valores >=0
    bbox_orig,                    # (x, y, w, h) en coords de la imagen original
    orig_size: int = 1024,        # tamaño de la imagen original (NIH: 1024)
    out_size: int = 224,          # resolución a la que evaluar (entrada del modelo)
    iobb_percentile: float = 80.0,
) -> GroundingScore:
    """Re-escala el heatmap a out_size, escala el bbox, y calcula las métricas."""
    hm = cv2.resize(heatmap.astype(np.float32), (out_size, out_size))
    bbox = scale_bbox(bbox_orig, orig_size, out_size)
    shape = (out_size, out_size)
    energy = energy_in_bbox(hm, bbox)
    base = bbox_area_fraction(bbox, shape)
    return GroundingScore(
        pointing_hit=pointing_game(hm, bbox),
        energy=energy,
        iobb=iobb(hm, bbox, iobb_percentile),
        baseline=base,
        lift=energy / (base + 1e-8),
    )
