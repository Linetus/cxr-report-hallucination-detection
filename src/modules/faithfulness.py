"""
Fidelidad (faithfulness) de las explicaciones XAI por perturbación.

Un heatmap es FIEL si borrar las regiones que marca como importantes cambia de
verdad la salida del modelo. Permite comparar cuantitativamente Cross-Attention
vs Integrated Gradients (¿cuál explica el razonamiento real, no solo lo plausible?).

Se perturban los PATCH FEATURES visuales (S=196) y se mide la probabilidad del
token objetivo generado:

  - deletion   : borrar patches en orden de importancia ↓; prob debe caer rápido.
                 AUC menor = más fiel.
  - insertion  : partir de baseline e ir añadiendo patches importantes; prob debe
                 subir rápido. AUC mayor = más fiel.
  - comprehensiveness (ERASER, DeYoung 2020): prob(full) − prob(sin top-k%).
                 Mayor = más fiel (lo importante sí lo era).
  - sufficiency: prob(full) − prob(solo top-k%). Cercano a 0 = más fiel
                 (lo importante basta para reproducir la predicción).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def _prob(model, visual, input_ids, pos: int, tid: int) -> float:
    """Probabilidad del token `tid` en la posición `pos` dado visual."""
    out = model.decoder.decode(input_ids=input_ids, visual_features=visual)
    return float(F.softmax(out.logits[:, pos, :], dim=-1)[0, tid])


def _order(importance) -> np.ndarray:
    return np.argsort(-np.asarray(importance, dtype=np.float64))   # más importante primero


@torch.no_grad()
def deletion_curve(model, visual, input_ids, pos, tid, importance, n_steps=14, baseline=None):
    S = visual.shape[1]
    order = _order(importance)
    base = torch.zeros_like(visual) if baseline is None else baseline
    v = visual.clone()
    probs = [_prob(model, v, input_ids, pos, tid)]
    step = max(1, S // n_steps)
    for i in range(0, S, step):
        v[:, order[i:i + step], :] = base[:, order[i:i + step], :]
        probs.append(_prob(model, v, input_ids, pos, tid))
    return np.array(probs)


@torch.no_grad()
def insertion_curve(model, visual, input_ids, pos, tid, importance, n_steps=14, baseline=None):
    S = visual.shape[1]
    order = _order(importance)
    base = torch.zeros_like(visual) if baseline is None else baseline
    v = base.clone()
    probs = [_prob(model, v, input_ids, pos, tid)]
    step = max(1, S // n_steps)
    for i in range(0, S, step):
        v[:, order[i:i + step], :] = visual[:, order[i:i + step], :]
        probs.append(_prob(model, v, input_ids, pos, tid))
    return np.array(probs)


def auc(curve: np.ndarray) -> float:
    """Área normalizada bajo la curva (media)."""
    return float(np.trapz(curve) / (len(curve) - 1))


@torch.no_grad()
def comprehensiveness(model, visual, input_ids, pos, tid, importance, k=0.2, baseline=None):
    S = visual.shape[1]
    topk = _order(importance)[:max(1, int(S * k))]
    base = torch.zeros_like(visual) if baseline is None else baseline
    v = visual.clone(); v[:, topk, :] = base[:, topk, :]
    return _prob(model, visual, input_ids, pos, tid) - _prob(model, v, input_ids, pos, tid)


@torch.no_grad()
def sufficiency(model, visual, input_ids, pos, tid, importance, k=0.2, baseline=None):
    S = visual.shape[1]
    topk = _order(importance)[:max(1, int(S * k))]
    base = torch.zeros_like(visual) if baseline is None else baseline
    v = base.clone(); v[:, topk, :] = visual[:, topk, :]
    return _prob(model, visual, input_ids, pos, tid) - _prob(model, v, input_ids, pos, tid)


@dataclass
class FaithfulnessScore:
    deletion_auc: float        # menor = más fiel
    insertion_auc: float       # mayor = más fiel
    comprehensiveness: float   # mayor = más fiel
    sufficiency: float         # cercano a 0 = más fiel


def evaluate_faithfulness(
    model, visual, input_ids, pos: int, tid: int, importance,
    n_steps: int = 14, k: float = 0.2, baseline=None,
) -> FaithfulnessScore:
    """Calcula las 4 métricas de fidelidad para un heatmap (importance sobre S patches)."""
    return FaithfulnessScore(
        deletion_auc=auc(deletion_curve(model, visual, input_ids, pos, tid, importance, n_steps, baseline)),
        insertion_auc=auc(insertion_curve(model, visual, input_ids, pos, tid, importance, n_steps, baseline)),
        comprehensiveness=comprehensiveness(model, visual, input_ids, pos, tid, importance, k, baseline),
        sufficiency=sufficiency(model, visual, input_ids, pos, tid, importance, k, baseline),
    )
