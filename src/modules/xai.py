"""
XAI module — the core contribution of this TFM.

Two methods are implemented and compared:

① CrossAttentionExtractor
   Aggregates the decoder cross-attention weights (token × image_patch) across
   heads and layers to produce a spatial heatmap per generated token.
   Fast, no extra compute, but attention ≠ causality (Jain & Wallace, 2019).

② IntegratedGradientsVisualizer
   Computes the attribution of each generated token w.r.t. the visual patch
   features via Integrated Gradients (Sundararajan et al., 2017) through Captum.
   Theoretically more faithful but ~n_steps× slower.

ReportHeatmapAligner is the high-level interface: it takes a generated report
segmented into sentences and returns one heatmap per sentence.

── Interfaz real del modelo (RadiologyReportModel) ────────────────────────────
Este módulo NO usa hooks ni parsea tuplas de salida. Cada `CrossAttentionLayer`
del decoder guarda sus pesos en `.last_attn_weights` (B, H, T, S) tras un forward,
y se exponen vía `model.cross_attention_layers`. El forward de entrenamiento usa
`input_ids` (no `decoder_input_ids`). Ver src/models/decoder.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ─── DATA CLASSES ────────────────────────────────────────────────────────────

@dataclass
class TokenHeatmap:
    """Heatmap for a single generated token."""
    token: str
    token_idx: int
    heatmap: np.ndarray          # (H, W) float32, values in [0, 1]
    method: str                  # "cross_attention" | "integrated_gradients"


@dataclass
class SentenceHeatmap:
    """Aggregated heatmap for a sentence span in the generated report."""
    sentence: str
    token_span: tuple[int, int]  # (start, end) token indices
    heatmap: np.ndarray          # (H, W) float32, values in [0, 1]
    method: str
    peak_region: Optional[np.ndarray] = None  # binary mask of top-20% activated pixels


# ─── METHOD 1: CROSS-ATTENTION ───────────────────────────────────────────────

class CrossAttentionExtractor:
    """
    Extrae heatmaps por token de los pesos de cross-attention del decoder.

    Cada `CrossAttentionLayer` guarda `last_attn_weights` de forma (B, H, T, S)
    tras un forward, con S = nº de patches visuales (196 en ViT-Base/16 @224).
    Agregación:
        - heads: "mean" (def) | "max"
        - capas: "mean" (def) | "last"
    Nota: el "rollout" de Abnar & Zuidema (2020) aplica a self-attention (matrices
    cuadradas T×T). La cross-attention es T×S y no compone entre capas, así que
    aquí "rollout" = agregación multi-capa (media), documentado como tal.
    """

    def __init__(self, model, head_agg: str = "mean", layer_agg: str = "mean"):
        assert head_agg in ("mean", "max")
        assert layer_agg in ("mean", "last")
        self.model = model
        self.head_agg = head_agg
        self.layer_agg = layer_agg

    @torch.no_grad()
    def _collect_attention(
        self, image_tensor: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Forward para poblar last_attn_weights y devolver (L, H, T, S)."""
        self.model.eval()
        _ = self.model(pixel_values=image_tensor, input_ids=input_ids)
        layers = self.model.cross_attention_layers
        weights = [l.last_attn_weights for l in layers]
        if any(w is None for w in weights):
            raise RuntimeError(
                "last_attn_weights vacío en alguna capa. ¿El forward usó "
                "visual features? (RadiologyReportModel.forward las inyecta)."
            )
        # cada w: (B, H, T, S) → asume B=1
        return torch.stack([w.squeeze(0).float().cpu() for w in weights], dim=0)

    def _aggregate(self, stacked: torch.Tensor) -> torch.Tensor:
        """(L, H, T, S) → (T, S) agregando heads y capas."""
        heads = stacked.mean(1) if self.head_agg == "mean" else stacked.max(1).values
        return heads.mean(0) if self.layer_agg == "mean" else heads[-1]

    def extract(
        self,
        image_tensor: torch.Tensor,         # (1, C, H, W)
        input_ids: torch.Tensor,            # (1, T) — ids del informe (generado)
        patch_grid: tuple[int, int] = (14, 14),
        tokenizer=None,
    ) -> list[TokenHeatmap]:
        """Devuelve un TokenHeatmap por token del informe."""
        stacked = self._collect_attention(image_tensor, input_ids)
        attn = self._aggregate(stacked)                 # (T, S)

        h_patches, w_patches = patch_grid
        ids = input_ids[0].tolist()
        heatmaps = []
        for t_idx in range(attn.shape[0]):
            spatial = attn[t_idx].reshape(h_patches, w_patches).numpy()
            spatial = _minmax(spatial)
            tok_str = tokenizer.decode([ids[t_idx]]) if tokenizer else str(t_idx)
            heatmaps.append(TokenHeatmap(
                token=tok_str, token_idx=t_idx,
                heatmap=spatial, method="cross_attention",
            ))
        return heatmaps


# ─── METHOD 2: INTEGRATED GRADIENTS ──────────────────────────────────────────

class IntegratedGradientsVisualizer:
    """
    Atribución por token con Integrated Gradients (Captum) respecto a los
    patch features visuales. Satisface el axioma de completeness; cuesta
    ~n_steps forwards por token → usar por frase, no por token, en producción.

    Referencia: Sundararajan et al. (2017).
    """

    def __init__(self, model, n_steps: int = 50):
        try:
            from captum.attr import IntegratedGradients
        except ImportError as e:
            raise ImportError("captum es necesario: pip install captum") from e
        self.model = model
        self.n_steps = n_steps
        self._ig = IntegratedGradients(self._forward_func)

    def _forward_func(
        self,
        visual_features: torch.Tensor,      # (1, S, D_v) — input a atribuir
        input_ids: torch.Tensor,
        target_pos: int,
    ) -> torch.Tensor:
        """Devuelve los logits en target_pos condicionados en visual_features."""
        dec = self.model.decoder
        dec._set_visual_features(dec.visual_proj(visual_features))
        try:
            out = dec.gpt2(input_ids=input_ids)
        finally:
            dec._clear_visual_features()
        return out.logits[:, target_pos, :]             # (1, V)

    def compute(
        self,
        visual_features: torch.Tensor,     # (1, S, D_v)
        input_ids: torch.Tensor,           # (1, T)
        target_pos: int,                   # posición cuyo logit se explica
        target_token_id: int,              # token objetivo (dim de vocab)
        patch_grid: tuple[int, int] = (14, 14),
    ) -> TokenHeatmap:
        baseline = torch.zeros_like(visual_features)
        attrs = self._ig.attribute(
            inputs=visual_features,
            baselines=baseline,
            target=int(target_token_id),
            additional_forward_args=(input_ids, target_pos),
            n_steps=self.n_steps,
        )
        # (1, S, D) → magnitud sobre D → (S,) → (h, w)
        spatial = attrs.squeeze(0).abs().sum(-1).detach().cpu().numpy()
        h, w = patch_grid
        spatial = _minmax(spatial.reshape(h, w))
        return TokenHeatmap(
            token=str(target_pos), token_idx=target_pos,
            heatmap=spatial, method="integrated_gradients",
        )


# ─── HIGH-LEVEL INTERFACE ─────────────────────────────────────────────────────

class ReportHeatmapAligner:
    """
    Interfaz principal XAI.

    Dado un informe (generado) y una radiografía, produce un SentenceHeatmap por
    frase con los métodos habilitados, y permite comparar su concordancia.

    Uso:
        aligner = ReportHeatmapAligner(model, tokenizer)
        res = aligner.align(image_tensor, report_text)          # dict por método
        overlay = aligner.overlay(image_np, res["cross_attention"][0])
    """

    def __init__(
        self,
        model,
        tokenizer,
        image_size: int = 224,
        patch_size: int = 16,
        xai_methods: tuple[str, ...] = ("cross_attention", "integrated_gradients"),
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_grid = getattr(model, "patch_grid", None) \
            or (image_size // patch_size, image_size // patch_size)
        self.xai_methods = xai_methods

        self._ca = CrossAttentionExtractor(model) if "cross_attention" in xai_methods else None
        self._ig = IntegratedGradientsVisualizer(model) if "integrated_gradients" in xai_methods else None

    @torch.no_grad()
    def generate_report(self, image_tensor: torch.Tensor, **gen_kwargs) -> str:
        """Genera el informe para la imagen (conveniencia)."""
        return self.model.generate(image_tensor, tokenizer=self.tokenizer, **gen_kwargs)[0]

    def align(
        self,
        image_tensor: torch.Tensor,      # (1, C, H, W)
        report_text: str,                # informe (típicamente el generado)
    ) -> dict[str, list[SentenceHeatmap]]:
        sentences = self._split_sentences(report_text)
        enc = self.tokenizer(report_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(image_tensor.device)

        results: dict[str, list[SentenceHeatmap]] = {}

        if self._ca is not None:
            token_maps = self._ca.extract(image_tensor, input_ids, self.patch_grid, self.tokenizer)
            results["cross_attention"] = self._aggregate_by_sentence(
                token_maps, sentences, "cross_attention")

        if self._ig is not None:
            visual = self.model.encode_image(image_tensor)      # (1, S, D)
            ids = input_ids[0].tolist()
            ig_maps: list[TokenHeatmap] = []
            for t in range(len(ids)):
                # el token en t se predice desde la posición t-1
                pos = max(t - 1, 0)
                tm = self._ig.compute(visual, input_ids, pos, ids[t], self.patch_grid)
                ig_maps.append(tm)
            results["integrated_gradients"] = self._aggregate_by_sentence(
                ig_maps, sentences, "integrated_gradients")

        return results

    # ─── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        import re
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _aggregate_by_sentence(
        self, token_maps: list[TokenHeatmap], sentences: list[str], method: str,
    ) -> list[SentenceHeatmap]:
        """Mean-pool de los heatmaps de token dentro de cada frase.

        Mapea tokens→frase por conteo de sub-tokens del tokenizador (más preciso
        que contar palabras)."""
        out, cursor = [], 0
        n_total = len(token_maps)
        for sent in sentences:
            n_tok = max(1, len(self.tokenizer.encode(sent)))
            end = min(cursor + n_tok, n_total)
            span = [tm.heatmap for tm in token_maps[cursor:end]]
            if span:
                agg = _minmax(np.mean(span, axis=0))
                peak = (agg >= np.percentile(agg, 80)).astype(np.uint8)
                out.append(SentenceHeatmap(
                    sentence=sent, token_span=(cursor, end),
                    heatmap=agg, method=method, peak_region=peak))
            cursor = end
        return out

    def overlay(
        self,
        original_image: np.ndarray,
        sentence_heatmap: SentenceHeatmap,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Superpone un heatmap de frase sobre la radiografía original."""
        if original_image.ndim == 2:
            base = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            base = original_image.copy()
        h, w = base.shape[:2]
        hm = cv2.resize(sentence_heatmap.heatmap, (w, h))
        hm_color = cv2.applyColorMap((hm * 255).astype(np.uint8), colormap)
        return cv2.addWeighted(base, 1 - alpha, hm_color, alpha, 0)

    @staticmethod
    def fidelity_correlation(sh_ca: SentenceHeatmap, sh_ig: SentenceHeatmap) -> float:
        """Correlación de Spearman entre los mapas de ambos métodos (concordancia)."""
        from scipy.stats import spearmanr
        rho, _ = spearmanr(sh_ca.heatmap.flatten(), sh_ig.heatmap.flatten())
        return float(rho)


# ─── UTILS ────────────────────────────────────────────────────────────────────

def _minmax(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min() + 1e-8)
