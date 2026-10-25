"""
Language Decoder — GPT-2 adapted for conditional image-to-text generation.

HuggingFace's GPT2LMHeadModel does not natively have cross-attention to an
external encoder. We inject cross-attention by:
  1. Replacing each GPT-2 block with GPT2BlockWithCrossAttention, which
     applies cross-attention to visual patch features after self-attention.
  2. Visual features are stored on each block as _visual_features before
     the forward pass — no monkey-patching or closures required.
  3. Cross-attention weights are stored in each CrossAttentionLayer so the
     XAI module can extract them without re-running the model.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel


# ─── CROSS-ATTENTION INJECTION ───────────────────────────────────────────────

class CrossAttentionLayer(nn.Module):
    """
    Multi-head cross-attention from decoder hidden states to visual patch features.

    Weights are stored in self.last_attn_weights after each forward call —
    the XAI module reads from here without requiring hooks.
    Shape: (B, num_heads, tgt_len, src_len)
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        # Gate tanh inicializado a 0 (estilo Flamingo / LLaMA-Adapter): al arrancar la
        # cross-attention es un no-op, preservando GPT-2 preentrenado intacto. Sin esto,
        # la cross-attn sin entrenar destruye el residual stream y el modelo colapsa.
        self.gate = nn.Parameter(torch.zeros(1))

        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        hidden_states: torch.Tensor,          # (B, T, D)
        visual_features: torch.Tensor,        # (B, S, D)
    ) -> torch.Tensor:
        B, T, D = hidden_states.shape
        S = visual_features.shape[1]

        # Pre-norm sobre el stream para la query (NO re-normaliza el residual completo)
        h = self.norm(hidden_states)
        Q = self.q_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(visual_features).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(visual_features).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, T, S)
        attn_weights = F.softmax(attn_logits, dim=-1)
        self.last_attn_weights = attn_weights.detach()

        attn_out = torch.matmul(self.dropout(attn_weights), V)       # (B, H, T, D_h)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.out_proj(attn_out)

        # Residual con gate tanh (gate=0 al inicio → devuelve hidden_states intacto)
        return hidden_states + torch.tanh(self.gate) * attn_out


class GPT2BlockWithCrossAttention(nn.Module):
    """
    Wraps a GPT-2 transformer block and inserts cross-attention after
    the self-attention sublayer, following the encoder-decoder pattern
    from Vaswani et al. (2017).

    Visual features are injected via self._visual_features, set by
    ReportDecoder.decode() / .generate() before each forward pass.
    This avoids monkey-patching and works correctly with beam search
    (the caller expands _visual_features to match the expanded batch).
    """

    def __init__(self, gpt2_block, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.gpt2_block = gpt2_block
        self.cross_attn = CrossAttentionLayer(hidden_dim, num_heads, dropout)
        self._visual_features: Optional[torch.Tensor] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> tuple:
        outputs = self.gpt2_block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hs = outputs[0]

        if self._visual_features is not None:
            hs = self.cross_attn(hs, self._visual_features)

        return (hs,) + outputs[1:]


# ─── DECODER ─────────────────────────────────────────────────────────────────

class ReportDecoder(nn.Module):
    """
    GPT-2 decoder with cross-attention injected into every transformer block.

    Args:
        model_name: HuggingFace model name ("gpt2", "distilgpt2", "microsoft/BioGPT")
        visual_dim: dimensionality of visual patch features from the encoder
        num_cross_attn_heads: heads in each cross-attention layer
        pretrained: load GPT-2 pretrained weights before injecting cross-attention
        dropout: applied in cross-attention layers
        max_length: maximum generation length
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        visual_dim: int = 768,
        num_cross_attn_heads: int = 8,
        pretrained: bool = True,
        dropout: float = 0.1,
        max_length: int = 256,
    ):
        super().__init__()
        self.max_length = max_length

        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name) if pretrained \
            else GPT2LMHeadModel(GPT2Config())
        hidden_dim = self.gpt2.config.hidden_size

        self.visual_proj = nn.Linear(visual_dim, hidden_dim) \
            if visual_dim != hidden_dim else nn.Identity()

        new_blocks = nn.ModuleList([
            GPT2BlockWithCrossAttention(block, hidden_dim, num_cross_attn_heads, dropout)
            for block in self.gpt2.transformer.h
        ])
        self.gpt2.transformer.h = new_blocks

        self.hidden_dim = hidden_dim

    @property
    def cross_attention_layers(self) -> list[CrossAttentionLayer]:
        """Returns all cross-attention layers — used by XAI module."""
        return [block.cross_attn for block in self.gpt2.transformer.h]

    def _set_visual_features(self, vis: torch.Tensor) -> None:
        for block in self.gpt2.transformer.h:
            block._visual_features = vis

    def _clear_visual_features(self) -> None:
        for block in self.gpt2.transformer.h:
            block._visual_features = None

    def decode(
        self,
        input_ids: torch.Tensor,            # (B, T)
        visual_features: torch.Tensor,      # (B, S, D_v)
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> "CausalLMOutput":
        """Teacher-forced forward pass for training."""
        vis = self.visual_proj(visual_features)     # (B, S, D_gpt2)
        self._set_visual_features(vis)
        try:
            output = self.gpt2(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        finally:
            self._clear_visual_features()
        return output

    @torch.no_grad()
    def generate(
        self,
        visual_features: torch.Tensor,      # (B, S, D_v)
        tokenizer,
        max_new_tokens: int = 200,
        num_beams: int = 4,
        bos_token_id: Optional[int] = None,
    ) -> list[str]:
        """
        Autoregressive report generation.

        visual_features are expanded by num_beams to match GPT-2's internal
        beam search batch expansion (B → B * num_beams).
        """
        B = visual_features.shape[0]
        device = visual_features.device
        vis = self.visual_proj(visual_features)

        # Beam search expands batch: (B, S, D) → (B * num_beams, S, D)
        vis_expanded = vis.repeat_interleave(num_beams, dim=0)
        self._set_visual_features(vis_expanded)

        bos = bos_token_id or tokenizer.bos_token_id or tokenizer.eos_token_id
        input_ids = torch.full((B, 1), bos, dtype=torch.long, device=device)

        try:
            output_ids = self.gpt2.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        finally:
            self._clear_visual_features()

        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)
