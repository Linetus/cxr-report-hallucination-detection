"""
Visual Encoder — ViT-Base/16 pretrained via timm.

Wraps timm's ViT to expose:
  - patch-level features  (B, num_patches, D)  → cross-attention input for decoder
  - CLS token embedding   (B, D)               → contrastive alignment (Stage 1)

The patch grid for ViT-Base/16 on 224px input is 14×14 = 196 patches.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("timm is required: pip install timm")


class VisualEncoder(nn.Module):
    """
    ViT-Base/16 encoder that returns patch embeddings for cross-attention.

    Args:
        model_name: any timm ViT identifier (default vit_base_patch16_224)
        pretrained: load ImageNet-21k weights
        output_dim: project patch embeddings to this dim (None = keep ViT native dim)
        freeze_layers: freeze the first N transformer blocks (0 = train all)
        dropout: dropout on projected features
    """

    PATCH_DIM = {
        "vit_base_patch16_224": 768,
        "vit_large_patch16_224": 1024,
        "vit_small_patch16_224": 384,
        "densenet121": 1024,
    }

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        output_dim: int | None = None,
        freeze_layers: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.native_dim = self.PATCH_DIM.get(model_name, 768)
        self.output_dim = output_dim or self.native_dim

        # Load backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # remove classification head
        )

        # Freeze early blocks if requested
        if freeze_layers > 0 and hasattr(self.backbone, "blocks"):
            for i, block in enumerate(self.backbone.blocks):
                if i < freeze_layers:
                    for param in block.parameters():
                        param.requires_grad = False

        # Optional projection to a common dim (useful when combining ViT with CNN)
        if self.output_dim != self.native_dim:
            self.proj = nn.Sequential(
                nn.Linear(self.native_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
                nn.Dropout(dropout),
            )
        else:
            self.proj = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: (B, 3, H, W) normalized image tensor

        Returns:
            dict with:
                "patch_features":  (B, num_patches, output_dim)
                "cls_embedding":   (B, output_dim)
        """
        # timm ViT with num_classes=0 returns (B, D) by default (CLS token).
        # We need the full sequence. Use forward_features instead.
        features = self.backbone.forward_features(pixel_values)
        # features: (B, 1 + num_patches, D)  — first token is CLS

        cls_token = features[:, 0, :]           # (B, D)
        patch_tokens = features[:, 1:, :]        # (B, num_patches, D)

        patch_features = self.proj(self.dropout(patch_tokens))
        cls_embedding = self.proj(self.dropout(cls_token))

        return {
            "patch_features": patch_features,   # (B, num_patches, output_dim)
            "cls_embedding": cls_embedding,      # (B, output_dim)
        }

    @property
    def num_patches(self) -> int:
        """Number of visual patches (e.g. 196 for ViT-Base/16 on 224px)."""
        if hasattr(self.backbone, "patch_embed"):
            grid_size = self.backbone.patch_embed.grid_size
            return grid_size[0] * grid_size[1]
        return 196  # fallback

    @property
    def patch_grid(self) -> tuple[int, int]:
        """Spatial layout of patches as (h, w)."""
        if hasattr(self.backbone, "patch_embed"):
            return tuple(self.backbone.patch_embed.grid_size)
        return (14, 14)


class RadDinoEncoder(nn.Module):
    """
    Encoder RAD-DINO (microsoft/rad-dino) — ViT-B/14 DINOv2 preentrenado en CXR.

    Se usa CONGELADO como feature extractor (transfer): preserva sus features
    fuertes y elimina el mayor foco de overfitting sobre el dataset pequeño.
    Salida a 518px: CLS (768) + 1369 patches (37×37), 0 register tokens.

    Preprocesado esperado (ver RADDINO_MEAN/STD en CXRPreprocessor): 518×518,
    normalización propia mean=0.5307 std=0.2583 (NO ImageNet).
    """

    MODEL_ID = "microsoft/rad-dino"

    def __init__(self, freeze: bool = True, dropout: float = 0.0):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(self.MODEL_ID)
        self.native_dim = self.backbone.config.hidden_size          # 768
        self.output_dim = self.native_dim
        self._patch_size = self.backbone.config.patch_size          # 14
        self._img_size = self.backbone.config.image_size            # 518
        self.proj = nn.Identity()
        self.dropout = nn.Dropout(dropout)

        self.frozen = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True):
        """Mantiene el backbone en eval si está congelado (desactiva dropout/BN internos)."""
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            out = self.backbone(pixel_values=pixel_values)
        hs = out.last_hidden_state                  # (B, 1 + 1369, 768)
        cls = hs[:, 0, :]                           # (B, 768)
        patches = hs[:, 1:, :]                       # (B, 1369, 768)
        return {
            "patch_features": self.dropout(patches),
            "cls_embedding": self.dropout(cls),
        }

    @property
    def num_patches(self) -> int:
        side = self._img_size // self._patch_size
        return side * side                          # 37*37 = 1369

    @property
    def patch_grid(self) -> tuple[int, int]:
        side = self._img_size // self._patch_size
        return (side, side)                         # (37, 37)
