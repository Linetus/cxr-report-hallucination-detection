"""Test end-to-end de la integración M2 (RAD-DINO congelado) en CPU."""
import sys, glob
sys.path.insert(0, r"./")
import torch
torch.set_num_threads(4)
from transformers import AutoTokenizer
from src.data.preprocessing import CXRPreprocessor
from src.models.model import RadiologyReportModel
from src.modules.xai import CrossAttentionExtractor

png = glob.glob(r"./data/raw/open-i/*.png")[0]
pre = CXRPreprocessor(image_size=518, mode="eval",
                      mean=CXRPreprocessor.RADDINO_MEAN, std=CXRPreprocessor.RADDINO_STD)
img = pre(CXRPreprocessor.load_image(png)).unsqueeze(0)
print("1) Imagen preprocesada:", tuple(img.shape), "(debe ser 1,3,518,518)")

tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token; tok.padding_side = "right"
model = RadiologyReportModel(encoder_type="rad_dino", decoder_pretrained=True).eval()
print("2) patch_grid:", model.patch_grid, "| num_patches:", model.encoder.num_patches)

vis = model.encode_image(img)
print("3) features visuales:", tuple(vis.shape), "| finito:", bool(torch.isfinite(vis).all()))

tr = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
tot = sum(p.numel() for p in model.parameters()) / 1e6
enc_frozen = not any(p.requires_grad for p in model.encoder.parameters())
print(f"4) params entrenables: {tr:.0f}M / {tot:.0f}M | encoder congelado: {enc_frozen}")

with torch.no_grad():
    gen = model.generate(img, tokenizer=tok, max_new_tokens=20, num_beams=1)[0]
print("5) generación (sin entrenar → genérica):", repr(gen[:90]))

ids = tok(gen, return_tensors="pt").input_ids
maps = CrossAttentionExtractor(model).extract(img, ids, model.patch_grid)
print("6) heatmap XAI:", maps[0].heatmap.shape, "(debe ser 37,37)")
print("\n✓ M2 (RAD-DINO congelado) funciona end-to-end.")
