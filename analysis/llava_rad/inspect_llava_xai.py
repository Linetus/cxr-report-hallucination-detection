"""Inspecciona la estructura de LLaVA-Rad para adaptar el XAI: tokens visuales + atención."""
import os
os.environ["HF_HOME"] = r"./hf_cache/"
import glob, torch
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX

MERGED = r"./hf_cache/llavarad-merged"
tokenizer, model, image_processor, context_len = load_pretrained_model(MERGED, None, "llavarad", load_8bit=True)

vt = model.get_vision_tower()
print("=== Vision tower ===")
print("  clase:", type(vt).__name__)
print("  num_patches:", vt.num_patches)
print("  select_feature:", getattr(vt, "select_feature", "?"))

png = sorted(glob.glob(r"./data/raw/open-i/*.png"))[0]
image = Image.open(png).convert("RGB")
it = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].half().unsqueeze(0).cuda()
with torch.no_grad():
    feats = vt(it)
print("  features imagen (tokens visuales):", tuple(feats.shape))  # (1, N_tokens, D)

# ¿cómo queda la secuencia con el <image> expandido?
query = "<image>\nDescribe the findings of the chest x-ray.\n"
conv = conv_templates["v1"].copy(); conv.append_message(conv.roles[0], query); conv.append_message(conv.roles[1], None)
ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
print("\n=== Secuencia ===")
print("  input_ids shape:", tuple(ids.shape))
print("  posición del IMAGE_TOKEN_INDEX (-200):", (ids[0]==IMAGE_TOKEN_INDEX).nonzero().flatten().tolist())
print("\n→ Nº tokens visuales =", feats.shape[1], "; se insertan donde está el <image>.")
print("Para heatmap: self-attention de tokens de texto → esos", feats.shape[1], "tokens visuales, reshape a", int(feats.shape[1]**0.5), "x", int(feats.shape[1]**0.5))
