"""Paso B: cargar el modelo fusionado en 8-bit + generar informe. GPU."""
import os
os.environ["HF_HOME"] = r"./hf_cache/"
import glob, torch, time
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX

MERGED = r"./hf_cache/llavarad-merged"
t0 = time.time()
print("Cargando modelo fusionado en 8-bit...", flush=True)
tokenizer, model, image_processor, context_len = load_pretrained_model(MERGED, None, "llavarad", load_8bit=True)
print(f"Cargado en {time.time()-t0:.0f}s | VRAM {round(torch.cuda.memory_allocated()/1e9,1)} GB", flush=True)

png = sorted(glob.glob(r"./data/raw/open-i/*.png"))[0]
image = Image.open(png).convert("RGB")
image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].half().unsqueeze(0).cuda()

query = "<image>\nDescribe the findings of the chest x-ray.\n"
conv = conv_templates["v1"].copy()
conv.append_message(conv.roles[0], query); conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
stop = KeywordsStoppingCriteria(["</s>"], tokenizer, input_ids)

t1 = time.time()
with torch.inference_mode():
    out = model.generate(input_ids, images=image_tensor, do_sample=False, temperature=0.0,
                         max_new_tokens=256, use_cache=True, stopping_criteria=[stop])
txt = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
print(f"\n=== INFORME LLaVA-Rad ({time.time()-t1:.0f}s) ===\n{txt}\n")
print("VRAM pico:", round(torch.cuda.max_memory_allocated()/1e9,1), "GB")
print("OK")
