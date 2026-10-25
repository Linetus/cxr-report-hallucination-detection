"""Extracción de heatmap de LLaVA-Rad: self-attention texto→tokens-visuales. GPU 8-bit."""
import os
os.environ["HF_HOME"] = r"./hf_cache/"
import glob, torch, numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX

MERGED = r"./hf_cache/llavarad-merged"
OUT = r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad/llava_hm.png"
NVIS = 1369; GRID = 37
tokenizer, model, image_processor, context_len = load_pretrained_model(MERGED, None, "llavarad", load_8bit=True)

png = sorted(glob.glob(r"./data/raw/open-i/*.png"))[0]
image = Image.open(png).convert("RGB")
it = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].half().unsqueeze(0).cuda()

query = "<image>\nDescribe the findings of the chest x-ray.\n"
conv = conv_templates["v1"].copy(); conv.append_message(conv.roles[0], query); conv.append_message(conv.roles[1], None)
prompt_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
img_pos = (prompt_ids[0]==IMAGE_TOKEN_INDEX).nonzero().item()   # 35

# 1) generar
stop = KeywordsStoppingCriteria(["</s>"], tokenizer, prompt_ids)
with torch.inference_mode():
    out = model.generate(prompt_ids, images=it, do_sample=False, max_new_tokens=40, use_cache=True, stopping_criteria=[stop])
gen_ids = out[0, prompt_ids.shape[1]:]
report = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
print("INFORME:", report[:150])

# 2) forward completo con atenciones (secuencia = prompt + generado)
full = torch.cat([prompt_ids, gen_ids.unsqueeze(0)], dim=1)
n_gen = gen_ids.shape[0]
with torch.inference_mode():
    o = model(input_ids=full, images=it, output_attentions=True, return_dict=True, use_cache=False)
atts = o.attentions  # tupla de capas, cada (1, heads, seq, seq) en la secuencia EXPANDIDA
seq = atts[0].shape[-1]
vis_range = slice(img_pos, img_pos+NVIS)     # 1369 tokens visuales
gen_range = slice(seq-n_gen, seq)            # tokens generados al final
# mean sobre capas y heads; atención de generado → visuales
attn = torch.stack([a[0].float().mean(0) for a in atts]).mean(0)   # (seq, seq)
hm = attn[gen_range, vis_range].mean(0).cpu().numpy()              # (1369,)
hm = hm.reshape(GRID, GRID); hm = (hm-hm.min())/(hm.max()-hm.min()+1e-8)
conc = hm.max()/(hm.mean()+1e-8)
print(f"heatmap {hm.shape} concentración={conc:.1f} seq={seq}")

# 3) overlay
disp = np.array(image.resize((224,224)).convert("L"))
hm_up = np.kron(hm, np.ones((224//GRID, 224//GRID)))[:224,:224]
fig, ax = plt.subplots(1,2, figsize=(9,4.5))
ax[0].imshow(disp, cmap="gray"); ax[0].set_title("RX"); ax[0].axis("off")
ax[1].imshow(disp, cmap="gray"); ax[1].imshow(hm_up, cmap="jet", alpha=0.5)
ax[1].set_title(report[:45], fontsize=8); ax[1].axis("off")
plt.tight_layout(); plt.savefig(OUT, dpi=90, bbox_inches="tight")
print("Figura:", OUT)
