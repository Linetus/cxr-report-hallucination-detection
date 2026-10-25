"""De-riesgo XAI: ¿localizan los heatmaps de cross-attention en M0-hiLR?"""
import sys, glob
sys.path.insert(0, r"./")
import numpy as np, torch
torch.set_num_threads(4)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from src.data import OpenIDataModule
from src.training.lightning_module import RadiologyLightningModule

CKPT = sorted(glob.glob(r"./outputs/checkpoints/m0_hilr/stage2_epoch=*.ckpt"))[-1]
OUT = r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad/xai_overlay.png"
print("Checkpoint:", CKPT.split("\\")[-1])

cfg = OmegaConf.load(r"./configs/base_config.yaml")
tok = AutoTokenizer.from_pretrained(cfg.model.decoder.name); tok.pad_token = tok.eos_token; tok.padding_side = "right"
dm = OpenIDataModule(data_root=cfg.dataset.root, tokenizer=tok, batch_size=16, num_workers=0,
    image_size=224, max_text_length=256, splits_cache=cfg.dataset.splits_cache,
    text_target=cfg.dataset.text_target, split_ratios=dict(cfg.dataset.split_ratios))
dm.prepare_data(); dm.setup()
model = RadiologyLightningModule(cfg=cfg, tokenizer=tok, stage=2)
model.load_state_dict(torch.load(CKPT, map_location="cpu")["state_dict"], strict=False)
model = model.eval()

batch = next(iter(dm.val_dataloader()))
img = batch["image"][3:4]  # la de "hyperaerated" antes

# 1) generar
with torch.no_grad():
    gen = model.model.generate(img, tokenizer=tok, max_new_tokens=45, num_beams=1)[0]
print("GEN:", gen[:200])
gen_ids = tok(gen, return_tensors="pt").input_ids
toks = [tok.decode([i]) for i in gen_ids[0]]

# 2) forward para poblar last_attn_weights de cada capa cross-attn
with torch.no_grad():
    _ = model.model(pixel_values=img, input_ids=gen_ids)
layers = model.model.decoder.cross_attention_layers
attn = torch.stack([l.last_attn_weights.squeeze(0) for l in layers])  # (L,H,T,S)
attn = attn.mean(1).mean(0)  # mean heads, mean layers -> (T,S)
T, S = attn.shape
g = int(round(S ** 0.5))
print(f"attn (T,S)=({T},{S}) grid={g}x{g}")

# 3) heatmap por token -> agrupar por frase (aprox por nº de palabras)
import re
sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", gen) if s.strip()][:3]
disp = img[0,0].numpy(); disp = (disp - disp.min())/(disp.max()-disp.min()+1e-8)

fig, axes = plt.subplots(1, len(sents)+1, figsize=(4*(len(sents)+1), 4))
axes[0].imshow(disp, cmap="gray"); axes[0].set_title("RX (input)"); axes[0].axis("off")
cursor = 1
for ax, sent in zip(axes[1:], sents):
    n = max(1, len(sent.split()))
    span = attn[cursor:min(cursor+n, T)]  # (n,S)
    cursor += n
    hm = span.mean(0).reshape(g, g).numpy()
    hm = (hm - hm.min())/(hm.max()-hm.min()+1e-8)
    hm_up = np.kron(hm, np.ones((224//g, 224//g)))
    ax.imshow(disp, cmap="gray"); ax.imshow(hm_up, cmap="jet", alpha=0.5)
    ax.set_title(sent[:40], fontsize=8); ax.axis("off")
plt.tight_layout(); plt.savefig(OUT, dpi=90, bbox_inches="tight")
print("Figura:", OUT)
# stats de concentración: ¿pico localizado o difuso?
peak = attn.mean(0).reshape(g,g).numpy()
print(f"Concentración atención (max/mean sobre patches): {peak.max()/peak.mean():.2f}  (>~2 = localizado, ~1 = difuso)")
