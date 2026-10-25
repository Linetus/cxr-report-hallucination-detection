"""¿La cross-attention colapsa la salida? Forward con vs sin inyección visual."""
import sys
sys.path.insert(0, r"./")
import torch
torch.set_num_threads(4)
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from src.data import OpenIDataModule
from src.training.lightning_module import RadiologyLightningModule

CKPT = r"./outputs/checkpoints/baseline_v2/stage2_epoch=00_val_rouge_l=0.0000.ckpt"
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
img, ids, am = batch["image"][:1], batch["input_ids"][:1], batch["attention_mask"][:1]
n = int(am[0].sum())

dec = model.model.decoder
def toks_pred(logits):
    p = logits[0].argmax(-1)[:n-1]
    return p, len(set(p.tolist())), repr(tok.decode(p[:30]))

with torch.no_grad():
    # CON visual (ruta normal)
    out_v = model.model(pixel_values=img, input_ids=ids, attention_mask=am)
    pv, uv, sv = toks_pred(out_v.logits)
    # SIN visual: limpia features → los bloques saltan cross_attn (decoder.py:114)
    dec._clear_visual_features()
    out_n = dec.gpt2(input_ids=ids, attention_mask=am)
    pn, un, sn = toks_pred(out_n.logits)

print(f"CON  visual: uniq={uv:2d}  pred={sv}")
print(f"SIN  visual: uniq={un:2d}  pred={sn}")
print("\n=> Si SIN>>CON en variedad ⇒ la cross-attention colapsa la salida.")

# ¿el token de puntos es artefacto del dato? frecuencia en los targets del batch
allids = batch["input_ids"][batch["attention_mask"].bool()]
from collections import Counter
c = Counter(allids.tolist())
dot_id = pv[0].item()
print(f"\nToken predicho id={dot_id} ({repr(tok.decode([dot_id]))}) "
      f"aparece {c.get(dot_id,0)}/{len(allids)} veces en los targets del batch.")
print("Top-5 tokens reales:", [(repr(tok.decode([t])), n_) for t,n_ in c.most_common(5)])
