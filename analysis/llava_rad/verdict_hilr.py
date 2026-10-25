"""Veredicto M0-hiLR: ¿se abrieron los gates y usa la imagen?"""
import sys, glob, statistics
sys.path.insert(0, r"./")
import torch
torch.set_num_threads(4)
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from src.data import OpenIDataModule
from src.training.lightning_module import RadiologyLightningModule

ck = sorted(glob.glob(r"./outputs/checkpoints/m0_hilr/stage2_epoch=*.ckpt"))
if not ck:
    print("Sin checkpoint m0_hilr aún."); sys.exit()
CKPT = ck[-1]
print("Checkpoint:", CKPT.split("\\")[-1])
sd = torch.load(CKPT, map_location="cpu")["state_dict"]

# --- Gates ---
gates = [abs(torch.tanh(v).item()) for k, v in sd.items() if k.endswith("cross_attn.gate")]
print(f"\nGATES |tanh|: medio={statistics.mean(gates):.4f} máx={max(gates):.4f} "
      f"(baseline M0 v3 era ~0.004)")

# --- Sensibilidad a la imagen ---
cfg = OmegaConf.load(r"./configs/base_config.yaml")
tok = AutoTokenizer.from_pretrained(cfg.model.decoder.name); tok.pad_token = tok.eos_token; tok.padding_side = "right"
dm = OpenIDataModule(data_root=cfg.dataset.root, tokenizer=tok, batch_size=16, num_workers=0,
    image_size=224, max_text_length=256, splits_cache=cfg.dataset.splits_cache,
    text_target=cfg.dataset.text_target, split_ratios=dict(cfg.dataset.split_ratios))
dm.prepare_data(); dm.setup()
model = RadiologyLightningModule(cfg=cfg, tokenizer=tok, stage=2)
model.load_state_dict(sd, strict=False); model = model.eval()

batch = next(iter(dm.val_dataloader()))
seen, idxs = set(), []
for i in range(len(batch["report_text"])):
    r = batch["report_text"][i][:60]
    if r not in seen:
        seen.add(r); idxs.append(i)
    if len(idxs) == 5: break
imgs = batch["image"][idxs]; refs = [batch["report_text"][i] for i in idxs]
with torch.no_grad():
    gens = model.model.generate(imgs, tokenizer=tok, max_new_tokens=45, num_beams=1)
for i, (g, r) in enumerate(zip(gens, refs)):
    print(f"\n--- img {i} ---\n[REF]: {r[:110]}\n[GEN]: {g[:140]}")
print(f"\nGeneraciones únicas: {len(set(gens))}/{len(gens)}  "
      f"(baseline era ~1/5 = ciego)")
