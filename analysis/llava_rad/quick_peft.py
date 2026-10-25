"""Prueba rápida M2-PEFT ep5: ¿usa la imagen? + gates. CPU."""
import sys, torch, statistics
sys.path.insert(0, r"./")
torch.set_num_threads(4)
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from src.data import OpenIDataModule
from src.data.preprocessing import CXRPreprocessor as P
from src.training.lightning_module import RadiologyLightningModule

CKPT = r"./outputs/checkpoints/m2_peft/stage2_epoch=05_val_rouge_l=0.0000.ckpt"
sd = torch.load(CKPT, map_location="cpu")["state_dict"]
g = [abs(torch.tanh(v).item()) for k,v in sd.items() if k.endswith("cross_attn.gate")]
print(f"GATES |tanh|: medio={statistics.mean(g):.4f} max={max(g):.4f}  (M2 sin PEFT era 0.012, M0-hiLR 0.010)")

cfg = OmegaConf.load(r"./configs/base_config.yaml"); cfg.model.encoder.type="rad_dino"
tok = AutoTokenizer.from_pretrained(cfg.model.decoder.name); tok.pad_token=tok.eos_token; tok.padding_side="right"
dm = OpenIDataModule(data_root=cfg.dataset.root, tokenizer=tok, batch_size=8, num_workers=0,
    image_size=518, max_text_length=256, splits_cache=cfg.dataset.splits_cache,
    text_target=cfg.dataset.text_target, split_ratios=dict(cfg.dataset.split_ratios),
    img_mean=list(P.RADDINO_MEAN), img_std=list(P.RADDINO_STD))
dm.prepare_data(); dm.setup()
lm = RadiologyLightningModule(cfg=cfg, tokenizer=tok, stage=2)
lm.load_state_dict(sd, strict=False); model = lm.model.eval()

batch = next(iter(dm.val_dataloader())); imgs = batch["image"]
print("\n=== Informes de 6 imágenes distintas (¿diferentes?) ===")
gens=[]
for i in range(6):
    with torch.no_grad():
        gg = model.generate(imgs[i:i+1], tokenizer=tok, max_new_tokens=40, num_beams=1)[0]
    gens.append(gg); print(f" img{i}: {gg[:95]}")
print(f"\n→ únicas: {len(set(gens))}/6  (M2 sin PEFT daba 1/6 = ciego; M0-hiLR daba 5/6)")
