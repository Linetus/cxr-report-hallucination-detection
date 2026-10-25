"""Genera pares (generado M2-PEFT, real) para G-Eval manual. CPU."""
import sys, torch, json
sys.path.insert(0, r"./")
torch.set_num_threads(4)
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from src.data import OpenIDataModule
from src.data.preprocessing import CXRPreprocessor as P
from src.training.lightning_module import RadiologyLightningModule

CKPT = r"./outputs/checkpoints/m2_peft/stage2_epoch=05_val_rouge_l=0.0000.ckpt"
cfg = OmegaConf.load(r"./configs/base_config.yaml"); cfg.model.encoder.type="rad_dino"
tok = AutoTokenizer.from_pretrained(cfg.model.decoder.name); tok.pad_token=tok.eos_token; tok.padding_side="right"
dm = OpenIDataModule(data_root=cfg.dataset.root, tokenizer=tok, batch_size=18, num_workers=0,
    image_size=518, max_text_length=256, splits_cache=cfg.dataset.splits_cache,
    text_target=cfg.dataset.text_target, split_ratios=dict(cfg.dataset.split_ratios),
    img_mean=list(P.RADDINO_MEAN), img_std=list(P.RADDINO_STD))
dm.prepare_data(); dm.setup()
lm = RadiologyLightningModule(cfg=cfg, tokenizer=tok, stage=2)
lm.load_state_dict(torch.load(CKPT, map_location="cpu")["state_dict"], strict=False)
model = lm.model.eval()

batch = next(iter(dm.val_dataloader()))
imgs, refs = batch["image"], batch["report_text"]
pairs = []
for i in range(18):
    with torch.no_grad():
        g = model.generate(imgs[i:i+1], tokenizer=tok, max_new_tokens=60, num_beams=4)[0]
    pairs.append({"i": i, "ref": refs[i], "gen": g})
    print(f"\n#{i}\n[REAL]: {refs[i][:260]}\n[GEN ]: {g[:260]}", flush=True)
json.dump(pairs, open(r"~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad/pairs.json","w"))
