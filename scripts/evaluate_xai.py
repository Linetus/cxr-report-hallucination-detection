"""
Evaluación XAI de un modelo: anclaje clínico (NIH) + fidelidad (Cross-Attn vs IG).
Harness reutilizable para comparar M0/M1/M2/M3.

Uso:
    python scripts/evaluate_xai.py --ckpt outputs/checkpoints/m0_hilr/stage2_epoch=02_val_rouge_l=0.0000.ckpt --name m0_hilr
    python scripts/evaluate_xai.py --ckpt ... --name m1 --n-ground 200 --n-faith 30
"""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.data import OpenIDataModule
from src.data.preprocessing import CXRPreprocessor
from src.training.lightning_module import RadiologyLightningModule
from src.modules.xai import CrossAttentionExtractor, IntegratedGradientsVisualizer
from src.modules.grounding import evaluate_localization
from src.modules.faithfulness import evaluate_faithfulness

NIH_IMG = Path("data/raw/nih/images")
NIH_CSV = Path("data/raw/nih/BBox_List_2017.csv")


def load_model(ckpt, cfg, tok, device):
    lm = RadiologyLightningModule(cfg=cfg, tokenizer=tok, stage=2)
    lm.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"], strict=False)
    return lm.model.eval().to(device)


def global_attention(model, img, tok, device):
    """Mapa de atención global (mean tokens/heads/layers) 14×14 para una imagen."""
    with torch.no_grad():
        gen = model.generate(img, tokenizer=tok, max_new_tokens=35, num_beams=1)[0]
        ids = tok(gen, return_tensors="pt").input_ids.to(device)
        _ = model(pixel_values=img, input_ids=ids)
    attn = torch.stack([l.last_attn_weights.squeeze(0) for l in model.cross_attention_layers])
    return attn.mean(1).mean(0).mean(0).reshape(14, 14).cpu().numpy(), ids


def eval_grounding(model, tok, device, n_max):
    """Anclaje sobre NIH: pointing-game + energy-lift."""
    rows = {}
    with open(NIH_CSV) as f:
        r = csv.reader(f); next(r)
        for row in r:
            if row and row[0] and (NIH_IMG / row[0].strip()).exists():
                rows.setdefault(row[0].strip(), []).append(
                    (row[1].strip(), tuple(float(x) for x in row[2:6])))
    names = list(rows)[:n_max]
    pre = CXRPreprocessor(image_size=224, mode="eval")
    hits, lifts = [], []
    for name in names:
        img = pre(CXRPreprocessor.load_image(str(NIH_IMG / name))).unsqueeze(0).to(device)
        gmap, _ = global_attention(model, img, tok, device)
        for _lab, bbox in rows[name]:
            s = evaluate_localization(gmap, bbox)
            hits.append(s.pointing_hit); lifts.append(s.lift)
    return {"n_bboxes": len(hits), "pointing_hit_rate": float(np.mean(hits)),
            "energy_lift_mean": float(np.mean(lifts))}


def eval_faithfulness(model, tok, dm, device, n_imgs, ig_steps=12):
    """Fidelidad Cross-Attn vs IG sobre una muestra de Open-I val (tokens de contenido)."""
    ca_ext = CrossAttentionExtractor(model)
    ig = IntegratedGradientsVisualizer(model, n_steps=ig_steps)
    acc = {"ca": [], "ig": []}
    seen = 0
    for batch in dm.val_dataloader():
        for b in range(batch["image"].shape[0]):
            if seen >= n_imgs:
                break
            img = batch["image"][b:b + 1].to(device)
            with torch.no_grad():
                gen = model.generate(img, tokenizer=tok, max_new_tokens=35, num_beams=1)[0]
            ids = tok(gen, return_tensors="pt").input_ids.to(device)
            if ids.shape[1] < 4:
                continue
            # primer token de contenido explicable
            try:
                pos = next(i for i in range(1, ids.shape[1] - 1)
                           if tok.decode([ids[0, i + 1]]).strip().isalpha()
                           and len(tok.decode([ids[0, i + 1]]).strip()) > 4)
            except StopIteration:
                continue
            tid = int(ids[0, pos + 1])
            visual = model.encode_image(img)
            imp_ca = ca_ext.extract(img, ids, (14, 14))[pos].heatmap.flatten()
            imp_ig = ig.compute(visual, ids, pos, tid, (14, 14)).heatmap.flatten()
            acc["ca"].append(evaluate_faithfulness(model, visual, ids, pos, tid, imp_ca, n_steps=12))
            acc["ig"].append(evaluate_faithfulness(model, visual, ids, pos, tid, imp_ig, n_steps=12))
            seen += 1
        if seen >= n_imgs:
            break

    def agg(scores):
        return {k: float(np.mean([getattr(s, k) for s in scores]))
                for k in ("deletion_auc", "insertion_auc", "comprehensiveness", "sufficiency")}
    return {"n_tokens": seen, "cross_attention": agg(acc["ca"]), "integrated_gradients": agg(acc["ig"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-ground", type=int, default=150)
    ap.add_argument("--n-faith", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = OmegaConf.load("configs/base_config.yaml")
    tok = AutoTokenizer.from_pretrained(cfg.model.decoder.name)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"
    model = load_model(args.ckpt, cfg, tok, args.device)

    dm = OpenIDataModule(data_root=cfg.dataset.root, tokenizer=tok, batch_size=16, num_workers=0,
        image_size=224, max_text_length=256, splits_cache=cfg.dataset.splits_cache,
        text_target=cfg.dataset.text_target, split_ratios=dict(cfg.dataset.split_ratios))
    dm.prepare_data(); dm.setup()

    print(f"Evaluando '{args.name}' en {args.device}...")
    results = {"name": args.name, "ckpt": args.ckpt}
    print("  · anclaje clínico (NIH)...")
    results["grounding"] = eval_grounding(model, tok, args.device, args.n_ground)
    print("  · fidelidad (Cross-Attn vs IG)...")
    results["faithfulness"] = eval_faithfulness(model, tok, dm, args.device, args.n_faith)

    out = Path("outputs/results"); out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}_xai_eval.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\n✓ Guardado en outputs/results/{args.name}_xai_eval.json")


if __name__ == "__main__":
    main()
