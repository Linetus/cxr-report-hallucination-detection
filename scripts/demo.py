"""
Demo: imagen RX → informe clínico + heatmaps por frase.

Usage:
    python scripts/demo.py --image data/raw/open-i/images/CXR1_1_IM-0001-3001.png
    python scripts/demo.py --image path/to/xray.png --checkpoint outputs/checkpoints/best.ckpt
    python scripts/demo.py --image path/to/xray.png --method cross_attention
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import CXRPreprocessor
from src.modules.xai import ReportHeatmapAligner


def load_model(checkpoint: str | None, device: torch.device):
    """Load trained model from checkpoint or return a dummy for testing."""
    if checkpoint and Path(checkpoint).exists():
        from src.models.model import RadiologyReportModel
        model = RadiologyReportModel.load_from_checkpoint(checkpoint)
        model.eval().to(device)
        return model
    else:
        print("[demo] No checkpoint found — running in stub mode (no real inference).")
        return None


def generate_report(model, image_tensor: torch.Tensor, tokenizer, max_length: int = 200) -> str:
    """Generate report from image. Returns placeholder text in stub mode."""
    if model is None:
        return (
            "Cardiac silhouette is mildly enlarged. "
            "There is increased opacity in the right lower lobe consistent with consolidation. "
            "The costophrenic angles are sharp. "
            "No pneumothorax identified."
        )
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values=image_tensor,
            max_new_tokens=max_length,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def run_demo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo] Device: {device}")

    # Load image
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    raw_image = CXRPreprocessor.load_image(image_path)
    preprocessor = CXRPreprocessor(image_size=224, mode="eval")
    image_tensor = preprocessor(raw_image).unsqueeze(0).to(device)   # (1, 3, 224, 224)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.checkpoint, device)

    # Generate report
    print("[demo] Generating report...")
    report = generate_report(model, image_tensor, tokenizer)
    print("\n=== GENERATED REPORT ===")
    print(report)
    print("=" * 40 + "\n")

    # XAI — produce heatmaps per sentence
    if model is not None:
        print("[demo] Computing XAI heatmaps...")
        aligner = ReportHeatmapAligner(
            model=model,
            tokenizer=tokenizer,
            image_size=224,
            xai_methods=(args.method,),
        )
        sentence_maps = aligner.align(image_tensor, report)
        maps_by_sentence = sentence_maps[args.method]
    else:
        # Stub: generate random heatmaps for layout preview
        print("[demo] Stub mode — generating random heatmaps for layout preview.")
        from src.modules.xai import SentenceHeatmap
        sentences = [s.strip() for s in report.split(".") if s.strip()]
        maps_by_sentence = [
            SentenceHeatmap(
                sentence=s + ".",
                token_span=(i * 5, (i + 1) * 5),
                heatmap=np.random.rand(14, 14).astype(np.float32),
                method=args.method,
            )
            for i, s in enumerate(sentences)
        ]

    # Visualize
    n = len(maps_by_sentence)
    fig = plt.figure(figsize=(6 + 4 * n, 8))
    gs = gridspec.GridSpec(1, n + 1, figure=fig)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(raw_image, cmap="gray")
    ax_orig.set_title("Original X-Ray", fontsize=11, fontweight="bold")
    ax_orig.axis("off")

    if model is not None:
        aligner_for_overlay = aligner
    else:
        from src.modules.xai import ReportHeatmapAligner as _A
        aligner_for_overlay = _A.__new__(_A)

    for i, sh in enumerate(maps_by_sentence):
        # Resize heatmap to image size for overlay
        h, w = raw_image.shape[:2]
        heatmap_resized = cv2.resize(sh.heatmap, (w, h))
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        if raw_image.ndim == 2:
            base_bgr = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
        else:
            base_bgr = raw_image.copy()
        overlay = cv2.addWeighted(base_bgr, 0.5, heatmap_colored, 0.5, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(overlay_rgb)
        wrapped = "\n".join(sh.sentence[j:j+35] for j in range(0, len(sh.sentence), 35))
        ax.set_title(wrapped, fontsize=8, wrap=True)
        ax.axis("off")

    method_label = args.method.replace("_", " ").title()
    plt.suptitle(f"XAI: {method_label} — Sentence-Level Heatmaps", fontsize=13, y=1.02)
    plt.tight_layout()

    out_path = Path("outputs") / f"demo_{image_path.stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"[demo] Saved visualization to {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Radiology report generation demo")
    parser.add_argument("--image", required=True, help="Path to chest X-ray image")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument(
        "--method",
        default="cross_attention",
        choices=["cross_attention", "integrated_gradients"],
        help="XAI method to use",
    )
    args = parser.parse_args()
    run_demo(args)
