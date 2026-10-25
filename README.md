# When can you trust an AI-generated radiology report?

Master's thesis (MSc in Artificial Intelligence, Universidad Alfonso X el Sabio, 2026) by **Daniel Gómez Arranz**.

**Question:** can a multimodal model's visual explanation, or its own uncertainty, tell us when a generated chest X-ray report contains errors?

**Answer, in the models and datasets evaluated:** the model's attention looks at the right places, but the shape of that explanation does not predict errors. The model's own **decision uncertainty** does: it detects hallucinated findings well, transfers to a second model and a second hospital population, and costs nothing extra at inference.

📄 Full thesis (Spanish, with English abstract): [`docs/memoria_TFM.pdf`](docs/memoria_TFM.pdf)

## Key results

| Finding | Evidence |
|---|---|
| Attention is clinically grounded | Lift 3.03 over chance against 984 radiologist bounding boxes (NIH ChestX-ray8) |
| Explanation shape does **not** predict reliability | AUC 0.54 (not significant); confirmed with the model's own trained grounding boxes |
| Uncertainty detects hallucinations | Minimum content-token probability: **AUC 0.90** (LLaVA-Rad, Open-I) |
| It transfers across models and domains | AUC 0.80–0.81 on MAIRA-2 and on PadChest-GR (Spain), except a weak generator out of distribution (0.61) |
| Excluding the report's opening token matters | AUC rises from 0.71 to 0.90: that token carries style uncertainty, not content uncertainty |
| It catches fabrications, not omissions | Omissions are often confident errors; the signal only catches part of them |

The calibrated signal becomes a per-report confidence "traffic light" (risk bands) shown in an interactive demo next to the report and its grounding.

**Limitations:** few hallucinations per cell (12–110), so confidence intervals are wide; MAIRA-2 was trained on PadChest, so its in-distribution results may overlap with training data; models ran 8-bit quantized on a single 12 GB GPU; and the clinical benefit was not tested with radiologists. The thesis discusses these in detail.

## Study design

A 2×2 design crosses two state-of-the-art 7B report generators, **LLaVA-Rad** and **MAIRA-2**, with two populations, **Open-I** (United States) and **PadChest-GR** (Spain). Each report is labelled correct, hallucination or omission using three independent sources of truth: structured diagnoses (MeSH), radiologist present/absent findings, and an LLM judge (G-Eval style) over the full narrative. Statistics use bootstrap confidence intervals and permutation tests, with explicit control of confounders such as report length and token position.

## Repository layout

```
analysis/                 main study (research scripts, mostly exploratory)
  llava_rad/              signal extraction and analysis for LLaVA-Rad, NIH grounding, calibration
  openi_maira2/           MAIRA-2 on Open-I (out of distribution)
  padchest_maira2/        MAIRA-2 on PadChest-GR, G-Eval labelling, grounding boxes
  final_n800.py           final Open-I analysis (n = 800)
demo/                     Gradio demos: report + grounding + calibrated confidence
src/, scripts/, configs/  first phase: own encoder–decoder generator (ViT/RAD-DINO + GPT-2
                          with gated cross-attention, knowledge distillation, PEFT)
docs/                     thesis PDF and LaTeX sources (figures not included)
```

The `analysis/` folder is the actual research trail, including approaches that turned out to be artefacts. It is kept as it was used rather than refactored into a library.

## Data and models

No images, datasets, model weights or derived features are included. To reproduce, obtain them from their sources and accept their licences:

- [Open-I / Indiana University chest X-rays](https://openi.nlm.nih.gov)
- [NIH ChestX-ray8](https://nihcc.app.box.com/v/ChestXray-NIHCC) (bounding-box subset)
- [PadChest-GR](https://bimcv.cipf.es/bimcv-projects/padchest-gr/) (requires an access request; images may not be redistributed)
- [LLaVA-Rad](https://github.com/microsoft/LLaVA-Rad) and [MAIRA-2](https://huggingface.co/microsoft/maira-2) (accept the model licence on Hugging Face)

Scripts expect data under `./data` and the Hugging Face cache under `./hf_cache`; adjust paths or set `HF_HOME` / `LLAVARAD_MODEL_PATH` as needed. Environments: `environment.yml` / `requirements.txt` for the own generator; LLaVA-Rad and MAIRA-2 each need their own environment (see `demo/README.md`).

## Note

Research code written for a thesis. Comments, file names and the thesis itself are mostly in Spanish.
