"""
NLP evaluation metrics for generated radiology reports.

Wraps HuggingFace evaluate + sacrebleu to compute BLEU-1/4, ROUGE-L, METEOR.
All functions accept lists of strings (decoded, not token IDs).
"""
from __future__ import annotations

import evaluate
import numpy as np


# Load metric objects once at import time (they cache downloads)
_rouge = evaluate.load("rouge")
_bleu = evaluate.load("sacrebleu")
_meteor = evaluate.load("meteor")


def compute_nlp_metrics(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """
    Compute BLEU-1, BLEU-4, ROUGE-L, and METEOR for a batch of generated reports.

    Args:
        predictions: list of generated report strings
        references:  list of ground-truth report strings (same order)

    Returns:
        dict with keys: bleu_1, bleu_4, rouge_l, meteor
    """
    assert len(predictions) == len(references), \
        f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"

    # ROUGE expects list of strings
    rouge_result = _rouge.compute(predictions=predictions, references=references)

    # sacrebleu expects references as list of lists
    refs_nested = [[r] for r in references]
    bleu_result = _bleu.compute(predictions=predictions, references=refs_nested)

    # METEOR expects list of strings
    meteor_result = _meteor.compute(predictions=predictions, references=references)

    return {
        "bleu_1": bleu_result["precisions"][0] / 100.0,
        "bleu_4": bleu_result["score"] / 100.0,
        "rouge_l": rouge_result["rougeL"],
        "meteor": meteor_result["meteor"],
    }


def compute_clinical_metrics(
    predictions: list[str],
    references: list[str],
    label_names: list[str] | None = None,
) -> dict[str, float]:
    """
    Placeholder for CheXpert-labeler based clinical coherence metrics.

    The CheXpert labeler extracts 14 binary pathology labels from free text.
    Comparing labels from generated vs reference reports gives clinical F1.

    Requires: chexpert-labeler (Java-based, separate installation).
    See: https://github.com/stanfordmlgroup/chexpert-labeler
    """
    # TODO: integrate CheXpert labeler in Phase 5
    raise NotImplementedError(
        "Clinical metrics require the CheXpert labeler. "
        "See notebooks/02_xai_analysis.ipynb for manual qualitative evaluation."
    )
