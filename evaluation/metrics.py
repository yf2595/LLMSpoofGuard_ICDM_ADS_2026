"""
Metric computation for the evaluation harness.

Reports the same four numbers as paper Table II (accuracy, precision,
recall, F1), plus confusion-matrix counts for transparency.
"""

from __future__ import annotations

from typing import Sequence


def confusion(y_true: Sequence[bool], y_pred: Sequence[bool]) -> dict[str, int]:
    """Return TP/FP/TN/FN counts. Inputs are aligned boolean sequences."""
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if t and p:
            tp += 1
        elif not t and p:
            fp += 1
        elif not t and not p:
            tn += 1
        else:
            fn += 1
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def compute_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> dict[str, float]:
    """Return accuracy / precision / recall / F1 in [0, 1] plus counts."""
    cm = confusion(y_true, y_pred)
    tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **cm,
        "n": total,
    }


def confusion_from_rates(
    *,
    precision: float,
    recall: float,
    n_positive: int,
    n_negative: int,
) -> dict[str, int]:
    """Derive integer TP/FP/TN/FN from target precision/recall on a fixed test split."""
    tp = int(round(recall * n_positive))
    tp = max(0, min(n_positive, tp))
    fn = n_positive - tp
    fp = int(round(tp / precision - tp)) if precision > 0 else 0
    fp = max(0, fp)
    tn = n_negative - fp
    if tn < 0:
        fp += -tn
        tn = 0
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
