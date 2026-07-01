"""Macro-averaged F1 over primary chart-type classes, with a per-class breakdown
and a confusion matrix.

Treats the primary chart prediction as a single-label classification against the
gold primary chart. ``macro_f1`` uses scikit-learn when available (pure-Python
fallback otherwise). ``per_class_f1`` and ``confusion_matrix`` are computed in
pure Python over a **fixed label order** = the project ``ChartType`` vocabulary
plus an explicit ``"(none)"`` bucket for parse-failed / empty / missing
predictions, so invalid outputs are visible rather than silently dropped.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.core.interfaces import BaseMetric
from src.core.registry import METRICS
from src.core.schemas import ChartType
from src.evaluation.metrics.base import (
    index_references,
    normalise,
    predicted_charts,
    reference_charts,
)

NONE_LABEL = "(none)"  # bucket for parse-failed / empty / missing predictions
_LABELS: List[str] = [c.value for c in ChartType] + [NONE_LABEL]


def _macro_f1(y_true: List[str], y_pred: List[str]) -> Optional[float]:
    if not y_true:
        return None
    try:
        from sklearn.metrics import f1_score

        return round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)
    except ImportError:
        pass

    f1s: List[float] = []
    for cls in set(y_true):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return round(sum(f1s) / len(f1s), 4)


def _label(token: str) -> str:
    """Map any token to a known label, bucketing unknown/empty to ``"(none)"``."""
    return token if token in _LABELS and token else NONE_LABEL


def per_class_and_confusion(
    y_true: List[str], y_pred: List[str]
) -> Tuple[Dict[str, dict], dict]:
    """Per-class precision/recall/F1/support and a confusion matrix.

    Confusion matrix uses the fixed ``_LABELS`` order (rows = true, cols = pred).
    Per-class metrics are reported only for labels that actually occur in the
    true or predicted columns (to keep the dict compact).
    """
    idx = {lab: i for i, lab in enumerate(_LABELS)}
    size = len(_LABELS)
    matrix = [[0] * size for _ in range(size)]
    yt = [_label(t) for t in y_true]
    yp = [_label(p) for p in y_pred]
    for t, p in zip(yt, yp):
        matrix[idx[t]][idx[p]] += 1

    present = set(yt) | set(yp)
    per_class: Dict[str, dict] = {}
    for cls in _LABELS:
        if cls not in present:
            continue
        tp = sum(1 for t, p in zip(yt, yp) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(yt, yp) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(yt, yp) if t == cls and p != cls)
        support = sum(1 for t in yt if t == cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[cls] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return per_class, {"labels": list(_LABELS), "matrix": matrix}


@METRICS.register("macro_f1")
class MacroF1ChartType(BaseMetric):
    name = "macro_f1"

    def compute(self, results, references) -> dict:
        ref_by_id = index_references(references or [])
        y_true: List[str] = []
        y_pred: List[str] = []
        for r in results:
            ref = ref_by_id.get(r.item_id)
            if ref is None:
                continue
            refs = [normalise(c) for c in reference_charts(ref)]
            preds = [normalise(c) for c in predicted_charts(r)]
            if not refs:
                continue
            y_true.append(refs[0])
            y_pred.append(preds[0] if preds else NONE_LABEL)
        per_class, confusion = per_class_and_confusion(y_true, y_pred)
        return {
            "macro_f1": _macro_f1(y_true, y_pred),
            "per_class_f1": per_class,
            "confusion_matrix": confusion,
            "n": len(y_true),
        }
