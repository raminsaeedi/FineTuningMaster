"""Macro-averaged F1 over primary chart-type classes.

Treats the primary chart prediction as a single-label classification against the
gold primary chart. Uses scikit-learn when available, with a pure-Python
fallback so the metric works in a minimal environment.
"""

from __future__ import annotations

from typing import List, Optional

from src.core.interfaces import BaseMetric
from src.core.registry import METRICS
from src.evaluation.metrics.base import (
    index_references,
    normalise,
    predicted_charts,
    reference_charts,
)


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
            y_pred.append(preds[0] if preds else "")
        return {"macro_f1": _macro_f1(y_true, y_pred), "n": len(y_true)}
