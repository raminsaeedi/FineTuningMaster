"""
evaluation/chart.py — Chart-type classification metrics.

Metrics:
    chart_top1_accuracy — % of examples where the primary predicted chart
                          type matches the primary reference chart type
    chart_top3_accuracy — % of examples where the reference primary chart
                          type appears in the top-3 predictions
    chart_macro_f1      — macro-averaged F1 over chart type classes
                          (uses scikit-learn if available; falls back to
                           a pure-Python implementation)

Only computed for examples where chart_types_reference is non-empty.
Returns None fields when no reference labels exist.
"""

from __future__ import annotations


def _normalise(s: str) -> str:
    """Lowercase + strip for robust chart-type comparison."""
    return s.lower().strip()


def _macro_f1(y_true: list[str], y_pred: list[str]) -> float | None:
    """Macro-averaged F1 using sklearn when available; pure-Python fallback."""
    try:
        from sklearn.metrics import f1_score
        return round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)
    except ImportError:
        pass

    # Pure-Python fallback
    classes = set(y_true)
    if not classes:
        return None
    f1s: list[float] = []
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return round(sum(f1s) / len(f1s), 4)


def compute_chart_metrics(results: list[dict]) -> dict:
    """
    Compute chart-type accuracy and macro-F1 over prediction rows.

    Parameters
    ----------
    results : list[dict]
        Each dict must have:
            chart_types_predicted  — list[str]  (predicted chart type names)
            chart_types_reference  — list[str]  (ground-truth chart type names)

    Returns
    -------
    dict with keys:
        chart_top1_accuracy (float 0–100 or None)
        chart_top3_accuracy (float 0–100 or None)
        chart_macro_f1      (float 0–1  or None)
    """
    labelled = [
        r for r in results
        if r.get("chart_types_reference") and r.get("chart_types_predicted") is not None
    ]

    if not labelled:
        return {
            "chart_top1_accuracy": None,
            "chart_top3_accuracy": None,
            "chart_macro_f1":      None,
        }

    top1_hits = 0
    top3_hits = 0
    all_true:  list[str] = []
    all_pred:  list[str] = []

    for r in labelled:
        refs  = [_normalise(c) for c in r["chart_types_reference"]]
        preds = [_normalise(c) for c in r["chart_types_predicted"]]

        primary_ref = refs[0] if refs else ""
        top1_hits  += 1 if (preds and preds[0] == primary_ref) else 0
        top3_hits  += 1 if any(p == primary_ref for p in preds[:3]) else 0

        all_true.append(primary_ref)
        all_pred.append(preds[0] if preds else "")

    n    = len(labelled)
    top1 = round(100.0 * top1_hits / n, 2)
    top3 = round(100.0 * top3_hits / n, 2)

    return {
        "chart_top1_accuracy": top1,
        "chart_top3_accuracy": top3,
        "chart_macro_f1":      _macro_f1(all_true, all_pred),
    }
