"""Pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction.

The Wilcoxon signed-rank test compares two matched methods on per-item scores.
When several pairs are tested, raw p-values are adjusted with the step-down Holm
procedure to control the family-wise error rate.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple


def holm_correction(p_values: List[float]) -> List[float]:
    """Return Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p_values[idx]
        running_max = max(running_max, adj)  # enforce monotonicity
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def pairwise_wilcoxon(
    scores_by_method: Dict[str, List[float]],
    alpha: float = 0.05,
) -> List[dict]:
    """Wilcoxon signed-rank for every method pair, with Holm-adjusted p-values."""
    from scipy.stats import wilcoxon

    methods = list(scores_by_method)
    pairs: List[Tuple[str, str]] = list(combinations(methods, 2))

    raw_results = []
    for a, b in pairs:
        xa, xb = scores_by_method[a], scores_by_method[b]
        diffs = [u - v for u, v in zip(xa, xb)]
        if all(d == 0 for d in diffs):
            stat, p = 0.0, 1.0  # identical — no evidence of difference
        else:
            try:
                stat, p = wilcoxon(xa, xb, zero_method="wilcox", correction=False)
                stat, p = float(stat), float(p)
            except ValueError:
                stat, p = 0.0, 1.0
        raw_results.append({"method_a": a, "method_b": b, "statistic": stat, "p_value": p})

    adjusted = holm_correction([r["p_value"] for r in raw_results])
    for r, p_adj in zip(raw_results, adjusted):
        r["p_holm"] = p_adj
        r["reject_h0"] = bool(p_adj < alpha)
    return raw_results
