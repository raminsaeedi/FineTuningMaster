"""Cochran's Q and McNemar tests for matched binary outcomes.

Used for dichotomous metrics such as Top-1 correct (1) / incorrect (0):
- Cochran's Q is the omnibus test across k>=3 methods;
- the exact McNemar test compares two methods on the discordant pairs, with Holm
  correction across all pairs.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List

from src.evaluation.stats.wilcoxon_holm import holm_correction


def cochran_q(outcomes_by_method: Dict[str, List[int]]) -> dict:
    """Cochran's Q test over matched binary outcomes for k>=3 methods."""
    methods = list(outcomes_by_method)
    if len(methods) < 3:
        return {
            "test": "cochran_q",
            "applicable": False,
            "reason": "Cochran's Q requires >= 3 methods; use McNemar for pairs.",
            "k": len(methods),
        }

    from scipy.stats import chi2

    k = len(methods)
    columns = [outcomes_by_method[m] for m in methods]
    n = len(columns[0])
    if any(len(c) != n for c in columns):
        raise ValueError("All methods must have the same number of items")

    col_totals = [sum(c) for c in columns]          # G_j
    row_totals = [sum(col[i] for col in columns) for i in range(n)]  # L_i
    grand = sum(col_totals)

    sum_col_sq = sum(g * g for g in col_totals)
    sum_row = sum(row_totals)
    sum_row_sq = sum(l * l for l in row_totals)
    denom = (k * sum_row - sum_row_sq)
    if denom == 0:
        return {"test": "cochran_q", "applicable": True, "statistic": 0.0, "p_value": 1.0, "df": k - 1, "k": k, "n": n}

    q = (k - 1) * (k * sum_col_sq - grand * grand) / denom
    df = k - 1
    p = float(chi2.sf(q, df))
    return {
        "test": "cochran_q",
        "applicable": True,
        "statistic": float(q),
        "p_value": p,
        "df": df,
        "k": k,
        "n": n,
        "methods": methods,
    }


def mcnemar_test(x: List[int], y: List[int]) -> dict:
    """Exact McNemar test on the discordant pairs of two binary methods."""
    from scipy.stats import binomtest

    b = sum(1 for xi, yi in zip(x, y) if xi == 1 and yi == 0)
    c = sum(1 for xi, yi in zip(x, y) if xi == 0 and yi == 1)
    n_disc = b + c
    if n_disc == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0}
    p = float(binomtest(b, n_disc, 0.5).pvalue)
    return {"b": b, "c": c, "n_discordant": n_disc, "p_value": p}


def pairwise_mcnemar(
    outcomes_by_method: Dict[str, List[int]],
    alpha: float = 0.05,
) -> List[dict]:
    """Exact McNemar for every method pair, with Holm-adjusted p-values."""
    methods = list(outcomes_by_method)
    results = []
    for a, b in combinations(methods, 2):
        res = mcnemar_test(outcomes_by_method[a], outcomes_by_method[b])
        results.append({"method_a": a, "method_b": b, **res})

    adjusted = holm_correction([r["p_value"] for r in results])
    for r, p_adj in zip(results, adjusted):
        r["p_holm"] = p_adj
        r["reject_h0"] = bool(p_adj < alpha)
    return results
