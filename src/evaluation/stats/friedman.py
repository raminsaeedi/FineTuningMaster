"""Friedman test — omnibus comparison across k>=3 matched methods.

Non-parametric repeated-measures test over ordinal/continuous scores. A
significant result motivates post-hoc pairwise Wilcoxon tests with Holm
correction.
"""

from __future__ import annotations

from typing import Dict, List


def friedman_test(scores_by_method: Dict[str, List[float]]) -> dict:
    """Run the Friedman test on per-item scores for each method.

    Parameters
    ----------
    scores_by_method : mapping method name -> list of per-item scores. All lists
        must be the same length and aligned by item.
    """
    methods = list(scores_by_method)
    if len(methods) < 3:
        return {
            "test": "friedman",
            "applicable": False,
            "reason": "Friedman requires >= 3 methods; use pairwise Wilcoxon instead.",
            "k": len(methods),
        }

    lengths = {len(v) for v in scores_by_method.values()}
    if len(lengths) != 1:
        raise ValueError(f"All methods must have the same number of items, got {lengths}")

    from scipy.stats import friedmanchisquare

    columns = [scores_by_method[m] for m in methods]
    stat, p = friedmanchisquare(*columns)
    return {
        "test": "friedman",
        "applicable": True,
        "statistic": float(stat),
        "p_value": float(p),
        "k": len(methods),
        "n": lengths.pop(),
        "methods": methods,
    }
