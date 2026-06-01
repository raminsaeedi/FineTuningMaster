"""Krippendorff's alpha (inter-rater reliability).

Implements the coincidence-matrix formulation (Krippendorff 2011) with nominal,
ordinal and interval distance metrics. Ordinal is the right choice for Likert
rubric scores. Returns ``None`` when there is no ratable data.

Reference value: alpha = 1.0 for perfect agreement; ~0 for chance-level; < 0 for
systematic disagreement.
"""

from __future__ import annotations

import collections
from typing import Dict, List, Optional


def krippendorff_alpha(units: List[List[float]], level: str = "ordinal") -> Optional[float]:
    """Compute Krippendorff's alpha from per-unit rating lists.

    Parameters
    ----------
    units : list of lists; each inner list holds the values assigned to one unit
        by the raters who rated it (missing ratings simply omitted).
    level : "nominal" | "ordinal" | "interval".
    """
    # Only units rated by >= 2 raters contribute.
    units = [u for u in units if len(u) >= 2]
    if not units:
        return None

    values = sorted({v for u in units for v in u})
    if len(values) < 2:
        return 1.0  # everyone used the same single value -> perfect agreement

    # Coincidence matrix o[(c,k)].
    o: Dict[tuple, float] = collections.defaultdict(float)
    for u in units:
        m = len(u)
        cnt = collections.Counter(u)
        for c in values:
            for k in values:
                pairs = cnt[c] * (cnt[c] - 1) if c == k else cnt[c] * cnt[k]
                if pairs:
                    o[(c, k)] += pairs / (m - 1)

    n_c = {c: sum(o[(c, k)] for k in values) for c in values}
    n = sum(n_c.values())
    if n == 0:
        return None

    def delta2(c: float, k: float) -> float:
        if level == "nominal":
            return 0.0 if c == k else 1.0
        if level == "interval":
            return float((c - k) ** 2)
        # ordinal
        lo, hi = (c, k) if c <= k else (k, c)
        between = [g for g in values if lo <= g <= hi]
        s = sum(n_c[g] for g in between) - (n_c[lo] + n_c[hi]) / 2.0
        return float(s * s)

    do = sum(o[(c, k)] * delta2(c, k) for c in values for k in values) / n
    de = sum(n_c[c] * n_c[k] * delta2(c, k) for c in values for k in values) / (n * (n - 1))
    if de == 0:
        return 1.0
    return round(1.0 - do / de, 4)
