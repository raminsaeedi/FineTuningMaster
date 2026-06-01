"""Cliff's delta — non-parametric effect size for two samples.

delta = (#(x_i > y_j) - #(x_i < y_j)) / (n_x * n_y), ranging in [-1, 1].
Magnitude thresholds follow Romano et al. (2006).
"""

from __future__ import annotations

from typing import List, Tuple


def _magnitude(delta: float) -> str:
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    if d < 0.330:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def cliffs_delta(x: List[float], y: List[float]) -> Tuple[float, str]:
    """Return ``(delta, magnitude_label)`` for samples ``x`` vs ``y``."""
    if not x or not y:
        return 0.0, "negligible"
    greater = less = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                greater += 1
            elif xi < yj:
                less += 1
    delta = (greater - less) / (len(x) * len(y))
    return round(delta, 4), _magnitude(delta)
