"""Paired effect sizes for matched-pairs comparisons.

Cliff's delta (see ``cliff_delta.py``) treats the two samples as *independent*
and ignores the pairing, so it is not the right effect size for our matched-item
design. These two are paired:

  matched-pairs rank-biserial correlation — the effect size that pairs with the
      Wilcoxon signed-rank test; r = (R+ − R−) / (R+ + R−) over signed ranks of
      the non-zero differences. Range [−1, 1].
  Cohen's d_z — standardised mean of the paired differences,
      d_z = mean(x − y) / sd(x − y). Range unbounded.

Both operate on aligned (paired) vectors ``x`` and ``y`` of equal length.
"""

from __future__ import annotations

import statistics
from typing import List, Sequence, Tuple


def _average_ranks(values: Sequence[float]) -> List[float]:
    """Ranks with ties resolved by averaging (1-based)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # average of 1-based ranks i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def paired_rank_biserial(x: Sequence[float], y: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation for ``x`` vs ``y``.

    Positive => x tends to exceed y. Returns 0.0 when all pairs are tied.
    """
    diffs = [a - b for a, b in zip(x, y)]
    nonzero = [d for d in diffs if d != 0]
    if not nonzero:
        return 0.0
    ranks = _average_ranks([abs(d) for d in nonzero])
    r_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    r_minus = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    total = r_plus + r_minus
    return round((r_plus - r_minus) / total, 4) if total else 0.0


def cohen_dz(x: Sequence[float], y: Sequence[float]) -> float:
    """Cohen's d_z: mean paired difference / sd of paired differences."""
    diffs = [a - b for a, b in zip(x, y)]
    if len(diffs) < 2:
        return 0.0
    sd = statistics.stdev(diffs)
    if sd == 0:
        return 0.0
    return round(statistics.mean(diffs) / sd, 4)


def paired_effect_size(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """Convenience: ``(rank_biserial, cohen_dz)`` for ``x`` vs ``y``."""
    return paired_rank_biserial(x, y), cohen_dz(x, y)
