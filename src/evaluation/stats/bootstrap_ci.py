"""Percentile bootstrap confidence intervals.

Reports an effect together with its uncertainty. ``bootstrap_ci`` gives a CI for
a single sample statistic; ``paired_bootstrap_diff`` gives a CI for the mean
paired difference between two methods (resampling items, preserving pairing).
The default of 10,000 resamples follows the thesis protocol.
"""

from __future__ import annotations

from typing import Callable, List, Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap CI for ``statistic`` of ``values``."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"point": None, "ci_low": None, "ci_high": None, "n": 0, "n_boot": n_boot}

    rng = np.random.default_rng(seed)
    n = arr.size
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, n)]
        boot[i] = statistic(sample)

    lo_q = (1 - ci) / 2 * 100
    hi_q = (1 + ci) / 2 * 100
    return {
        "point": float(statistic(arr)),
        "ci_low": float(np.percentile(boot, lo_q)),
        "ci_high": float(np.percentile(boot, hi_q)),
        "ci_level": ci,
        "n": int(n),
        "n_boot": n_boot,
    }


def paired_bootstrap_diff(
    x: Sequence[float],
    y: Sequence[float],
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for the mean paired difference ``x - y`` (items resampled)."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.shape != ya.shape or xa.size == 0:
        raise ValueError("x and y must be non-empty and the same length (paired)")

    diffs = xa - ya
    rng = np.random.default_rng(seed)
    n = diffs.size
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot[i] = diffs[rng.integers(0, n, n)].mean()

    lo_q = (1 - ci) / 2 * 100
    hi_q = (1 + ci) / 2 * 100
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(boot, lo_q)),
        "ci_high": float(np.percentile(boot, hi_q)),
        "ci_level": ci,
        "n": int(n),
        "n_boot": n_boot,
    }
