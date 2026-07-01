"""Percentile bootstrap confidence intervals.

Reports an effect together with its uncertainty. ``bootstrap_ci`` gives a CI for
a single sample statistic; ``paired_bootstrap_diff`` gives a CI for the mean
paired difference between two methods (resampling items, preserving pairing).
The default of 10,000 resamples follows the thesis protocol.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

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


def per_method_bootstrap_cis(
    vectors_by_method: Dict[str, Sequence[float]],
    scale: float = 1.0,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, dict]:
    """Per-method bootstrap CI of the mean over each method's per-item vector.

    ``scale`` multiplies ``point``/``ci_low``/``ci_high`` (e.g. 100 to turn a 0/1
    accuracy mean into a percentage). Returns ``{method: {point, ci_low, ci_high, n}}``.
    """
    out: Dict[str, dict] = {}
    for method, values in vectors_by_method.items():
        ci_res = bootstrap_ci(values, n_boot=n_boot, ci=ci, seed=seed)

        def _s(v):
            return round(v * scale, 4) if v is not None else None

        out[method] = {
            "point": _s(ci_res["point"]),
            "ci_low": _s(ci_res["ci_low"]),
            "ci_high": _s(ci_res["ci_high"]),
            "ci_level": ci,
            "n": ci_res["n"],
        }
    return out
