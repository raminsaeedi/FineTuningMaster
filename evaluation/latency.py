"""
evaluation/latency.py — Inference latency metrics.

Metrics:
    avg_latency_s — mean latency across all predictions
    p50_latency_s — median latency (50th percentile)
    p95_latency_s — 95th-percentile latency (tail latency)
"""

from __future__ import annotations


def compute_latency_metrics(results: list[dict]) -> dict:
    """
    Compute latency statistics over a list of prediction rows.

    Parameters
    ----------
    results : list[dict]
        Each dict must have a ``latency_s`` key (float, seconds per prediction).

    Returns
    -------
    dict with keys:
        avg_latency_s (float)
        p50_latency_s (float)
        p95_latency_s (float)

    Returns an empty dict when results is empty.
    """
    latencies = sorted(r.get("latency_s", 0.0) for r in results)
    n = len(latencies)
    if n == 0:
        return {}

    avg = sum(latencies) / n
    p50 = latencies[int(0.50 * n)]
    p95 = latencies[min(int(0.95 * n), n - 1)]

    return {
        "avg_latency_s": round(avg, 3),
        "p50_latency_s": round(p50, 3),
        "p95_latency_s": round(p95, 3),
    }
