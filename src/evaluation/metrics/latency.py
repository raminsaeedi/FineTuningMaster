"""Latency metric — average, median (p50) and p95 generation time.

Latency is recorded per prediction in ``GenerationResult.latency_ms``. Reported
in both milliseconds and seconds for convenience in the efficiency table.
"""

from __future__ import annotations

from src.core.interfaces import BaseMetric
from src.core.registry import METRICS


def _percentile(sorted_values, q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


@METRICS.register("latency")
class Latency(BaseMetric):
    name = "latency"

    def compute(self, results, references=None) -> dict:
        expected_ids = {str(ref.get("item_id", "")) for ref in (references or [])}
        measured_results = [r for r in results if not expected_ids or r.item_id in expected_ids]
        values = sorted(float(r.latency_ms) for r in measured_results if r.latency_ms is not None)
        n_requested = len(references) if references else len(measured_results)
        coverage = 100.0 * len(values) / n_requested if n_requested else None
        if not values:
            return {
                "avg_latency_ms": None, "p50_latency_ms": None, "p95_latency_ms": None,
                "avg_latency_s": None, "p50_latency_s": None, "p95_latency_s": None,
                "n": 0, "n_measured": 0, "n_requested": n_requested,
                "coverage_rate": coverage,
            }
        avg = sum(values) / len(values)
        p50 = _percentile(values, 0.50)
        p95 = _percentile(values, 0.95)
        return {
            "avg_latency_ms": round(avg, 1),
            "p50_latency_ms": round(p50, 1),
            "p95_latency_ms": round(p95, 1),
            "avg_latency_s": round(avg / 1000.0, 3),
            "p50_latency_s": round(p50 / 1000.0, 3),
            "p95_latency_s": round(p95 / 1000.0, 3),
            "n": len(values),
            "n_measured": len(values),
            "n_requested": n_requested,
            "coverage_rate": round(coverage, 2) if coverage is not None else None,
        }
