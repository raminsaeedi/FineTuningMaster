"""
evaluation — Metric library for the dashboard fine-tuning thesis.

Provides four metric groups, each in its own module:

    evaluation.schema     — json_parse_rate, schema_validity_rate,
                            completeness_score, field_coverage
    evaluation.chart      — chart top-1/top-3 accuracy, macro-F1
    evaluation.robustness — paraphrase_consistency, missing_info_validity_rate
    evaluation.latency    — avg, p50, p95 latency

Usage:
    from evaluation.schema     import compute_schema_metrics
    from evaluation.chart      import compute_chart_metrics
    from evaluation.robustness import compute_robustness_metrics
    from evaluation.latency    import compute_latency_metrics
"""

from evaluation.chart      import compute_chart_metrics
from evaluation.latency    import compute_latency_metrics
from evaluation.robustness import compute_robustness_metrics
from evaluation.schema     import compute_schema_metrics

__all__ = [
    "compute_schema_metrics",
    "compute_chart_metrics",
    "compute_robustness_metrics",
    "compute_latency_metrics",
]
