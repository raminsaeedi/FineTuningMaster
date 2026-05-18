"""
evaluation/schema.py — Schema-compliance metrics.

Metrics:
    json_parse_rate      — % of outputs that are parseable JSON
    schema_validity_rate — % of outputs that pass the required-fields check
    completeness_score   — mean fraction of required fields present (0–1)
    field_coverage       — per-field presence rate dict
"""

from __future__ import annotations

REQUIRED_FIELDS = [
    "context_summary",
    "kpi_task_chart_mapping",
    "layout_hierarchy",
    "labels_scales_colors",
    "interactions",
    "design_rationales",
]


def compute_schema_metrics(results: list[dict]) -> dict:
    """
    Compute schema-compliance metrics over a list of prediction rows.

    Parameters
    ----------
    results : list[dict]
        Each dict is one row from a predictions/*.jsonl file and must have
        the keys: valid_json, valid_schema, completeness, parsed.

    Returns
    -------
    dict with keys:
        json_parse_rate      (float, 0–100)
        schema_validity_rate (float, 0–100)
        completeness_score   (float, 0–1)
        field_coverage       (dict[str, float])  — one entry per required field
    """
    n = len(results)
    if n == 0:
        return {}

    json_ok          = sum(1 for r in results if r.get("valid_json"))
    schema_ok        = sum(1 for r in results if r.get("valid_schema"))
    completeness_sum = sum(r.get("completeness", 0.0) for r in results)

    field_counts: dict[str, int] = {f: 0 for f in REQUIRED_FIELDS}
    for r in results:
        parsed = r.get("parsed") or {}
        for field in REQUIRED_FIELDS:
            if field in parsed:
                field_counts[field] += 1

    return {
        "json_parse_rate":      round(100.0 * json_ok / n, 2),
        "schema_validity_rate": round(100.0 * schema_ok / n, 2),
        "completeness_score":   round(completeness_sum / n, 4),
        "field_coverage":       {f: round(c / n, 4) for f, c in field_counts.items()},
    }
