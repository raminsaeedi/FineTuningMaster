"""Project-wide constant values, aligned to the masterplan schema.

The required output keys drive schema-compliance scoring. The task/chart
vocabularies are derived from the enums in ``schemas`` so there is one source of
truth.
"""

from __future__ import annotations

from src.core.schemas import ChartType, TaskType

# The six top-level keys a valid DesignOutput must contain.
REQUIRED_KEYS: tuple[str, ...] = (
    "context_summary",
    "kpi_chart_mapping",
    "layout",
    "styling",
    "interactions",
    "rationales",
)

# Allowed analytical task types and chart types (the scoring vocabulary).
TASK_TYPES: tuple[str, ...] = tuple(t.value for t in TaskType)
CHART_TYPES: tuple[str, ...] = tuple(c.value for c in ChartType)

# Version tag for the Phase-1 reporting artifacts (manifest.json / metrics.json /
# eval_per_item.jsonl). Runs whose manifest lacks this field are treated as legacy.
REPORT_SCHEMA_VERSION: str = "1"
