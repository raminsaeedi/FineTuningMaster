"""Shared helpers for metric implementations.

Predictions arrive as ``GenerationResult`` objects; references as a list of
dicts ``{"item_id", "recommendation"}``. These helpers extract chart-type tokens
from the typed ``kpi_chart_mapping`` and join predictions to references by id.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from src.core.schemas import GenerationResult


def chart_token(value: Any) -> str:
    """Normalise a chart_type value (enum or string) to its lowercase token."""
    if isinstance(value, Enum):
        value = value.value
    return str(value).lower().strip()


def normalise(s: Any) -> str:
    return chart_token(s)


def predicted_charts(result: GenerationResult) -> List[str]:
    """Recommended chart tokens from a prediction, in mapping order."""
    if result.parsed is None:
        return []
    return [chart_token(m.chart_type) for m in result.parsed.kpi_chart_mapping if m.chart_type is not None]


def predicted_alternatives(result: GenerationResult, index: int = 0) -> List[str]:
    """Alternative chart tokens for the mapping at ``index`` (default first)."""
    if result.parsed is None or len(result.parsed.kpi_chart_mapping) <= index:
        return []
    return [chart_token(a) for a in result.parsed.kpi_chart_mapping[index].alternatives]


def reference_charts(reference: dict) -> List[str]:
    """Recommended chart tokens from a gold recommendation dict, in order."""
    reco = reference.get("recommendation", {}) or {}
    mapping = reco.get("kpi_chart_mapping", []) or []
    out: List[str] = []
    for entry in mapping:
        if isinstance(entry, dict) and entry.get("chart_type"):
            out.append(chart_token(entry["chart_type"]))
    return out


def index_references(references: List[dict]) -> Dict[str, dict]:
    """Map references by item_id for order-independent joins."""
    return {r.get("item_id", ""): r for r in references}
