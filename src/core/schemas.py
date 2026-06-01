"""Pydantic data contract — aligned to the masterplan (Teil 3).

This is the abstract schema from the thesis plan: a brief described by users,
goals, KPIs and data columns; a structured design output with typed task/chart
enums and separated layout/styling/interactions/rationales. Model outputs are
parsed into these types; gold data is generated in the same shape.

Fields use ``extra="allow"`` so a slightly-extended model output still parses,
but the declared field set and the TaskType/ChartType vocabularies are the
contract used for scoring.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskType(str, Enum):
    TREND = "trend"
    COMPARISON = "comparison"
    COMPOSITION = "composition"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    RANKING = "ranking"
    DEVIATION = "deviation"
    PART_TO_WHOLE = "part_to_whole"
    FLOW = "flow"


class ChartType(str, Enum):
    LINE = "line"
    BAR = "bar"
    STACKED_BAR = "stacked_bar"
    GROUPED_BAR = "grouped_bar"
    AREA = "area"
    PIE = "pie"
    DONUT = "donut"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX = "box"
    KPI_CARD = "kpi_card"
    TABLE = "table"
    GAUGE = "gauge"
    SANKEY = "sankey"
    TREEMAP = "treemap"
    MAP = "map"


class DashboardBrief(BaseModel):
    """Standardised input for all methods (masterplan Teil 3)."""

    model_config = ConfigDict(extra="allow")

    item_id: str = ""
    users: str = ""
    goals: List[str] = Field(default_factory=list)
    kpis: List[str] = Field(default_factory=list)
    columns: List[Dict[str, str]] = Field(default_factory=list)  # [{"name":..,"dtype":..}]
    constraints: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class KPIChartMapping(BaseModel):
    model_config = ConfigDict(extra="allow")

    kpi: str = ""
    task_type: TaskType
    chart_type: ChartType
    # alternatives: model often puts non-chart values here (KPI names, categories…).
    # Accept Any and let the normaliser drop non-chart tokens before this point;
    # whatever remains is validated as ChartType. Using Any avoids hard rejection.
    alternatives: List[Any] = Field(default_factory=list)
    encoding: Dict[str, Any] = Field(default_factory=dict)


class Rationale(BaseModel):
    model_config = ConfigDict(extra="allow")

    claim: str = ""
    principle: str = ""


class DesignOutput(BaseModel):
    """Standardised output of all methods (masterplan Teil 3)."""

    model_config = ConfigDict(extra="allow")

    context_summary: Dict[str, Any] = Field(default_factory=dict)
    kpi_chart_mapping: List[KPIChartMapping] = Field(default_factory=list)
    layout: Dict[str, Any] = Field(default_factory=dict)
    styling: Dict[str, Any] = Field(default_factory=dict)
    # interactions: models often return a list of dicts ({type, label, …}) instead
    # of plain strings. Accept both — scoring only needs presence, not string type.
    interactions: List[Any] = Field(default_factory=list)
    rationales: List[Rationale] = Field(default_factory=list)


class GoldItem(BaseModel):
    """A single labelled example: brief + reference design output."""

    item_id: str
    brief: DashboardBrief
    recommendation: DesignOutput
    split: Optional[str] = None


class GenerationResult(BaseModel):
    """What every ``method.generate(brief)`` returns — one prediction row."""

    item_id: str
    method_name: str
    model_name: str
    config_hash: str = ""
    raw_text: str = ""
    parsed: Optional[DesignOutput] = None
    parse_error: Optional[str] = None
    retrieved_docs: Optional[List[Dict[str, Any]]] = None  # RAG methods only
    latency_ms: float = 0.0
    seed: int = 42
    variant: str = "original"  # "original" | "paraphrased" | "missing_info"
