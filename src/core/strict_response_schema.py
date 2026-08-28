"""Strict generation-only schema for constrained dashboard responses.

Parsing and scoring continue to use the lenient models in ``src.core.schemas``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from src.core.schemas import ChartType, TaskType

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, pattern=r".*\S.*"),
]
AggregateFunction = Literal["SUM", "AVG", "COUNT", "MIN", "MAX"]


class StrictEncoding(BaseModel):
    """Machine-checkable channel mapping required from generated charts."""

    model_config = ConfigDict(extra="forbid")

    x: NonEmptyString
    y: NonEmptyString
    aggregate: AggregateFunction | None


class StrictKPIChartMapping(BaseModel):
    """Strict chart recommendation used only during generation."""

    model_config = ConfigDict(extra="forbid")

    kpi: NonEmptyString
    task_type: TaskType
    chart_type: ChartType
    alternatives: list[ChartType]
    encoding: StrictEncoding


class StrictRationale(BaseModel):
    """Evidence statement without undeclared generated fields."""

    model_config = ConfigDict(extra="forbid")

    claim: str
    principle: str


class StrictDesignOutput(BaseModel):
    """Required response sections emitted by constrained generation."""

    model_config = ConfigDict(extra="forbid")

    context_summary: dict[str, Any]
    kpi_chart_mapping: list[StrictKPIChartMapping] = Field(min_length=1)
    layout: dict[str, Any]
    styling: dict[str, Any]
    interactions: list[Any]
    rationales: list[StrictRationale]
