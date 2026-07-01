"""nvBench / nvBench 2.0 builder (STUB — data NOT ingested in this task).

Licenses are usable (nvBench: MIT; nvBench 2.0: CC-BY-4.0; see
docs/datasets/sources_table.md), but per the reduced scope no external data is
downloaded/ingested here — this fixes the mapping contract only.

Expected raw record (when provisioned), one per (query, visualization)::

    {
      "db_schema": [{"name": str, "dtype": str}],  # table columns
      "nl_query": str,                              # natural-language query
      "chart": str,                                 # nvBench chart label
      "encoding": {"x": str, "y": str, "aggregate": str | None},
      "alternatives": [str, ...],   # nvBench 2.0: other valid charts (set-valued)
      "reasoning": str | None       # nvBench 2.0: ambiguity-resolution reasoning
    }

Mapping contract -> single-KPI mini-dashboard ``GoldItem``::

    brief.users   = generic analyst persona
    brief.goals   = [intent derived from nl_query]
    brief.kpis    = [measure from encoding.y / nl_query]
    brief.columns = db_schema, dtype normalized to {datetime, categorical, numeric}
    recommendation.kpi_chart_mapping[0] = {
        kpi, task_type (inferred; LOW conf),
        chart_type (nvBench label -> ChartType; unmapped -> drop + log),
        encoding (from source),
        alternatives (nvBench 2.0 valid charts -> ChartType; else [])}
    rationales = [{claim, principle}] from nvBench 2.0 reasoning when present
    layout/styling = minimal-valid; interactions = []
    usage_tier = "train_aug"; split via ``trainval_split`` (train/val only).

Optional enhancement: group records sharing a db_schema into one multi-KPI brief.
Validation: outputs pass DashboardBrief/DesignOutput; chart_type/alternatives in enum.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.core.schemas import GoldItem
from src.data_pipeline.builders.base import BaseBuilder, DataNotProvisionedError


class NvBenchBuilder(BaseBuilder):
    source = "nvbench"
    usage_tier = "train_aug"

    def load_raw(self) -> List[Dict[str, Any]]:
        raise DataNotProvisionedError(
            "nvBench/nvBench 2.0 not ingested in this task. License is usable "
            "(MIT / CC-BY-4.0); ingestion is a later task. See training_data_mapping.md."
        )

    def to_gold_items(self) -> List[GoldItem]:
        raise DataNotProvisionedError(
            "NvBenchBuilder is a stub; implement raw->GoldItem mapping during the "
            "ingestion task (see module docstring for the contract)."
        )
