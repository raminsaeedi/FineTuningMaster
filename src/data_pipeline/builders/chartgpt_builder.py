"""ChartGPT builder (STUB — data/license NOT provisioned).

ChartGPT (arXiv:2311.01920) has an unconfirmed license -> cite-and-ask
(see docs/datasets/sources_table.md). Do NOT download/vendor data in this task.

Expected raw record (when provisioned), one per chart::

    {
      "utterance": str,                       # abstract NL request
      "table": [{"name": str, "dtype": str}],  # source data columns
      "chart": str,                           # ChartGPT chart label
      "encoding": {"x": str, "y": str, "color": str | None},
      "operations": [str, ...]                # optional transforms
    }

Mapping contract -> single-KPI mini-dashboard ``GoldItem``::

    brief.users   = generic analyst persona
    brief.goals   = [intent derived from utterance]
    brief.kpis    = [primary measure from encoding.y / utterance]
    brief.columns = table, dtype normalized to {datetime, categorical, numeric}
    recommendation.kpi_chart_mapping[0] = {
        kpi, task_type (inferred via data/eval/task_crosswalk.yaml; LOW conf),
        chart_type (ChartGPT label -> ChartType; unmapped -> drop + log),
        encoding (from source), alternatives: []}
    layout/styling = minimal-valid; interactions = []; rationales = []
    usage_tier = "train_aug"; split via ``trainval_split`` (train/val only).

Validation: every output must pass DashboardBrief/DesignOutput and chart_type
must be a ChartType enum value.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.core.schemas import GoldItem
from src.data_pipeline.builders.base import BaseBuilder, DataNotProvisionedError


class ChartGPTBuilder(BaseBuilder):
    source = "chartgpt"
    usage_tier = "train_aug"

    def load_raw(self) -> List[Dict[str, Any]]:
        raise DataNotProvisionedError(
            "ChartGPT data/license not provisioned (cite-and-ask). See "
            "docs/datasets/sources_table.md and training_data_mapping.md."
        )

    def to_gold_items(self) -> List[GoldItem]:
        raise DataNotProvisionedError(
            "ChartGPTBuilder is a stub; implement raw->GoldItem mapping after "
            "license confirmation (see module docstring for the contract)."
        )
