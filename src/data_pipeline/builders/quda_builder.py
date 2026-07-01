"""Quda builder (STUB — data/license NOT provisioned).

Quda (arXiv:2005.03257) has no explicit reusable license -> cite-and-ask
(see docs/datasets/sources_table.md). Do NOT download/vendor data in this task.

Quda's signal is the analytic *task* of a natural-language query (one or more of
the Amar 10 tasks); it does NOT provide a chart. Its value is task_type diversity.

Expected raw record (when provisioned), one per query::

    {
      "query": str,             # natural-language analytics query
      "tasks": [str, ...]       # one or more Amar task labels
    }

Mapping contract -> single-KPI mini-dashboard ``GoldItem``::

    brief.users   = generic analyst persona
    brief.goals   = [query]
    brief.kpis    = [subject extracted from query]
    brief.columns = [] or minimal inferred
    recommendation.kpi_chart_mapping[0] = {
        kpi,
        task_type = Quda label mapped via data/eval/task_crosswalk.yaml (HIGH conf),
        chart_type = DERIVED from task_type via the generator's TASK_CHART rule
                     (TRAINING-ONLY derivation; never used as eval gold),
        encoding: {}, alternatives: []}
    layout/styling = minimal-valid; interactions = []; rationales = []
    usage_tier = "train_aug"; split via ``trainval_split`` (train/val only).

Note: the chart_type here is rule-derived, not independent — acceptable for
training only. Quda mappings must never become evaluation gold.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.core.schemas import GoldItem
from src.data_pipeline.builders.base import BaseBuilder, DataNotProvisionedError


class QudaBuilder(BaseBuilder):
    source = "quda"
    usage_tier = "train_aug"

    def load_raw(self) -> List[Dict[str, Any]]:
        raise DataNotProvisionedError(
            "Quda data/license not provisioned (cite-and-ask). See "
            "docs/datasets/sources_table.md and training_data_mapping.md."
        )

    def to_gold_items(self) -> List[GoldItem]:
        raise DataNotProvisionedError(
            "QudaBuilder is a stub; implement query->task_type mapping after "
            "license confirmation (see module docstring for the contract)."
        )
