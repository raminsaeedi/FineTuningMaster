"""Synthetic-generator builder — thin wrapper over the existing generator.

Wraps ``synth_generator.generate_dataset`` so synthetic data shares the builder
contract. Behaviour matches the existing pipeline: full dashboards, content-based
``item_id`` (computed from the raw brief, excluding the provenance tag so ids stay
identical to ``build_data``), and the standard 80/10/10 ``assign_split``.

The synthetic *test* split is INTERNAL and circular (its task->chart labels come
from the generator's own fixed rule). It is suitable only for limited L2 / format
/ robustness checks, NEVER for the main chart-quality validity claim.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.core.schemas import DashboardBrief, DesignOutput, GoldItem
from src.data_pipeline.builders.base import BaseBuilder
from src.data_pipeline.dataset import compute_item_id
from src.data_pipeline.splits import assign_split
from src.data_pipeline.synth_generator import generate_dataset


class SyntheticBuilder(BaseBuilder):
    """Produce full-dashboard ``GoldItem``s from the principled generator."""

    source = "synthetic"
    usage_tier = "train_aug"  # full 80/10/10; the test split is internal-only

    def __init__(self, n: int = 600, base_seed: int = 42) -> None:
        self.n = n
        self.base_seed = base_seed

    def load_raw(self) -> List[Dict[str, Any]]:
        return generate_dataset(n=self.n, base_seed=self.base_seed)

    def to_gold_items(self) -> List[GoldItem]:
        items: List[GoldItem] = []
        for rec in self.load_raw():
            raw_brief = dict(rec.get("brief", {}))
            # id from the untagged brief -> identical to build_data's id
            item_id = compute_item_id(raw_brief)
            brief_raw = self._tag(dict(raw_brief))
            brief_raw.setdefault("item_id", item_id)
            items.append(
                GoldItem(
                    item_id=item_id,
                    brief=DashboardBrief(**brief_raw),
                    recommendation=DesignOutput(**dict(rec.get("recommendation", {}))),
                    split=assign_split(item_id),
                )
            )
        return items
