"""Functional nvBench builder: map NVBench.json into project GoldItems.

nvBench (SIGMOD 2021) is MIT-licensed (registered and verified; see
``data/raw_external/nvbench/source_manifest.json``). This builder maps the real
source records into schema-valid ``GoldItem``s for **train/validation
augmentation only** — nvBench data never enters independent evaluation gold or an
external test split (enforced by group-aware ``trainval_split``).

Source parsing uses the actual nvBench fields: the top-level visualization key,
``chart``, ``db_id``, ``vis_query``, ``vis_obj`` and ``nl_queries``. Real column
data types are recovered from the prepared SQLite cache where required.

All source-label mappings live in a versioned YAML
(``src/config/data/nvbench_mapping.yaml``); unsupported chart labels are rejected
and reported, never silently coerced.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

from src.core.schemas import GoldItem
from src.data_pipeline.builders.base import BaseBuilder, DataNotProvisionedError
from src.data_pipeline.nvbench_source import (
    DbMetadataResolver,
    RejectedRecord,
    build_gold_item,
    item_chart,
    load_mapping,
)

_DEFAULT_MAPPING = "src/config/data/nvbench_mapping.yaml"


class BuildResult(NamedTuple):
    accepted: List[GoldItem]
    rejections: List[Dict[str, Any]]
    stats: Dict[str, Any]


class NvBenchBuilder(BaseBuilder):
    """Map real nvBench records into train/val augmentation ``GoldItem``s."""

    source = "nvbench"
    usage_tier = "train_aug"

    def __init__(
        self,
        nvbench_json_path: str | Path,
        cache_root: Optional[str | Path] = None,
        mapping_path: str | Path = _DEFAULT_MAPPING,
    ) -> None:
        self.nvbench_json_path = Path(nvbench_json_path)
        self.cache_root = Path(cache_root) if cache_root else None
        self.mapping_path = Path(mapping_path)

    # -- raw loading -------------------------------------------------------- #
    def load_raw(self) -> List[Dict[str, Any]]:
        """Return raw nvBench records as ``{"key": str, "record": dict}``, sorted."""
        if not self.nvbench_json_path.exists():
            raise DataNotProvisionedError(
                f"NVBench.json not found: {self.nvbench_json_path}. "
                "Register the nvBench source first."
            )
        with self.nvbench_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        def _sort_key(k: str):
            return (0, int(k)) if k.isdigit() else (1, k)

        return [{"key": k, "record": data[k]} for k in sorted(data, key=_sort_key)]

    # -- mapping ------------------------------------------------------------ #
    def build(self) -> BuildResult:
        """Map all records; return accepted items, rejections and distributions."""
        mapping = load_mapping(self.mapping_path)
        resolver = DbMetadataResolver(self.cache_root)

        accepted: List[GoldItem] = []
        rejections: List[Dict[str, Any]] = []

        for raw in self.load_raw():
            key = raw["key"]
            record = raw["record"]
            nl_queries = record.get("nl_queries", []) or []
            for qi, nl_query in enumerate(nl_queries):
                try:
                    accepted.append(
                        build_gold_item(key, record, qi, str(nl_query), mapping, resolver)
                    )
                except RejectedRecord as exc:
                    rejections.append(
                        {
                            "visualization_key": key,
                            "query_index": qi,
                            "reason": exc.reason,
                            "detail": exc.detail,
                            "original_chart_label": record.get("chart", ""),
                            "db_id": record.get("db_id", ""),
                            **exc.evidence,
                        }
                    )

        stats = {
            "n_accepted": len(accepted),
            "n_rejected": len(rejections),
            "chart_distribution": dict(
                collections.Counter(item_chart(it) for it in accepted)
            ),
            "task_distribution": dict(
                collections.Counter(
                    it.recommendation.kpi_chart_mapping[0].task_type.value for it in accepted
                )
            ),
            "split_distribution": dict(
                collections.Counter(it.split for it in accepted)
            ),
            "rejection_reasons": dict(
                collections.Counter(r["reason"] for r in rejections)
            ),
            "db_metadata_available": resolver.available,
            "mapping_version": mapping.get("mapping_version"),
            "task_rule_version": mapping.get("task_rules", {}).get("version"),
        }
        return BuildResult(accepted=accepted, rejections=rejections, stats=stats)

    def to_gold_items(self) -> List[GoldItem]:
        """Return only the accepted, schema-valid ``GoldItem``s."""
        return self.build().accepted
