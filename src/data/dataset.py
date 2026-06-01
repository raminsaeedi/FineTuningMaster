"""Loading gold items and assigning stable ids.

The raw gold files store ``{brief, recommendation}`` without an id. We derive a
content-based ``item_id`` from the brief so the same brief always lands in the
same split (see ``src.data.splits``) and so predictions can be joined back to
references by id rather than by file position.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from src.core.schemas import DashboardBrief, DesignOutput, GoldItem
from src.data.splits import assign_split
from src.utils.io import read_jsonl


def compute_item_id(brief: Dict[str, Any]) -> str:
    """Stable id derived from the brief content (order-independent)."""
    canonical = json.dumps(brief, sort_keys=True, ensure_ascii=False)
    digest = hashlib.md5(canonical.encode("utf-8")).hexdigest()[:8]
    return f"item_{digest}"


def _record_to_gold(record: Dict[str, Any]) -> GoldItem:
    brief_raw = dict(record.get("brief", {}))
    item_id = record.get("item_id") or compute_item_id(brief_raw)
    brief_raw.setdefault("item_id", item_id)
    split = record.get("split") or assign_split(item_id)
    return GoldItem(
        item_id=item_id,
        brief=DashboardBrief(**brief_raw),
        recommendation=DesignOutput(**dict(record.get("recommendation", {}))),
        split=split,
    )


def load_gold_items(path: str | Path) -> List[GoldItem]:
    """Load a JSONL gold file into ``GoldItem`` objects (ids/splits filled in)."""
    return [_record_to_gold(r) for r in read_jsonl(path)]


def filter_split(items: List[GoldItem], split: str) -> List[GoldItem]:
    """Return only the items whose assigned split matches ``split``."""
    return [it for it in items if it.split == split]


def load_pool(paths: List[str | Path]) -> List[GoldItem]:
    """Merge several gold files into one de-duplicated pool keyed by item_id."""
    seen: Dict[str, GoldItem] = {}
    for path in paths:
        if not Path(path).exists():
            continue
        for item in load_gold_items(path):
            seen.setdefault(item.item_id, item)
    return list(seen.values())
