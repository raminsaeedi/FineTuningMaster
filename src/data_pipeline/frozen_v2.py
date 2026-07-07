"""Freeze helpers for dataset v2.

Turns validated candidate items (from ``raw_batches/``) into the deterministic
frozen splits. Splits reuse the existing content-hash rule
(:func:`src.data_pipeline.splits.assign_split`); the **stored** split value keeps
the legacy literals ``"train" | "val" | "test"`` for schema/loader compatibility,
while the ``"test"`` bucket is written to the file named ``internal_test.jsonl``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from src.core.schemas import GoldItem
from src.data_pipeline.builders.leakage import fingerprint

# Stored split literal -> frozen file name. Note "test" -> internal_test.jsonl.
STORED_SPLIT_TO_FILE: Dict[str, str] = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "internal_test.jsonl",
}


def gold_to_record(item: GoldItem) -> dict:
    """Serialise a GoldItem to a frozen JSONL record (enums -> plain strings)."""
    return {
        "item_id": item.item_id,
        "split": item.split,
        "brief": item.brief.model_dump(mode="json"),
        "recommendation": item.recommendation.model_dump(mode="json"),
    }


def dedupe_by_fingerprint(items: List[GoldItem]) -> Tuple[List[GoldItem], List[Dict[str, str]]]:
    """Drop items whose brief fingerprint already appeared (keeps first).

    Item-id de-duplication is handled upstream by ``load_pool``; this catches
    distinct ids that nonetheless describe the same brief content.
    """
    seen: set[str] = set()
    kept: List[GoldItem] = []
    dropped: List[Dict[str, str]] = []
    for it in items:
        fp = fingerprint(it.brief)
        if fp in seen:
            dropped.append({"item_id": it.item_id, "reason": "duplicate_fingerprint"})
            continue
        seen.add(fp)
        kept.append(it)
    return kept, dropped


def bucket_by_split(items: List[GoldItem]) -> Dict[str, List[GoldItem]]:
    """Group items by their stored split literal (train/val/test)."""
    buckets: Dict[str, List[GoldItem]] = {"train": [], "val": [], "test": []}
    for it in items:
        if it.split in buckets:
            buckets[it.split].append(it)
    return buckets
