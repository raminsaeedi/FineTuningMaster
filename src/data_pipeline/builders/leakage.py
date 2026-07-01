"""Simple first-pass leakage / duplicate prevention for augmentation data.

Two checks only (intentionally minimal — no fuzzy/shingled matching yet):
  1. exact ``item_id`` collision,
  2. normalized free-text fingerprint collision (lowercased, whitespace- and
     order-normalized brief text).

Any candidate colliding with a reference (evaluation) set is dropped and
reported, so an augmentation item can never duplicate a held-out eval item.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Tuple

from src.core.schemas import GoldItem


def fingerprint(brief: Any) -> str:
    """Order-independent, lowercased fingerprint of a brief's text content."""
    b = brief.model_dump() if hasattr(brief, "model_dump") else dict(brief)
    parts = {
        "users": str(b.get("users", "")).strip().lower(),
        "goals": sorted(str(g).strip().lower() for g in (b.get("goals") or [])),
        "kpis": sorted(str(k).strip().lower() for k in (b.get("kpis") or [])),
        "columns": sorted(
            str(c.get("name", "")).strip().lower() for c in (b.get("columns") or [])
        ),
    }
    canonical = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def filter_against(
    candidates: List[GoldItem], reference: Iterable[GoldItem]
) -> Tuple[List[GoldItem], List[Dict[str, str]]]:
    """Drop candidates colliding with ``reference`` by item_id or fingerprint.

    Returns ``(kept, dropped_report)``; ``dropped_report`` lists each dropped
    item's id and the reason ("item_id" or "fingerprint"). Callers should log
    the report (no silent truncation).
    """
    ref_ids = {it.item_id for it in reference}
    ref_fps = {fingerprint(it.brief) for it in reference}
    kept: List[GoldItem] = []
    dropped: List[Dict[str, str]] = []
    for it in candidates:
        if it.item_id in ref_ids:
            dropped.append({"item_id": it.item_id, "reason": "item_id"})
        elif fingerprint(it.brief) in ref_fps:
            dropped.append({"item_id": it.item_id, "reason": "fingerprint"})
        else:
            kept.append(it)
    return kept, dropped
