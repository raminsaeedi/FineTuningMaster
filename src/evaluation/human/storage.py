"""Read/write rater rating files (one JSONL per rater), with resume support."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Set

DEFAULT_RATINGS_DIR = "results/human_ratings"


def ratings_path(ratings_dir: str | Path, rater_id: str) -> Path:
    return Path(ratings_dir) / f"{rater_id}.jsonl"


def load_done_units(ratings_dir: str | Path, rater_id: str) -> Set[str]:
    """Return the set of unit_ids this rater has already rated (for resume)."""
    path = ratings_path(ratings_dir, rater_id)
    if not path.exists():
        return set()
    done: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["unit_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append_rating(
    ratings_dir: str | Path,
    rater_id: str,
    unit_id: str,
    item_id: str,
    method: str,
    scores: Dict[str, int],
    comment: str = "",
) -> None:
    """Append one rating row to the rater's file."""
    path = ratings_path(ratings_dir, rater_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "rater_id": rater_id,
        "unit_id": unit_id,
        "item_id": item_id,
        "method": method,  # kept for unblinding during analysis; never shown in the UI
        "scores": scores,
        "comment": comment,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_all_ratings(ratings_dir: str | Path) -> List[dict]:
    """Load every rating row across all rater files."""
    rows: List[dict] = []
    d = Path(ratings_dir)
    if not d.exists():
        return rows
    for path in sorted(d.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows
