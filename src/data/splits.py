"""Deterministic, hash-based train/val/test assignment.

Splits are derived from a stable hash of each item's id, never from a random
shuffle. This is what makes the protocol reproducible: adding new examples later
leaves every existing item in its original split, so the test set can never leak
into training as the dataset grows.
"""

from __future__ import annotations

import hashlib

# Cumulative split boundaries on the [0, 1) hash bucket.
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
# test gets the remaining 0.1


def assign_split(item_id: str) -> str:
    """Map a stable item id to ``"train"``, ``"val"`` or ``"test"``."""
    h = int(hashlib.md5(item_id.encode("utf-8")).hexdigest(), 16)
    p = (h % 10_000) / 10_000.0
    if p < TRAIN_FRACTION:
        return "train"
    if p < TRAIN_FRACTION + VAL_FRACTION:
        return "val"
    return "test"
