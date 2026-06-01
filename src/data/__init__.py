"""Data layer: deterministic splits, gold-item loading, training formatter.

This package knows about the data contract (``src.core.schemas``) but nothing
about models or evaluation.
"""

from src.data.dataset import compute_item_id, filter_split, load_gold_items
from src.data.splits import assign_split

__all__ = ["assign_split", "compute_item_id", "load_gold_items", "filter_split"]
