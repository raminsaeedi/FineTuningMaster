"""Builder framework: map raw source data into the project ``GoldItem`` schema.

Builders do ONLY raw -> schema mapping. Deduplication / leakage checks live in
``leakage`` and split policy is the helper below; downstream pipeline code
(``build_data``) is unchanged. See ``docs/datasets/training_data_mapping.md``.

Augmentation sources are train/val only and never final evaluation gold:
"No dataset artifact, label set, or label-generation lineage is used both for
training/augmentation and final independent evaluation gold."
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.core.schemas import GoldItem
from src.data_pipeline.splits import assign_split


class DataNotProvisionedError(RuntimeError):
    """Raised when a builder's source data/license is not yet provisioned."""


def trainval_split(item_id: str) -> str:
    """Deterministic split for augmentation sources: train/val only, never test.

    Reuses the project hash split (``assign_split``) and remaps the ``test``
    bucket to ``train`` so augmentation data can never land in any test split.
    """
    split = assign_split(item_id)
    return "train" if split == "test" else split


class BaseBuilder(ABC):
    """Map one raw source into provenance-tagged ``GoldItem`` records."""

    #: short source key, e.g. "synthetic", "chartgpt"
    source: str = ""
    #: usage tier; augmentation sources are train/val only
    usage_tier: str = "train_aug"

    @abstractmethod
    def load_raw(self) -> List[Dict[str, Any]]:
        """Return raw source records (or raise ``DataNotProvisionedError``)."""

    @abstractmethod
    def to_gold_items(self) -> List[GoldItem]:
        """Map raw records into schema-valid ``GoldItem``s, tagged with provenance."""

    def _tag(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Stamp ``extra.source`` / ``extra.usage_tier`` onto a brief dict."""
        extra = dict(brief.get("extra") or {})
        extra.setdefault("source", self.source)
        extra.setdefault("usage_tier", self.usage_tier)
        brief["extra"] = extra
        return brief
