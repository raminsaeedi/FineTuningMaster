"""Method D — fine-tuned + RAG.

Combines the fine-tuned adapter (method C) with retrieval (method B).

The adapter is *not* this experiment's own — it belongs to the method C run with
the same model, dataset and seed. ``method.adapter_source_experiment`` names that
run, and :mod:`src.utils.adapter` turns it into a concrete path keyed by seed, so
``C seed 43`` feeds ``D seed 43`` without anyone editing paths by hand. The
adapter's recorded training metadata is validated against this config before
loading; a mismatch raises rather than silently producing results attributed to
the wrong seed.
"""

from __future__ import annotations

from typing import Optional

from src.core.registry import METHODS
from src.methods.base import RAGHFMethod
from src.utils.adapter import resolve_adapter_path, validate_adapter


@METHODS.register("ft_rag")
class FineTunedRAGMethod(RAGHFMethod):
    name = "ft_rag"

    def _adapter_path(self) -> Optional[str]:
        adapter_dir = resolve_adapter_path(self.cfg)
        validate_adapter(adapter_dir, self.cfg)
        return str(adapter_dir)
