"""Method D — fine-tuned + RAG.

Combines the fine-tuned adapter (method C) with retrieval (method B): loads the
adapter from ``cfg.method.adapter_path`` and injects retrieved guidelines into
the prompt.
"""

from __future__ import annotations

from typing import Optional

from src.core.registry import METHODS
from src.methods.base import RAGHFMethod, _get


@METHODS.register("ft_rag")
class FineTunedRAGMethod(RAGHFMethod):
    name = "ft_rag"

    def _adapter_path(self) -> Optional[str]:
        path = _get(self.method_cfg, "adapter_path")
        if not path:
            raise ValueError(
                "method 'ft_rag' requires cfg.method.adapter_path to point at a "
                "trained adapter folder."
            )
        return str(path)
