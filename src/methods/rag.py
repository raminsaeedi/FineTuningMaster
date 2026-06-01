"""Method B — RAG.

Retrieves design guidelines for each brief and injects them into the prompt,
running on the base (non-fine-tuned) model. Uses the retriever named in
``cfg.method.retriever`` (default: TF-IDF over the knowledge base).
"""

from __future__ import annotations

from src.core.registry import METHODS
from src.methods.base import RAGHFMethod


@METHODS.register("rag")
class RAGMethod(RAGHFMethod):
    name = "rag"
