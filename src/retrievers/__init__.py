"""Retrievers for RAG methods (B and D), behind the RETRIEVERS registry.

Importing this package registers the available retrievers. The TF-IDF retriever
depends only on scikit-learn (already a base dependency) and runs on CPU, so it
needs no extra install. A dense (embedding) retriever can be added later as
another registered variant without touching the methods.
"""

from src.retrievers import dense, tfidf  # noqa: F401  (register on import)
from src.retrievers.base import format_passages

__all__ = ["tfidf", "dense", "format_passages"]
