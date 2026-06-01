"""
rag/retriever.py — RAGRetriever: load TF-IDF index and retrieve relevant guideline chunks.

Usage:
    from rag import RAGRetriever

    retriever = RAGRetriever()   # loads pre-built index from rag/index/
    context   = retriever.retrieve("Sales dashboard with monthly revenue")
    # → formatted string with top-k guideline sections, ready to prepend to a prompt

Build the index first (one-time setup):
    python -m rag.indexer

Requires: scikit-learn, numpy
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

_RAG_DIR    = Path(__file__).resolve().parent
INDEX_DIR   = _RAG_DIR / "index"
CHUNKS_FILE = INDEX_DIR / "chunks.json"
TFIDF_FILE  = INDEX_DIR / "tfidf_index.pkl"


class RAGRetriever:
    """
    Retrieve the most relevant guideline sections for a given query using TF-IDF.

    Parameters
    ----------
    index_dir : Path, optional
        Directory containing chunks.json and tfidf_index.pkl.
        Defaults to rag/index/ (relative to this file).
    top_k : int
        Default number of chunks to return (can be overridden per call).

    The index is loaded lazily on first call to retrieve() and cached for
    subsequent calls in the same process.
    """

    def __init__(
        self,
        index_dir: Path | str | None = None,
        top_k: int = 3,
    ) -> None:
        self._index_dir  = Path(index_dir) if index_dir else INDEX_DIR
        self._chunks_file = self._index_dir / "chunks.json"
        self._tfidf_file  = self._index_dir / "tfidf_index.pkl"
        self.default_top_k = top_k

        # In-memory cache (populated by _load())
        self._chunks:     list[dict] | None = None
        self._vectorizer                    = None
        self._matrix                        = None

    # ------------------------------------------------------------------
    # Index loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load index from disk into memory (idempotent — skips if already loaded)."""
        if self._chunks is not None:
            return  # already loaded

        if not self._chunks_file.exists() or not self._tfidf_file.exists():
            raise FileNotFoundError(
                f"RAG index not found in {self._index_dir}.\n"
                "Build it first with: python -m rag.indexer"
            )

        logger.info(f"Loading RAG index from {self._index_dir} …")
        with open(self._chunks_file, "r", encoding="utf-8") as f:
            self._chunks = json.load(f)

        with open(self._tfidf_file, "rb") as f:
            data = pickle.load(f)

        self._vectorizer = data["vectorizer"]
        self._matrix     = data["matrix"]
        logger.info(f"  {len(self._chunks)} chunks loaded")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> str:
        """
        Retrieve the top-k most relevant guideline sections for *query*.

        Parameters
        ----------
        query : str
            The dashboard brief or any free-text query.
        top_k : int, optional
            Override the instance-level default.

        Returns
        -------
        str
            Formatted string with the top-k sections, ready to be injected
            into a system prompt or instruction block.
        """
        chunks = self.retrieve_as_list(query, top_k=top_k)
        if not chunks:
            return "No relevant guidelines found."

        sections = []
        for rank, item in enumerate(chunks, start=1):
            source = item["source"].replace(".md", "").replace("_", " ").title()
            sections.append(
                f"[Guideline {rank} | {source} | {item['section_title']} "
                f"| relevance={item['score']:.3f}]\n"
                f"{item['text'].strip()}"
            )
        return "\n\n---\n\n".join(sections)

    def retrieve_as_list(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Same as retrieve() but returns a list of dicts.

        Each dict has keys:
            source          — filename of the source guideline
            section_title   — ## heading of the section
            text            — full section text
            score           — cosine similarity score (float)
        """
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError as exc:
            raise ImportError(
                "scikit-learn and numpy are required for RAG retrieval.\n"
                "Install with: pip install scikit-learn numpy"
            ) from exc

        self._load()
        k = top_k if top_k is not None else self.default_top_k

        query_vec = self._vectorizer.transform([query])
        scores    = cosine_similarity(query_vec, self._matrix).flatten()
        top_idx   = np.argsort(scores)[::-1][:k]

        return [
            {
                "source":        self._chunks[i]["source"],
                "section_title": self._chunks[i]["section_title"],
                "text":          self._chunks[i]["text"],
                "score":         float(scores[i]),
            }
            for i in top_idx
        ]

    def is_ready(self) -> bool:
        """Return True if the index files exist on disk."""
        return self._chunks_file.exists() and self._tfidf_file.exists()
