"""TF-IDF retriever over the knowledge-base chunks.

A lightweight, CPU-only dense-free retriever: it fits a TF-IDF vectorizer over
the guideline chunks at setup and returns the top-k chunks by cosine similarity
to the query. Depends only on scikit-learn. A future embedding-based retriever
(e.g. BGE + FAISS) can be registered alongside this one under a different key.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Mapping

from src.core.registry import RETRIEVERS
from src.retrievers.base import BaseRetriever, get_cfg
from src.utils.io import read_jsonl

logger = logging.getLogger(__name__)


@RETRIEVERS.register("tfidf")
class TfidfRetriever(BaseRetriever):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg)
        self.chunks: List[dict] = []
        self.vectorizer = None
        self.matrix = None

    def setup(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        chunks_path = Path(str(get_cfg(self.cfg, "chunks_path", "data/knowledge_base/chunks.jsonl")))
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Knowledge-base chunks not found: {chunks_path}. "
                "Run `python scripts/build_kb.py` first."
            )
        self.chunks = read_jsonl(chunks_path)
        if not self.chunks:
            raise ValueError(f"Knowledge base is empty: {chunks_path}")

        texts = [c.get("text", "") for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(texts)
        logger.info("TF-IDF retriever ready: %d chunks", len(self.chunks))

    def retrieve(self, query: str, k: int) -> List[dict]:
        from sklearn.metrics.pairwise import cosine_similarity

        if self.vectorizer is None:
            raise RuntimeError("Retriever not set up. Call setup() first.")
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = scores.argsort()[::-1][:k]
        return [
            {**self.chunks[i], "score": float(scores[i])}
            for i in top_idx
            if scores[i] > 0.0
        ]
