"""Dense (embedding) retriever over the knowledge-base chunks.

Embeds chunks and queries with a sentence-transformer (default BGE-small) and
ranks by cosine similarity. For the small guideline KB an in-memory normalized
matrix is enough, so no FAISS dependency is required. Provides a semantic
alternative to the TF-IDF retriever for the retriever ablation.

Requires the ``[rag-dense]`` extra (sentence-transformers). Imports are lazy.
The embedder can be injected (``self.embedder``) for testing without a download.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Mapping

from src.core.registry import RETRIEVERS
from src.retrievers.base import BaseRetriever, get_cfg
from src.utils.io import read_jsonl

logger = logging.getLogger(__name__)


@RETRIEVERS.register("dense")
class DenseRetriever(BaseRetriever):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg)
        self.chunks: List[dict] = []
        self.embedder = None      # may be injected for tests
        self.matrix = None

    def _load_embedder(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(str(get_cfg(self.cfg, "embedder_id", "BAAI/bge-small-en-v1.5")))

    def _encode(self, texts: List[str]):
        import numpy as np

        vecs = self.embedder.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")

    def setup(self) -> None:
        if not self.chunks:
            chunks_path = Path(str(get_cfg(self.cfg, "chunks_path", "data/knowledge_base/chunks.jsonl")))
            if not chunks_path.exists():
                raise FileNotFoundError(
                    f"Knowledge-base chunks not found: {chunks_path}. Run `python scripts/build_kb.py`."
                )
            self.chunks = read_jsonl(chunks_path)
        if not self.chunks:
            raise ValueError("Knowledge base is empty.")
        if self.embedder is None:
            self.embedder = self._load_embedder()
        self.matrix = self._encode([c.get("text", "") for c in self.chunks])
        logger.info("Dense retriever ready: %d chunks", len(self.chunks))

    def retrieve(self, query: str, k: int) -> List[dict]:
        if self.matrix is None:
            raise RuntimeError("Retriever not set up. Call setup() first.")
        q = self._encode([query])[0]
        sims = self.matrix @ q  # vectors are normalized -> dot product = cosine
        order = sims.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": float(sims[i])} for i in order if sims[i] > 0]
