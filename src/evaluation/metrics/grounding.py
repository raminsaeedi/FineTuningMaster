"""Grounding metric for RAG methods (B and D): is each rationale claim supported
by the retrieved context?

The metric is **claim-based**: every rationale claim is checked independently
against the retrieved passages. Two support modes:

  * semantic (preferred) — a sentence encoder embeds each claim and each passage;
    a claim is "supported" if its maximum cosine similarity to any passage is at
    least ``SEMANTIC_THRESHOLD``. This captures paraphrase / synonym support that
    word overlap misses. Enabled when ``sentence-transformers`` is importable and
    ``GROUNDING_SEMANTIC=1`` (opt-in, since it loads an encoder).
  * lexical_proxy (fallback) — coarse content-word overlap. This is a *proxy*,
    not a faithfulness judge; the returned ``mode`` says so explicitly so the
    number is never mistaken for true semantic grounding.

Returns ``None`` rates when predictions carry no retrieved documents (non-RAG
methods), so it is safe to include for every experiment.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

from src.core.interfaces import BaseMetric
from src.core.registry import METRICS

_WORD = re.compile(r"[a-z]{4,}")  # content-ish words, 4+ letters
SEMANTIC_THRESHOLD = 0.5          # min cosine similarity for "supported"
LEXICAL_THRESHOLD = 0.2           # min word-overlap coverage for "supported"


def _words(text: str) -> set:
    return set(_WORD.findall(text.lower()))


def _rationale_texts(parsed) -> List[str]:
    out: List[str] = []
    if parsed is None:
        return out
    for r in parsed.rationales:
        text = " ".join(p for p in [r.claim, r.principle] if p)
        if text:
            out.append(text)
    return out


def _passages(result) -> List[str]:
    return [str(d.get("text", "")) for d in (result.retrieved_docs or [])]


def _lexical_supported(claim: str, passages: List[str]) -> Optional[bool]:
    """True/False if the claim is lexically supported; None if claim has no words."""
    cw = _words(claim)
    if not cw:
        return None
    context_vocab: set = set()
    for p in passages:
        context_vocab |= _words(p)
    return (len(cw & context_vocab) / len(cw)) >= LEXICAL_THRESHOLD


def _load_encoder():
    """Return a sentence encoder if opt-in + library available, else None."""
    if os.environ.get("GROUNDING_SEMANTIC") != "1":
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    model_name = os.environ.get("GROUNDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    try:
        return SentenceTransformer(model_name)
    except Exception:
        return None


def _semantic_supported_fn(encoder):
    import numpy as np

    def supported(claim: str, passages: List[str]) -> Optional[bool]:
        if not claim.strip() or not passages:
            return None
        emb = encoder.encode([claim] + passages, normalize_embeddings=True)
        emb = np.asarray(emb)
        sims = emb[1:] @ emb[0]
        return bool(sims.max() >= SEMANTIC_THRESHOLD)

    return supported


@METRICS.register("grounding")
class Grounding(BaseMetric):
    name = "grounding"

    def compute(self, results, references=None) -> dict:
        rated = [r for r in results if r.retrieved_docs]
        if not rated:
            return {"unsupported_claim_rate": None, "supported_claim_rate": None,
                    "n": 0, "n_claims": 0, "mode": None}

        encoder = _load_encoder()
        if encoder is not None:
            supported_fn, mode = _semantic_supported_fn(encoder), "semantic"
        else:
            supported_fn, mode = _lexical_supported, "lexical_proxy"

        per_item_unsupported: List[float] = []
        total_claims = 0
        for r in rated:
            passages = _passages(r)
            claims = _rationale_texts(r.parsed)
            scored: List[bool] = []
            for claim in claims:
                verdict = supported_fn(claim, passages)
                if verdict is not None:
                    scored.append(verdict)
            if not scored:
                continue
            total_claims += len(scored)
            unsupported = sum(1 for s in scored if not s)
            per_item_unsupported.append(unsupported / len(scored))

        if not per_item_unsupported:
            return {"unsupported_claim_rate": None, "supported_claim_rate": None,
                    "n": 0, "n_claims": 0, "mode": mode}
        rate = sum(per_item_unsupported) / len(per_item_unsupported)
        return {
            "unsupported_claim_rate": round(100.0 * rate, 2),
            "supported_claim_rate": round(100.0 * (1.0 - rate), 2),
            "n": len(per_item_unsupported),
            "n_claims": total_claims,
            "mode": mode,  # "semantic" | "lexical_proxy" — read the rate accordingly
        }
