"""Grounding proxy for RAG methods (B and D).

``unsupported_claim_rate`` is a lightweight proxy: for each rationale string in
the recommendation, it measures how much of its content vocabulary appears in
the retrieved passages. Claims whose words are largely absent from the retrieved
context are counted as "unsupported". This is a coarse lexical proxy, NOT a
faithfulness judge — it is meant as a directional signal, reported as such.

Returns ``None`` when predictions carry no retrieved documents (non-RAG methods),
so it is safe to include in the metric list for every experiment.
"""

from __future__ import annotations

import re
from typing import List

from src.core.interfaces import BaseMetric
from src.core.registry import METRICS

_WORD = re.compile(r"[a-z]{4,}")  # content-ish words, 4+ letters


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


@METRICS.register("grounding")
class Grounding(BaseMetric):
    name = "grounding"

    def compute(self, results, references=None) -> dict:
        rated = [r for r in results if r.retrieved_docs]
        if not rated:
            return {"unsupported_claim_rate": None, "n": 0}

        per_item_unsupported: List[float] = []
        for r in rated:
            context_vocab: set = set()
            for d in r.retrieved_docs or []:
                context_vocab |= _words(str(d.get("text", "")))
            claims = _rationale_texts(r.parsed)
            if not claims:
                continue
            unsupported = 0
            for claim in claims:
                cw = _words(claim)
                if not cw:
                    continue
                coverage = len(cw & context_vocab) / len(cw)
                if coverage < 0.2:  # <20% of claim words seen in context
                    unsupported += 1
            per_item_unsupported.append(unsupported / len(claims))

        if not per_item_unsupported:
            return {"unsupported_claim_rate": None, "n": 0}
        rate = sum(per_item_unsupported) / len(per_item_unsupported)
        return {"unsupported_claim_rate": round(100.0 * rate, 2), "n": len(per_item_unsupported)}
