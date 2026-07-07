"""Dependency-free near-duplicate similarity for leakage checks.

Uses character 3-gram Jaccard similarity over a normalized brief text. This is a
reproducible first-pass detector (no embeddings); an optional embedding-based
upgrade is tracked separately. Keep thresholds explicit and report top pairs — no
silent truncation.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple


def brief_text(brief: dict) -> str:
    """Normalized text view of a brief (users + goals + kpis + column names)."""
    parts: List[str] = [str(brief.get("users", ""))]
    parts += [str(g) for g in (brief.get("goals") or [])]
    parts += [str(k) for k in (brief.get("kpis") or [])]
    parts += [str(c.get("name", "")) for c in (brief.get("columns") or []) if isinstance(c, dict)]
    text = " ".join(parts).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text)).strip()


def char_ngrams(text: str, n: int = 3) -> Set[str]:
    text = text.replace(" ", "_")
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def near_duplicate_pairs(
    left: List[Tuple[str, dict]],
    right: List[Tuple[str, dict]],
    threshold: float = 0.8,
    n: int = 3,
) -> List[Dict[str, object]]:
    """Cross-compare two id→brief lists; return pairs with Jaccard >= threshold.

    Each item is ``(id, brief_dict)``. Sorted by descending similarity.
    """
    left_grams = [(lid, char_ngrams(brief_text(b), n)) for lid, b in left]
    right_grams = [(rid, char_ngrams(brief_text(b), n)) for rid, b in right]
    pairs: List[Dict[str, object]] = []
    for lid, lg in left_grams:
        for rid, rg in right_grams:
            sim = jaccard(lg, rg)
            if sim >= threshold:
                pairs.append({"left_id": lid, "right_id": rid, "similarity": round(sim, 4)})
    pairs.sort(key=lambda p: p["similarity"], reverse=True)
    return pairs
