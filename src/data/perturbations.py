"""Deterministic, dependency-free input perturbations for the robustness study.

Two perturbation families are produced for every test brief, keyed by the SAME
``item_id`` as the original so the robustness metric can pair them:

  paraphrase   — reword the brief's free-text fields with meaning-preserving
                 synonym swaps. A robust model should keep the same primary
                 chart (``paraphrase_consistency``).
  missing_info — drop non-essential information (constraints, the last KPI and
                 the last data column). A robust model should still emit a fully
                 valid schema (``missing_info_validity_rate``).

Everything here is pure and deterministic (no LLM, no randomness), so the
perturbed sets are reproducible and safe to commit/regenerate.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict

# Meaning-preserving synonym swaps applied to framing/verb words only — domain
# nouns (KPI names, columns) are left intact so the recommended chart shouldn't
# change. Keys are matched case-insensitively on whole words.
_SYNONYMS: Dict[str, str] = {
    "show": "display",
    "display": "show",
    "track": "monitor",
    "monitor": "track",
    "compare": "contrast",
    "analyze": "examine",
    "analyse": "examine",
    "identify": "detect",
    "understand": "interpret",
    "view": "see",
    "goal": "objective",
    "goals": "objectives",
    "key": "main",
    "metric": "measure",
    "metrics": "measures",
    "across": "over",
    "overview": "summary",
}


def paraphrase_text(text: str) -> str:
    """Reword a string with deterministic, meaning-preserving synonym swaps."""
    if not text:
        return text

    def repl(m: re.Match) -> str:
        word = m.group(0)
        low = word.lower()
        if low not in _SYNONYMS:
            return word
        sub = _SYNONYMS[low]
        if word.istitle():
            return sub.title()
        if word.isupper():
            return sub.upper()
        return sub

    return re.sub(r"[A-Za-z]+", repl, text)


def paraphrase_brief(brief: Dict[str, Any]) -> Dict[str, Any]:
    """Return a paraphrased copy of a brief dict (free-text fields only)."""
    out = deepcopy(brief)
    if isinstance(out.get("users"), str):
        out["users"] = paraphrase_text(out["users"])
    if isinstance(out.get("constraints"), str):
        out["constraints"] = paraphrase_text(out["constraints"])
    for field in ("goals", "kpis"):
        if isinstance(out.get(field), list):
            out[field] = [paraphrase_text(x) if isinstance(x, str) else x for x in out[field]]
    return out


def drop_info_brief(brief: Dict[str, Any]) -> Dict[str, Any]:
    """Return an under-specified copy: drop constraints, last KPI and last column."""
    out = deepcopy(brief)
    out["constraints"] = None
    kpis = out.get("kpis")
    if isinstance(kpis, list) and len(kpis) > 1:
        out["kpis"] = kpis[:-1]
    cols = out.get("columns")
    if isinstance(cols, list) and len(cols) > 1:
        out["columns"] = cols[:-1]
    return out
