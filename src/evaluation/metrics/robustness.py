"""Robustness metrics across input perturbations.

These compare an original run against perturbed-variant runs of the same items
(joined by ``item_id``), so they take multiple result lists rather than a single
one — they are called directly by the evaluation script rather than through the
generic metric loop.

  paraphrase_consistency    — % of items whose primary predicted chart is
                              unchanged when the brief is paraphrased
  missing_info_validity_rate — % of missing-info predictions that still produce
                              a fully valid schema
"""

from __future__ import annotations

from typing import List, Optional

from src.core.constants import REQUIRED_KEYS
from src.core.schemas import GenerationResult
from src.evaluation.metrics.base import normalise, predicted_charts
from src.inference.postprocess import extract_json_dict


def _by_id(results: List[GenerationResult]) -> dict[str, GenerationResult]:
    return {r.item_id: r for r in results}


def paraphrase_consistency(
    original: List[GenerationResult],
    paraphrased: Optional[List[GenerationResult]],
) -> Optional[float]:
    if not paraphrased:
        return None
    orig = _by_id(original)
    para = _by_id(paraphrased)
    shared = [i for i in orig if i in para]
    if not shared:
        return None
    consistent = 0
    for item_id in shared:
        o = [normalise(c) for c in predicted_charts(orig[item_id])]
        p = [normalise(c) for c in predicted_charts(para[item_id])]
        if o and p and o[0] == p[0]:
            consistent += 1
    return round(100.0 * consistent / len(shared), 2)


def missing_info_validity_rate(
    missing_info: Optional[List[GenerationResult]],
) -> Optional[float]:
    if not missing_info:
        return None
    valid = 0
    for r in missing_info:
        obj = extract_json_dict(r.raw_text)
        if obj is not None and all(k in obj for k in REQUIRED_KEYS):
            valid += 1
    return round(100.0 * valid / len(missing_info), 2)


def compute_robustness(
    original: List[GenerationResult],
    paraphrased: Optional[List[GenerationResult]] = None,
    missing_info: Optional[List[GenerationResult]] = None,
) -> dict:
    return {
        "paraphrase_consistency": paraphrase_consistency(original, paraphrased),
        "missing_info_validity_rate": missing_info_validity_rate(missing_info),
    }
