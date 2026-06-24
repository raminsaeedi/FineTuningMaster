"""Robustness metrics across input perturbations.

These compare an original run against perturbed-variant runs of the same items
(joined by ``item_id``), so they take multiple result lists rather than a single
one — they are called directly by the evaluation script rather than through the
generic metric loop.

  paraphrase_consistency     — % of items whose primary predicted chart is
                               unchanged when the brief is paraphrased (stability)
  paraphrase_accuracy        — top-1 correctness on the paraphrased briefs
  paraphrase_accuracy_delta  — paraphrased accuracy − original accuracy on the
                               same items (negative = accuracy lost under
                               paraphrase). Stability alone is not enough: a model
                               can be perfectly stable yet consistently wrong, so
                               consistency is reported *together with* accuracy.

  missing_info_clarification_rate — % of missing-info predictions that ask for
                               clarification / signal uncertainty (the desired
                               behaviour when the brief is under-specified)
  missing_info_schema_rate   — % that instead confidently emit a full valid schema
                               despite the missing information (reported for
                               contrast; high values here are NOT a good outcome)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.core.constants import REQUIRED_KEYS
from src.core.schemas import GenerationResult
from src.evaluation.metrics.base import index_references, normalise, predicted_charts, reference_charts
from src.inference.postprocess import extract_json_dict

# Signals that the model asked for clarification or expressed uncertainty rather
# than confidently guessing on an under-specified brief.
_CLARIFY = re.compile(
    r"clarif|more (info|information|detail|context)|unclear|ambiguous|insufficient|"
    r"cannot (determine|tell|recommend)|can't (determine|tell)|not enough|need (more|to know)|"
    r"please (provide|specify|share)|uncertain|unsure|missing (data|info|information|column|kpi)|"
    r"underspecified|under-specified|no (kpi|goal|data|column)",
    re.IGNORECASE,
)


def _by_id(results: List[GenerationResult]) -> Dict[str, GenerationResult]:
    return {r.item_id: r for r in results}


def _top1_correct(result: GenerationResult, gold_charts: List[str]) -> bool:
    preds = [normalise(c) for c in predicted_charts(result)]
    refs = [normalise(c) for c in gold_charts]
    return bool(preds and refs and preds[0] == refs[0])


def _gold_charts_by_id(references: Optional[List[dict]]) -> Dict[str, List[str]]:
    ref_by_id = index_references(references or [])
    return {i: reference_charts(ref) for i, ref in ref_by_id.items()}


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


def paraphrase_accuracy(
    original: List[GenerationResult],
    paraphrased: Optional[List[GenerationResult]],
    references: Optional[List[dict]],
) -> Dict[str, Optional[float]]:
    """Top-1 accuracy on paraphrased briefs, plus the drop vs original.

    Computed only over items that have a gold reference and appear in both runs,
    so the original/paraphrased accuracies are directly comparable.
    """
    out: Dict[str, Optional[float]] = {"paraphrase_accuracy": None, "paraphrase_accuracy_delta": None}
    if not paraphrased or not references:
        return out
    gold = _gold_charts_by_id(references)
    orig = _by_id(original)
    para = _by_id(paraphrased)
    shared = [i for i in orig if i in para and gold.get(i)]
    if not shared:
        return out
    orig_hits = sum(_top1_correct(orig[i], gold[i]) for i in shared)
    para_hits = sum(_top1_correct(para[i], gold[i]) for i in shared)
    para_acc = 100.0 * para_hits / len(shared)
    orig_acc = 100.0 * orig_hits / len(shared)
    out["paraphrase_accuracy"] = round(para_acc, 2)
    out["paraphrase_accuracy_delta"] = round(para_acc - orig_acc, 2)
    return out


def missing_info_behaviour(
    missing_info: Optional[List[GenerationResult]],
) -> Dict[str, Optional[float]]:
    """For under-specified briefs, the desired behaviour is to ask for
    clarification / signal uncertainty — NOT to confidently emit a full schema."""
    out: Dict[str, Optional[float]] = {
        "missing_info_clarification_rate": None,
        "missing_info_schema_rate": None,
    }
    if not missing_info:
        return out
    clarifies = 0
    schema_ok = 0
    for r in missing_info:
        text = r.raw_text or ""
        if _CLARIFY.search(text):
            clarifies += 1
        obj = extract_json_dict(text)
        if obj is not None and all(k in obj for k in REQUIRED_KEYS):
            schema_ok += 1
    n = len(missing_info)
    out["missing_info_clarification_rate"] = round(100.0 * clarifies / n, 2)
    out["missing_info_schema_rate"] = round(100.0 * schema_ok / n, 2)
    return out


def compute_robustness(
    original: List[GenerationResult],
    paraphrased: Optional[List[GenerationResult]] = None,
    missing_info: Optional[List[GenerationResult]] = None,
    references: Optional[List[dict]] = None,
) -> dict:
    out: dict = {"paraphrase_consistency": paraphrase_consistency(original, paraphrased)}
    out.update(paraphrase_accuracy(original, paraphrased, references))
    out.update(missing_info_behaviour(missing_info))
    return out
