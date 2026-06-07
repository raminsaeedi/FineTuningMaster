"""Top-k chart-type accuracy (masterplan semantics).

Compares the primary recommended chart (first ``kpi_chart_mapping`` entry)
against the gold primary chart. Top-3 counts a hit if the gold primary chart is
the prediction's primary chart or one of its listed alternatives. Also reports a
per-KPI top-1 accuracy aligned by mapping index, which is more informative on
multi-KPI briefs. Only items with both a reference and a prediction are scored.

Caveat (reported as ``n_with_alternatives``): top-3 can only differ from top-1
when the model actually emits ``alternatives``. Small models rarely do, so a
top-3 ≈ top-1 result means "no usable alternatives were produced", not "no
second-choice chart would have been correct". Read the two together.
"""

from __future__ import annotations

from src.core.interfaces import BaseMetric
from src.core.registry import METRICS
from src.evaluation.metrics.base import (
    chart_token,
    index_references,
    predicted_alternatives,
    predicted_charts,
    reference_charts,
)


@METRICS.register("top_k_accuracy")
class TopKAccuracy(BaseMetric):
    name = "top_k_accuracy"

    def compute(self, results, references) -> dict:
        ref_by_id = index_references(references or [])

        top1 = top3 = scored = 0
        per_kpi_hits = per_kpi_total = 0
        n_with_alternatives = 0
        for r in results:
            ref = ref_by_id.get(r.item_id)
            if ref is None:
                continue
            refs = reference_charts(ref)
            preds = predicted_charts(r)
            if not refs or not preds:
                continue
            scored += 1

            primary = refs[0]
            pred_main = preds[0]
            alternatives = predicted_alternatives(r, 0)
            if alternatives:
                n_with_alternatives += 1
            pred_top3 = [pred_main] + alternatives
            if pred_main == primary:
                top1 += 1
            if primary in pred_top3:
                top3 += 1

            # Per-KPI aligned accuracy.
            for gold_c, pred_c in zip(refs, preds):
                per_kpi_total += 1
                per_kpi_hits += int(gold_c == pred_c)

        if scored == 0:
            return {"top_1_accuracy": None, "top_3_accuracy": None,
                    "per_kpi_top_1_accuracy": None, "n": 0, "n_with_alternatives": 0}
        return {
            "top_1_accuracy": round(100.0 * top1 / scored, 2),
            "top_3_accuracy": round(100.0 * top3 / scored, 2),
            "per_kpi_top_1_accuracy": round(100.0 * per_kpi_hits / max(per_kpi_total, 1), 2),
            "n": scored,
            # How many scored items actually carried alternatives — top-3 can only
            # exceed top-1 on these; if ~0, top-3 is uninformative (see module docstring).
            "n_with_alternatives": n_with_alternatives,
        }
