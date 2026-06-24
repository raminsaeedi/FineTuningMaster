"""Top-k chart-type accuracy (masterplan semantics).

Compares the primary recommended chart (first ``kpi_chart_mapping`` entry)
against the gold primary chart.

Top-1 (corrected): scored over **every** item that has a gold reference. An item
whose output did not parse, or that produced no chart, counts as **wrong** (not
skipped). The denominator ``n`` is therefore the number of test items with a
reference — comparable across methods regardless of parse-failure rate.

Top-3 (corrected): a genuine top-3 metric requires the model to emit an *ordered
list of 3 recommendations*. Small models rarely emit any ``alternatives``, so a
plain top-3 number is degenerate (≈ top-1) and scientifically misleading.
We therefore:
  * report ``top_3_accuracy_supported`` only over the subset of items that
    actually carry alternatives (the items where top-3 is meaningful), and
  * report a global ``top_3_accuracy`` ONLY when alternatives are present for at
    least ``TOP3_MIN_COVERAGE`` of scored items; otherwise it is ``None`` and
    ``top_3_valid`` is ``False`` so the result is read as "no valid top-3".
"""

from __future__ import annotations

from src.core.interfaces import BaseMetric
from src.core.registry import METRICS
from src.evaluation.metrics.base import (
    index_references,
    predicted_alternatives,
    predicted_charts,
    reference_charts,
)

# Minimum fraction of scored items that must carry alternatives for a global
# top-3 number to be considered valid (otherwise top-3 is degenerate vs top-1).
TOP3_MIN_COVERAGE = 0.5


@METRICS.register("top_k_accuracy")
class TopKAccuracy(BaseMetric):
    name = "top_k_accuracy"

    def compute(self, results, references) -> dict:
        ref_by_id = index_references(references or [])

        n = 0                      # items with a gold reference (top-1 denominator)
        n_predicted = 0            # items that produced a usable primary chart
        top1 = 0
        top3_global_hits = 0       # gold primary in [primary]+alternatives, over all n
        per_kpi_hits = per_kpi_total = 0
        n_with_alternatives = 0
        top3_supported_hits = 0    # top-3 hits restricted to items with alternatives
        for r in results:
            ref = ref_by_id.get(r.item_id)
            if ref is None:
                continue
            refs = reference_charts(ref)
            if not refs:
                continue
            # Every item with a reference is scored; a parse failure / empty
            # prediction simply has no charts and so counts as wrong.
            n += 1
            primary = refs[0]
            preds = predicted_charts(r)
            if not preds:
                continue
            n_predicted += 1

            pred_main = preds[0]
            alternatives = predicted_alternatives(r, 0)
            pred_top3 = [pred_main] + alternatives
            if pred_main == primary:
                top1 += 1
            if primary in pred_top3:
                top3_global_hits += 1
            if alternatives:
                n_with_alternatives += 1
                if primary in pred_top3:
                    top3_supported_hits += 1

            # Per-KPI aligned accuracy.
            for gold_c, pred_c in zip(refs, preds):
                per_kpi_total += 1
                per_kpi_hits += int(gold_c == pred_c)

        if n == 0:
            return {"top_1_accuracy": None, "top_3_valid": False, "top_3_accuracy": None,
                    "top_3_accuracy_supported": None, "per_kpi_top_1_accuracy": None,
                    "n": 0, "n_predicted": 0, "n_parse_failures": 0, "n_with_alternatives": 0}

        top3_valid = (n_with_alternatives / n) >= TOP3_MIN_COVERAGE
        return {
            "top_1_accuracy": round(100.0 * top1 / n, 2),
            # Global top-3 only reported when alternatives are present often enough
            # to make it meaningful; otherwise None + top_3_valid=False.
            "top_3_valid": top3_valid,
            "top_3_accuracy": round(100.0 * top3_global_hits / n, 2) if top3_valid else None,
            # Top-3 over the subset of items that actually emitted alternatives.
            "top_3_accuracy_supported": (
                round(100.0 * top3_supported_hits / n_with_alternatives, 2)
                if n_with_alternatives else None
            ),
            "per_kpi_top_1_accuracy": round(100.0 * per_kpi_hits / max(per_kpi_total, 1), 2),
            "n": n,                                   # items with a reference (denominator)
            "n_predicted": n_predicted,               # items with a usable prediction
            "n_parse_failures": n - n_predicted,      # counted as wrong in top-1
            "n_with_alternatives": n_with_alternatives,
        }
