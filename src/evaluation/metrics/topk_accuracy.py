"""Top-k chart-type accuracy (masterplan semantics).

Compares the primary recommended chart (first ``kpi_chart_mapping`` entry)
against the gold primary chart.

Top-1 (corrected): scored over **every** item that has a gold reference. An item
whose output did not parse, or that produced no chart, counts as **wrong** (not
skipped). The denominator ``n`` is therefore the number of test items with a
reference — comparable across methods regardless of parse-failure rate.

Top-3 (corrected): a per-item top-3 is *supported* only when the model emits an
**ordered list of 3 distinct recommendations** — the primary chart plus at least
two distinct, valid (``ChartType``) alternatives, after de-duplicating and
dropping any alternative equal to the primary. Small models rarely do this, so a
plain top-3 is degenerate (≈ top-1) and misleading. We therefore:
  * ``top_3_accuracy_supported`` — top-3 hit rate over the supported subset only;
  * ``top_3_support_rate`` / ``n_with_3_recs`` — how often 3 distinct recs exist;
  * ``top_3_accuracy`` — a headline number reported ONLY when
    ``top_3_support_rate >= TOP3_MIN_SUPPORT``; otherwise it is ``None`` and
    ``top_3_valid`` is ``False`` (read as "no valid top-3").
"""

from __future__ import annotations

from src.core.interfaces import BaseMetric
from src.core.registry import METRICS
from src.core.schemas import ChartType
from src.evaluation.metrics.base import (
    align_results,
    predicted_alternatives,
    predicted_charts,
    reference_charts,
)

# Minimum fraction of scored items that must carry 3 distinct ordered
# recommendations for the global top-3 number to be considered valid.
TOP3_MIN_SUPPORT = 0.8

# Valid chart tokens (lowercase) — alternatives outside this set are ignored.
_VALID_CHARTS = {c.value for c in ChartType}


@METRICS.register("top_k_accuracy")
class TopKAccuracy(BaseMetric):
    name = "top_k_accuracy"

    def compute(self, results, references) -> dict:
        n = 0                      # items with a gold reference (top-1 denominator)
        n_predicted = 0            # items that produced a usable primary chart
        n_missing_predictions = 0
        top1 = 0
        top3_global_hits = 0       # gold primary in the model's (<=3) distinct recs
        per_kpi_hits = per_kpi_total = 0
        n_with_alternatives = 0    # items emitting >= 1 raw alternative (diagnostic)
        n_with_3_recs = 0          # items emitting 3 distinct ordered recs
        top3_supported_hits = 0    # top-3 hits restricted to the 3-rec subset
        for ref, r in align_results(results, references or []):
            refs = reference_charts(ref)
            if not refs:
                continue
            # Every item with a reference is scored; a parse failure / empty
            # prediction simply has no charts and so counts as wrong.
            n += 1
            if r is None:
                n_missing_predictions += 1
                per_kpi_total += len(refs)
                continue
            primary = refs[0]
            preds = predicted_charts(r)
            if not preds:
                continue
            n_predicted += 1

            pred_main = preds[0]
            if pred_main == primary:
                top1 += 1

            raw_alts = predicted_alternatives(r, 0)
            if raw_alts:
                n_with_alternatives += 1
            # Build the ordered, distinct recommendation list: primary first, then
            # valid alternatives that are not duplicates and not equal to primary.
            recs = [pred_main]
            for alt in raw_alts:
                if alt in _VALID_CHARTS and alt not in recs:
                    recs.append(alt)
            top3 = recs[:3]
            if primary in top3:
                top3_global_hits += 1
            if len(recs) >= 3:
                n_with_3_recs += 1
                if primary in top3:
                    top3_supported_hits += 1

            # Per-KPI aligned accuracy.
            for index, gold_c in enumerate(refs):
                per_kpi_total += 1
                per_kpi_hits += int(index < len(preds) and gold_c == preds[index])

        if n == 0:
            return {"top_1_accuracy": None, "top_3_valid": False, "top_3_accuracy": None,
                    "top_3_accuracy_supported": None, "top_3_support_rate": None,
                    "per_kpi_top_1_accuracy": None, "n": 0, "n_predicted": 0,
                    "n_parse_failures": 0, "n_missing_predictions": 0,
                    "n_with_alternatives": 0, "n_with_3_recs": 0}

        top3_support_rate = n_with_3_recs / n
        top3_valid = top3_support_rate >= TOP3_MIN_SUPPORT
        return {
            "top_1_accuracy": round(100.0 * top1 / n, 2),
            # Global top-3 reported only when 3 distinct recs are present often
            # enough to be meaningful; otherwise None + top_3_valid=False.
            "top_3_valid": top3_valid,
            "top_3_accuracy": round(100.0 * top3_global_hits / n, 2) if top3_valid else None,
            # Top-3 over the subset of items that emitted 3 distinct ordered recs.
            "top_3_accuracy_supported": (
                round(100.0 * top3_supported_hits / n_with_3_recs, 2)
                if n_with_3_recs else None
            ),
            "top_3_support_rate": round(top3_support_rate, 4),
            "per_kpi_top_1_accuracy": round(100.0 * per_kpi_hits / max(per_kpi_total, 1), 2),
            "n": n,                                   # items with a reference (denominator)
            "n_predicted": n_predicted,               # items with a usable prediction
            "n_parse_failures": n - n_predicted,      # counted as wrong in top-1
            "n_missing_predictions": n_missing_predictions,
            "n_with_alternatives": n_with_alternatives,
            "n_with_3_recs": n_with_3_recs,
        }
