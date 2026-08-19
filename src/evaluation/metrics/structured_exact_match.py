"""Coverage-aware exact matches for learned structured-output fields."""

from __future__ import annotations

from typing import Any

from src.core.interfaces import BaseMetric
from src.core.registry import METRICS
from src.evaluation.metrics.base import align_results

_HIDDEN_CONTEXT_FIELDS = ("db_id", "source", "n_kpis")


def _rate(hits: int, n: int) -> float | None:
    return round(100.0 * hits / n, 2) if n else None


def _reference_mappings(reference: dict) -> list[dict[str, Any]]:
    recommendation = reference.get("recommendation", {}) or {}
    return [mapping for mapping in recommendation.get("kpi_chart_mapping", []) or []
            if isinstance(mapping, dict)]


def _prediction_mappings(result) -> list[dict[str, Any]]:
    if result is None or result.parsed is None:
        return []
    return [mapping.model_dump(mode="json") for mapping in result.parsed.kpi_chart_mapping]


@METRICS.register("structured_exact_match")
class StructuredExactMatch(BaseMetric):
    name = "structured_exact_match"

    def compute(self, results, references) -> dict:
        aligned = [
            (reference, result)
            for reference, result in align_results(results, references or [])
            if _reference_mappings(reference)
        ]
        n = len(aligned)
        task_hits = kpi_hits = count_hits = encoding_hits = 0
        aggregate_hits = aggregate_n = 0
        context = {
            field: {"present": 0, "exact": 0, "n": 0}
            for field in _HIDDEN_CONTEXT_FIELDS
        }

        for reference, result in aligned:
            gold = _reference_mappings(reference)
            predicted = _prediction_mappings(result)
            task_hits += [m.get("task_type") for m in predicted] == [
                m.get("task_type") for m in gold
            ]
            kpi_hits += [m.get("kpi") for m in predicted] == [m.get("kpi") for m in gold]
            count_hits += len(predicted) == len(gold)
            encoding_hits += [m.get("encoding", {}) for m in predicted] == [
                m.get("encoding", {}) for m in gold
            ]

            aggregate_applicable = all(
                "aggregate" in (mapping.get("encoding", {}) or {}) for mapping in gold
            )
            if aggregate_applicable:
                aggregate_n += 1
                aggregate_hits += len(predicted) == len(gold) and all(
                    "aggregate" in (predicted[index].get("encoding", {}) or {})
                    and predicted[index]["encoding"]["aggregate"]
                    == gold[index]["encoding"]["aggregate"]
                    for index in range(len(gold))
                )

            gold_context = (
                (reference.get("recommendation", {}) or {}).get("context_summary", {}) or {}
            )
            predicted_context = (
                result.parsed.context_summary if result is not None and result.parsed else {}
            )
            for field in _HIDDEN_CONTEXT_FIELDS:
                if field not in gold_context:
                    continue
                context[field]["n"] += 1
                context[field]["present"] += field in predicted_context
                context[field]["exact"] += (
                    field in predicted_context and predicted_context[field] == gold_context[field]
                )

        n_predictions = sum(result is not None for _, result in aligned)
        diagnostics = {
            field: {
                "presence_rate": _rate(values["present"], values["n"]),
                "exact_match_rate": _rate(values["exact"], values["n"]),
                "n_applicable": values["n"],
                "visible_in_prompt": False,
            }
            for field, values in context.items()
        }
        return {
            "exact_task_classification": _rate(task_hits, n),
            "exact_kpi_selection": _rate(kpi_hits, n),
            "exact_mapping_count": _rate(count_hits, n),
            "exact_encoding": _rate(encoding_hits, n),
            "exact_aggregate": _rate(aggregate_hits, aggregate_n),
            "n_aggregate_applicable": aggregate_n,
            "hidden_context_diagnostics": diagnostics,
            "n": n,
            "n_predictions": n_predictions,
            "n_missing_predictions": n - n_predictions,
            "prediction_coverage_rate": _rate(n_predictions, n),
        }
