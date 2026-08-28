"""Coverage-aware exact matches for learned structured-output fields."""

from __future__ import annotations

from typing import Any

from src.core.interfaces import BaseMetric
from src.core.registry import METRICS
from src.evaluation.metrics.base import align_results

_HIDDEN_CONTEXT_FIELDS = ("db_id", "source", "n_kpis")

# Encoding fields the prompt actually asks the model to produce. The gold
# ``encoding`` additionally carries source-derived bookkeeping (``source_x``,
# ``visual_grouping``, ``having``, ``time_grain``, ...) that is never requested,
# so strict whole-dict equality is unreachable by construction and would report
# 0% for every method. ``exact_encoding`` therefore scores the requested core
# fields; the unreachable strict number is kept as a diagnostic.
_CORE_ENCODING_FIELDS = ("x", "y", "aggregate")


def _core_encoding(mapping: dict[str, Any]) -> dict[str, str]:
    """Requested encoding fields, case-normalised (``SUM`` == ``sum``)."""
    encoding = mapping.get("encoding", {}) or {}
    return {
        field: str(encoding[field]).strip().lower()
        for field in _CORE_ENCODING_FIELDS
        if encoding.get(field) is not None
    }


def _rate(hits: int, n: int) -> float | None:
    return round(100.0 * hits / n, 2) if n else None


def _normalised_value(value: Any) -> str:
    return str(value).strip().lower()


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
        task_hits = kpi_hits = count_hits = encoding_hits = strict_encoding_hits = 0
        aggregate_hits = aggregate_n = 0
        encoding_field_hits = {field: 0 for field in _CORE_ENCODING_FIELDS}
        encoding_field_n = {field: 0 for field in _CORE_ENCODING_FIELDS}
        encoding_mapping_hits = encoding_mapping_n = 0
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
            encoding_hits += [_core_encoding(m) for m in predicted] == [
                _core_encoding(m) for m in gold
            ]
            strict_encoding_hits += [m.get("encoding", {}) for m in predicted] == [
                m.get("encoding", {}) for m in gold
            ]

            # Micro accuracy per KPI mapping. Missing mappings/fields count as
            # wrong; fields absent from gold are not applicable. This separates
            # structural validity from semantic x/y/aggregate correctness.
            for index, gold_mapping in enumerate(gold):
                gold_encoding = gold_mapping.get("encoding", {}) or {}
                applicable = [
                    field for field in _CORE_ENCODING_FIELDS
                    if gold_encoding.get(field) is not None
                ]
                if not applicable:
                    continue
                encoding_mapping_n += 1
                predicted_encoding = (
                    (predicted[index].get("encoding", {}) or {})
                    if index < len(predicted)
                    else {}
                )
                mapping_exact = True
                for field in applicable:
                    encoding_field_n[field] += 1
                    matched = (
                        predicted_encoding.get(field) is not None
                        and _normalised_value(predicted_encoding[field])
                        == _normalised_value(gold_encoding[field])
                    )
                    encoding_field_hits[field] += int(matched)
                    mapping_exact = mapping_exact and matched
                encoding_mapping_hits += int(mapping_exact)

            aggregate_applicable = all(
                "aggregate" in (mapping.get("encoding", {}) or {}) for mapping in gold
            )
            if aggregate_applicable:
                aggregate_n += 1
                aggregate_hits += len(predicted) == len(gold) and all(
                    "aggregate" in (predicted[index].get("encoding", {}) or {})
                    # Case-insensitive: the aggregate is a SQL function name, so
                    # "COUNT" and "count" are the same answer.
                    and str(predicted[index]["encoding"]["aggregate"]).strip().lower()
                    == str(gold[index]["encoding"]["aggregate"]).strip().lower()
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
            # Whole-dict equality incl. source-derived gold bookkeeping fields
            # that the prompt never requests — diagnostic only, expected ~0.
            "exact_encoding_strict": _rate(strict_encoding_hits, n),
            "encoding_fields_scored": list(_CORE_ENCODING_FIELDS),
            "encoding_x_accuracy": _rate(
                encoding_field_hits["x"], encoding_field_n["x"]
            ),
            "encoding_y_accuracy": _rate(
                encoding_field_hits["y"], encoding_field_n["y"]
            ),
            "encoding_aggregate_accuracy": _rate(
                encoding_field_hits["aggregate"], encoding_field_n["aggregate"]
            ),
            "encoding_mapping_exact_accuracy": _rate(
                encoding_mapping_hits, encoding_mapping_n
            ),
            "n_encoding_mappings": encoding_mapping_n,
            "n_encoding_x_fields": encoding_field_n["x"],
            "n_encoding_y_fields": encoding_field_n["y"],
            "n_encoding_aggregate_fields": encoding_field_n["aggregate"],
            "exact_aggregate": _rate(aggregate_hits, aggregate_n),
            "n_aggregate_applicable": aggregate_n,
            "hidden_context_diagnostics": diagnostics,
            "n": n,
            "n_predictions": n_predictions,
            "n_missing_predictions": n - n_predictions,
            "prediction_coverage_rate": _rate(n_predictions, n),
        }
