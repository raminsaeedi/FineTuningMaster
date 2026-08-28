"""Supervised retrieval relevance metrics for RAG experiments.

Qrels are JSONL rows shaped as ``{"item_id": str, "relevant_chunks":
[{"chunk_id": str, "relevance": 1 | 2}, ...]}``. Grade 0 is omitted. Scores
use ranked retrieved chunk IDs only and never infer relevance from text overlap.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from src.core.interfaces import BaseMetric
from src.core.registry import METRICS
from src.evaluation.stats.bootstrap_ci import bootstrap_ci

_MISSING = object()


def _get(cfg: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    try:
        return cfg.get(key, default)
    except AttributeError:
        return getattr(cfg, key, default)


def _unavailable(reason: str, *, applicable: bool | None = None) -> dict:
    return {
        "available": False,
        "applicable": applicable,
        "reason": reason,
        "recall_at_3": None,
        "mrr_at_3": None,
        "ndcg_at_3": None,
        "query_coverage": None,
        "top_3_retrieval_support_rate": None,
        "n_qrels": 0,
        "n_predictions": 0,
        "n_queries_with_hits": 0,
        "n_missing_predictions": 0,
        "n_with_3_unique_retrieved_ids": 0,
        "confidence_intervals": None,
    }


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"retrieval qrels file not found: {path}")
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid qrels JSON at line {line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"qrels row must be an object at line {line_number}")

            item_id = row.get("item_id")
            if not isinstance(item_id, str) or not item_id.strip():
                raise ValueError(f"item_id must be a non-empty string at line {line_number}")
            if item_id in qrels:
                raise ValueError(f"duplicate item_id '{item_id}' at line {line_number}")

            entries = row.get("relevant_chunks")
            if not isinstance(entries, list) or not entries:
                raise ValueError(
                    f"relevant_chunks must be a non-empty list at line {line_number}"
                )
            relevant: dict[str, int] = {}
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"relevant_chunks entries must be objects at line {line_number}"
                    )
                chunk_id = entry.get("chunk_id")
                if not isinstance(chunk_id, str) or not chunk_id.strip():
                    raise ValueError(
                        f"chunk_id must be a non-empty string at line {line_number}"
                    )
                if chunk_id in relevant:
                    raise ValueError(
                        f"duplicate chunk_id '{chunk_id}' for item_id '{item_id}' "
                        f"at line {line_number}"
                    )
                grade = entry.get("relevance")
                if type(grade) is not int or grade not in {1, 2}:
                    raise ValueError(
                        f"relevance must be integer 1 or 2 at line {line_number}"
                    )
                relevant[chunk_id] = grade
            qrels[item_id] = relevant
    if not qrels:
        raise ValueError("retrieval qrels file contains no rows")
    return qrels


def _dcg(grades: list[float]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))


@METRICS.register("retrieval_relevance")
class RetrievalRelevance(BaseMetric):
    """Compute Recall@3, MRR@3 and graded nDCG@3 from explicit qrels."""

    name = "retrieval_relevance"

    def compute(self, results, references=None) -> dict:
        method_cfg = _get(self.cfg, "method", _MISSING) if self.cfg is not None else _MISSING
        retriever_cfg = (
            _get(method_cfg, "retriever", _MISSING)
            if method_cfg is not _MISSING
            else _MISSING
        )
        if method_cfg is not _MISSING and (
            retriever_cfg is _MISSING or retriever_cfg is None
        ):
            return _unavailable(
                "not_applicable: cfg.method.retriever is not configured",
                applicable=False,
            )

        eval_cfg = _get(self.cfg, "eval", {}) if self.cfg is not None else {}
        qrels_path = _get(eval_cfg, "retrieval_qrels_path")
        if not qrels_path:
            return _unavailable(
                "eval.retrieval_qrels_path is not configured",
                applicable=True if method_cfg is not _MISSING else None,
            )

        qrels = _load_qrels(Path(str(qrels_path)))
        results_by_id = {result.item_id: result for result in results}
        recalls: list[float] = []
        reciprocal_ranks: list[float] = []
        ndcgs: list[float] = []
        n_predictions = 0
        n_queries_with_hits = 0
        n_with_3_unique_retrieved_ids = 0

        for item_id, relevant in qrels.items():
            result = results_by_id.get(item_id)
            if result is not None:
                n_predictions += 1
            ranked_ids: list[str | None] = []
            docs = ((result.retrieved_docs or []) if result is not None else [])[:3]
            for doc in docs:
                raw_id = doc.get("id")
                chunk_id = str(raw_id).strip() if raw_id is not None else ""
                ranked_ids.append(chunk_id or None)
            valid_ranked_ids = [chunk_id for chunk_id in ranked_ids if chunk_id is not None]
            if valid_ranked_ids:
                n_queries_with_hits += 1
            if (
                len(ranked_ids) == 3
                and len(valid_ranked_ids) == 3
                and len(set(valid_ranked_ids)) == 3
            ):
                n_with_3_unique_retrieved_ids += 1

            unique_hits = {chunk_id for chunk_id in valid_ranked_ids if chunk_id in relevant}
            recalls.append(len(unique_hits) / len(relevant))

            first_relevant_rank = next(
                (rank for rank, chunk_id in enumerate(ranked_ids, start=1) if chunk_id in relevant),
                None,
            )
            reciprocal_ranks.append(1.0 / first_relevant_rank if first_relevant_rank else 0.0)

            seen: set[str] = set()
            ranked_grades: list[float] = []
            for chunk_id in ranked_ids:
                grade = (
                    relevant.get(chunk_id, 0.0)
                    if chunk_id is not None and chunk_id not in seen
                    else 0.0
                )
                ranked_grades.append(grade)
                if chunk_id is not None:
                    seen.add(chunk_id)
            ideal_grades = sorted(relevant.values(), reverse=True)[:3]
            ideal_dcg = _dcg(ideal_grades)
            ndcgs.append(_dcg(ranked_grades) / ideal_dcg if ideal_dcg else 0.0)

        n_qrels = len(qrels)
        confidence_intervals = {
            name: {
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in bootstrap_ci(values).items()
            }
            for name, values in {
                "recall_at_3": recalls,
                "mrr_at_3": reciprocal_ranks,
                "ndcg_at_3": ndcgs,
            }.items()
        }
        return {
            "available": True,
            "applicable": True,
            "recall_at_3": round(sum(recalls) / n_qrels, 6),
            "mrr_at_3": round(sum(reciprocal_ranks) / n_qrels, 6),
            "ndcg_at_3": round(sum(ndcgs) / n_qrels, 6),
            "query_coverage": round(n_queries_with_hits / n_qrels, 6),
            "top_3_retrieval_support_rate": round(
                n_with_3_unique_retrieved_ids / n_qrels, 6
            ),
            "n_qrels": n_qrels,
            "n_predictions": n_predictions,
            "n_queries_with_hits": n_queries_with_hits,
            "n_missing_predictions": n_qrels - n_predictions,
            "n_with_3_unique_retrieved_ids": n_with_3_unique_retrieved_ids,
            "confidence_intervals": confidence_intervals,
        }
