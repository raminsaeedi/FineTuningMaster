"""Hybrid dataset construction: enriched nvBench + validated synthetic records.

Combines the source-grounded, LLM-enriched nvBench train/val records with the
validated synthetic train/val records. nvBench contributes the analytical ground
(goals, KPIs, columns, chart types, encodings, constraints); synthetic records
contribute richer multi-KPI dashboard structures. Neither is human or expert gold.

Split membership is never changed: nvBench train stays train, val stays val, and
the same holds for the synthetic source. Cross-source conflicts are resolved in
favour of the source-grounded nvBench record.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from src.core.schemas import GoldItem
from src.data_pipeline.enrichment import ENRICHABLE_FIELDS
from src.data_pipeline.enrichment_full import (
    LINEAGE_DETERMINISTIC,
    LINEAGE_LLM,
    LINEAGE_SOURCE_BACKED,
    lineage_classification,
)
from src.data_pipeline.leakage_similarity import char_ngrams, jaccard

HYBRID_SPEC_VERSION = "hybrid-v1"

SOURCE_NVBENCH = "source_nvbench"
SOURCE_SYNTHETIC = "synthetic_generated"

# Two briefs whose character 3-gram Jaccard reaches this are treated as the same
# analytical intent; the synthetic one is dropped in favour of nvBench.
NEAR_DUPLICATE_THRESHOLD = 0.8


# ---------------------------------------------------------------- provenance


def _brief_extra(record: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(((record.get("brief") or {}).get("extra") or {}))


def tag_source(record: Mapping[str, Any], source: str) -> Dict[str, Any]:
    """Return a copy carrying explicit dataset source and field lineage."""
    tagged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in record.items()}
    brief = dict(tagged.get("brief") or {})
    extra = dict(brief.get("extra") or {})
    extra["dataset_source"] = source

    if source == SOURCE_NVBENCH:
        classes = lineage_classification(record)
    else:
        # Synthetic records are deterministically generated from documented source
        # templates; nothing in them is source-backed evidence.
        classes = {
            LINEAGE_SOURCE_BACKED: [],
            LINEAGE_DETERMINISTIC: ["brief", "kpi_chart_mapping", *ENRICHABLE_FIELDS],
            LINEAGE_LLM: [],
        }
    extra.setdefault("lineage_classes", classes)
    extra["hybrid_spec_version"] = HYBRID_SPEC_VERSION
    brief["extra"] = extra
    tagged["brief"] = brief
    return tagged


def has_provenance(record: Mapping[str, Any]) -> bool:
    extra = _brief_extra(record)
    return bool(extra.get("dataset_source")) and bool(extra.get("lineage_classes"))


# ---------------------------------------------------------------- dedup


def normalized_goal(record: Mapping[str, Any]) -> str:
    goals = (record.get("brief") or {}).get("goals") or []
    text = " ".join(str(g) for g in goals).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def brief_fingerprint_text(record: Mapping[str, Any]) -> str:
    brief = record.get("brief") or {}
    parts = [normalized_goal(record),
             " ".join(str(k) for k in brief.get("kpis") or []),
             " ".join(str(c.get("name")) for c in brief.get("columns") or []
                      if isinstance(c, Mapping))]
    return re.sub(r"[^a-z0-9]+", " ", " ".join(parts).lower()).strip()


def cross_source_duplicates(nvbench: Sequence[Mapping[str, Any]],
                            synthetic: Sequence[Mapping[str, Any]],
                            threshold: float = NEAR_DUPLICATE_THRESHOLD,
                            ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Drop synthetic records that collide with a source-grounded nvBench record.

    Checks exact item_id, exact/normalized goal text and near-duplicate briefs.
    nvBench always wins; complementary synthetic examples that merely share a
    chart class are kept.
    """
    nv_ids = {str(r.get("item_id")) for r in nvbench}
    nv_goals = {normalized_goal(r) for r in nvbench if normalized_goal(r)}
    # Goal text carries the analytical intent; the full brief adds KPIs and columns.
    # Either signal reaching the threshold means the same record in two guises.
    nv_grams = [(str(r.get("item_id")),
                 char_ngrams(brief_fingerprint_text(r)),
                 char_ngrams(normalized_goal(r))) for r in nvbench]

    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for record in synthetic:
        item_id = str(record.get("item_id"))
        goal = normalized_goal(record)
        reason = None
        match = None
        if item_id in nv_ids:
            reason, match = "duplicate_item_id", item_id
        elif goal and goal in nv_goals:
            reason, match = "duplicate_normalized_goal", goal
        else:
            brief_grams = char_ngrams(brief_fingerprint_text(record))
            goal_grams = char_ngrams(goal)
            for nv_id, nv_brief_gram, nv_goal_gram in nv_grams:
                brief_score = jaccard(brief_grams, nv_brief_gram)
                goal_score = jaccard(goal_grams, nv_goal_gram) if goal_grams and nv_goal_gram else 0.0
                if brief_score >= threshold:
                    reason, match = "near_duplicate_brief", f"{nv_id} (brief jaccard={brief_score:.3f})"
                    break
                if goal_score >= threshold:
                    reason, match = "near_duplicate_goal", f"{nv_id} (goal jaccard={goal_score:.3f})"
                    break
        if reason:
            dropped.append({"item_id": item_id, "split": record.get("split"),
                            "reason": reason, "conflicting_nvbench": match,
                            "policy": "prefer source-grounded nvBench record"})
        else:
            kept.append(dict(record))
    return kept, dropped


# ---------------------------------------------------------------- validation


def schema_problems(records: Iterable[Mapping[str, Any]]) -> List[str]:
    """Validate every record against the current Pydantic contract."""
    problems: List[str] = []
    for record in records:
        try:
            GoldItem(**dict(record))
        except Exception as exc:  # noqa: BLE001 - reported per record
            problems.append(f"{record.get('item_id')}: {type(exc).__name__}")
    return problems


def required_field_problems(records: Iterable[Mapping[str, Any]]) -> List[str]:
    """Non-empty checks for the fields every training record must carry."""
    problems: List[str] = []
    for record in records:
        item_id = record.get("item_id")
        brief = record.get("brief") or {}
        recommendation = record.get("recommendation") or {}
        if not str(brief.get("users") or "").strip():
            problems.append(f"{item_id}: empty users")
        if not (brief.get("goals") or []):
            problems.append(f"{item_id}: empty goals")
        if not (brief.get("kpis") or []):
            problems.append(f"{item_id}: empty kpis")
        if not (recommendation.get("kpi_chart_mapping") or []):
            problems.append(f"{item_id}: empty kpi_chart_mapping")
        for field in ("layout", "styling", "interactions", "rationales"):
            if not recommendation.get(field):
                problems.append(f"{item_id}: empty {field}")
    return problems


def source_group_of(record: Mapping[str, Any]) -> str:
    provenance = (_brief_extra(record).get("provenance") or {})
    return str(provenance.get("source_group_id") or record.get("item_id"))


def split_overlap(train: Sequence[Mapping[str, Any]],
                  val: Sequence[Mapping[str, Any]]) -> Dict[str, List[str]]:
    """Item-id and source-group overlap between the two hybrid splits."""
    train_ids = {str(r.get("item_id")) for r in train}
    val_ids = {str(r.get("item_id")) for r in val}
    train_groups = {source_group_of(r) for r in train}
    val_groups = {source_group_of(r) for r in val}
    return {
        "duplicate_item_ids": sorted(train_ids & val_ids),
        "shared_source_groups": sorted(train_groups & val_groups),
    }


def holdout_collisions(records: Sequence[Mapping[str, Any]],
                       holdout: Sequence[Mapping[str, Any]],
                       threshold: float = NEAR_DUPLICATE_THRESHOLD) -> Dict[str, List[Any]]:
    """Collisions against a held-out set (nvBench test, human-evaluation items)."""
    hold_ids = {str(r.get("item_id")) for r in holdout}
    hold_goals = {normalized_goal(r) for r in holdout if normalized_goal(r)}
    hold_groups = {source_group_of(r) for r in holdout}
    hold_grams = [(str(r.get("item_id")), char_ngrams(brief_fingerprint_text(r))) for r in holdout]

    near: List[Dict[str, Any]] = []
    for record in records:
        grams = char_ngrams(brief_fingerprint_text(record))
        for hold_id, hold_gram in hold_grams:
            score = jaccard(grams, hold_gram)
            if score >= threshold:
                near.append({"item_id": record.get("item_id"), "holdout_item_id": hold_id,
                             "jaccard": round(score, 3)})
                break
    return {
        "shared_item_ids": sorted({str(r.get("item_id")) for r in records} & hold_ids),
        "shared_normalized_goals": sorted({normalized_goal(r) for r in records if normalized_goal(r)}
                                          & hold_goals),
        "shared_source_groups": sorted({source_group_of(r) for r in records} & hold_groups),
        "near_duplicates": near,
    }


# ------------------------------------------------------------- distributions


def _mapping0(record: Mapping[str, Any]) -> Dict[str, Any]:
    mappings = (record.get("recommendation") or {}).get("kpi_chart_mapping") or [{}]
    return dict(mappings[0]) if mappings else {}


def _domain_of(record: Mapping[str, Any]) -> str:
    extra = _brief_extra(record)
    provenance = extra.get("provenance") or {}
    return str(provenance.get("db_id") or extra.get("domain") or "unknown")


def distribution_rows(records: Sequence[Mapping[str, Any]], split: str) -> List[Dict[str, Any]]:
    """Long-format distribution rows (source, chart, task, domain, KPI count)."""
    rows: List[Dict[str, Any]] = []
    counters: Dict[Tuple[str, str], int] = {}

    def bump(dimension: str, value: str) -> None:
        key = (dimension, str(value))
        counters[key] = counters.get(key, 0) + 1

    for record in records:
        extra = _brief_extra(record)
        mapping = _mapping0(record)
        n_kpis = len((record.get("recommendation") or {}).get("kpi_chart_mapping") or [])
        bump("source", extra.get("dataset_source", "unknown"))
        bump("chart_type", mapping.get("chart_type", "unknown"))
        bump("task_type", mapping.get("task_type", "unknown"))
        bump("domain", _domain_of(record))
        bump("kpi_cardinality", "multi_kpi" if n_kpis > 1 else "single_kpi")

    for (dimension, value), count in sorted(counters.items()):
        rows.append({"split": split, "dimension": dimension, "value": value, "records": count})
    return rows


def summarize(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    sources: Dict[str, int] = {}
    charts: Dict[str, int] = {}
    multi = 0
    for record in records:
        source = _brief_extra(record).get("dataset_source", "unknown")
        sources[source] = sources.get(source, 0) + 1
        chart = _mapping0(record).get("chart_type", "unknown")
        charts[chart] = charts.get(chart, 0) + 1
        if len((record.get("recommendation") or {}).get("kpi_chart_mapping") or []) > 1:
            multi += 1
    return {
        "records": len(records),
        "by_source": dict(sorted(sources.items())),
        "by_chart_type": dict(sorted(charts.items())),
        "multi_kpi_records": multi,
        "single_kpi_records": len(records) - multi,
        "unique_item_ids": len({str(r.get("item_id")) for r in records}),
    }
