"""Version-2 nvBench selection, split, R1-sampling, and immutability helpers.

The v2 layer reuses the proven v1 maximum-cardinality repair algorithm while
adding explicit source-record deduplication and a human-R1 coverage sampler.
It never reads Tier B/C records as replacement candidates.
"""

from __future__ import annotations

import collections
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.data_pipeline.nvbench_large_v1 import (
    NON_SCATTER_CHARTS,
    NORMALIZED_CHART_TYPES,
    _brief_of,
    _db_of,
    _group_of,
    _hash_key,
    _norm_goal,
    _prov_of,
    repair_selected_v1,
    select_large_v1,
    split_train_val_test,
)

PREFERRED_TARGET_V2 = 1819
MINIMUM_ACCEPTABLE_V2 = 1800
SEED_V2 = 42
MAX_PER_GROUP_V2 = 2
NEAR_DUP_THRESHOLD_V2 = 0.8


def _deduplicate_source_records(
    tier_a_records: Iterable[Dict[str, Any]],
    previous_selected_ids: Iterable[str],
    *,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Choose one deterministic Tier-A row per exact source-record ID.

    A previously selected row wins its duplicate cluster when still valid;
    otherwise the seeded hash order decides. Missing source IDs are retained so
    downstream mandatory validation can fail explicitly rather than hide them.
    """
    previous = set(previous_selected_ids)
    clusters: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    missing: List[Dict[str, Any]] = []
    for record in tier_a_records:
        source_record_id = record.get("source_record_id")
        if not source_record_id:
            missing.append(record)
        else:
            clusters[str(source_record_id)].append(record)

    kept: List[Dict[str, Any]] = list(missing)
    dropped: List[str] = []
    duplicate_clusters: Dict[str, List[str]] = {}
    for source_record_id, records in sorted(clusters.items()):
        ordered = sorted(
            records,
            key=lambda record: (
                0 if record.get("item_id") in previous else 1,
                _hash_key(f"{seed}:source-record", str(record.get("item_id"))),
            ),
        )
        kept.append(ordered[0])
        if len(ordered) > 1:
            ids = [str(record.get("item_id")) for record in ordered]
            duplicate_clusters[source_record_id] = ids
            dropped.extend(ids[1:])
    kept.sort(key=lambda record: str(record.get("item_id")))
    return kept, {
        "input_count": sum(len(records) for records in clusters.values()) + len(missing),
        "output_count": len(kept),
        "missing_source_record_id_count": len(missing),
        "duplicate_source_record_clusters": duplicate_clusters,
        "dropped_duplicate_item_ids": sorted(dropped),
    }


def repair_selected_v2(
    tier_a_records: List[Dict[str, Any]],
    eval_sources: List[Dict[str, Any]],
    previous_selected_ids: List[str],
    *,
    previous_chart_distribution: Optional[Dict[str, int]] = None,
    seed: int = SEED_V2,
    preferred_target: int = PREFERRED_TARGET_V2,
    minimum_acceptable: int = MINIMUM_ACCEPTABLE_V2,
    db_cap: int = 100,
    near_dup_threshold: float = NEAR_DUP_THRESHOLD_V2,
    max_per_group: int = MAX_PER_GROUP_V2,
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    """Repair v1 membership using only valid, deduplicated Tier-A v2 rows."""
    empty_goal_ids = sorted(
        str(record.get("item_id")) for record in tier_a_records if not _norm_goal(record)
    )
    eligible = [
        record for record in tier_a_records
        if record.get("quality_tier") == "A"
        and record.get("quality_score", 0) >= 90
        and not record.get("failed_rules")
        and bool(_norm_goal(record))
    ]
    deduplicated, dedup_report = _deduplicate_source_records(
        eligible, previous_selected_ids, seed=seed
    )
    selected, report = repair_selected_v1(
        deduplicated,
        eval_sources,
        previous_selected_ids,
        previous_chart_distribution=previous_chart_distribution,
        seed=seed,
        preferred_target=preferred_target,
        minimum_acceptable=minimum_acceptable,
        db_cap=db_cap,
        near_dup_threshold=near_dup_threshold,
        max_per_group=max_per_group,
    )
    retention_report = report
    # Retaining a large historical subset can trap a greedy conflict-graph
    # repair below the scientifically valid maximum. If that happens, perform a
    # deterministic full Tier-A re-optimization under the exact same gates.
    if selected is None:
        reoptimized, reoptimized_report = select_large_v1(
            deduplicated,
            eval_sources,
            seed=seed,
            total=preferred_target,
            minimum_acceptable=minimum_acceptable,
            db_cap=db_cap,
            near_dup_threshold=near_dup_threshold,
            max_per_group=max_per_group,
        )
        if reoptimized is not None:
            previous_set = set(previous_selected_ids)
            selected_set = {record["item_id"] for record in reoptimized}
            reoptimized_report.update({
                "status": "reoptimized_selected_corpus",
                "retained_previous_count": len(selected_set & previous_set),
                "removed_previous_ids": sorted(previous_set - selected_set),
                "removed_previous_reasons": {
                    item_id: "not_selected_by_full_v2_reoptimization"
                    for item_id in sorted(previous_set - selected_set)
                },
                "replacement_ids": sorted(selected_set - previous_set),
                "retention_first_attempt": {
                    key: value for key, value in retention_report.items()
                    if key != "multi_record_groups"
                },
            })
            selected, report = reoptimized, reoptimized_report
        else:
            report = {
                **retention_report,
                "full_reoptimization": {
                    key: value for key, value in reoptimized_report.items()
                    if key != "multi_record_groups"
                },
            }
    report = {
        **report,
        "source_record_deduplication": dedup_report,
        "excluded_empty_normalized_goal_ids": empty_goal_ids,
        "selection_version": "v2",
    }
    if selected is not None:
        source_ids = [record.get("source_record_id") for record in selected]
        report["unique_source_record_ids"] = len(source_ids) == len(set(source_ids))
    return selected, report


def split_train_val_test_v2(
    selected: List[Dict[str, Any]],
    *,
    seed: int = SEED_V2,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    train, val, test, report = split_train_val_test(
        selected, seed=seed, val_fraction=val_fraction, test_fraction=test_fraction
    )
    report = {**report, "split_algorithm_version": "nvbench_large_v2_split_v1_group_safe"}
    return train, val, test, report


def _feature_tags(
    record: Dict[str, Any],
    group_counts: Dict[str, int],
    min_score: int,
    max_score: int,
) -> set:
    prov = _prov_of(record)
    constraints = prov.get("constraints") or {}
    grouping = prov.get("grouping") or {}
    kpi_evidence = ((record.get("evidence") or {}).get("kpi_suitability") or {}).get("evidence") or {}
    tags = {f"chart:{record.get('chart_type')}"}
    if constraints.get("filters"):
        tags.add("feature:filters")
    if constraints.get("sort"):
        tags.add("feature:sort")
    if constraints.get("limit") is not None:
        tags.add("feature:limit")
    visual = constraints.get("visual_grouping") or {}
    if grouping.get("sql_group_by_fields") or visual.get("fields"):
        tags.add("feature:grouping")
    if visual.get("origin") == "vql_bin":
        tags.add("feature:vql_bin")
    if constraints.get("time_grain"):
        tags.add("feature:time_grain")
    group_size = group_counts.get(_group_of(record), 0)
    if group_size == 1:
        tags.add("feature:one_record_group")
    if group_size == 2:
        tags.add("feature:two_record_group")
    if record.get("quality_score") == min_score:
        tags.add("feature:low_tier_a_score")
    if record.get("quality_score") == max_score:
        tags.add("feature:high_tier_a_score")
    identifier = kpi_evidence.get("identifier") or {}
    identifier_evidence = set(identifier.get("evidence") or [])
    if (
        kpi_evidence.get("policy") == "count-requires-explicit-entity-count-intent"
        and kpi_evidence.get("explicit_count_intent")
        and (identifier.get("is_identifier") or "name_pattern" in identifier_evidence)
    ):
        tags.add("feature:valid_identifier_count")
    if (
        constraints.get("sort")
        and constraints.get("limit") is not None
        and any(
            (prov.get("axis_typing") or {}).get(axis, {}).get("aggregate")
            for axis in ("x", "y")
        )
    ):
        # Tier-A membership means the repaired scope validator accepted this
        # combined ordering/limit pattern. Include a positive counterpart to
        # the demoted top-N-before-aggregation regression in human R1.
        tags.add("feature:valid_constraint_scope")
    if record.get("chart_type") == "scatter":
        tags.add("feature:valid_scatter")
        if any(
            (prov.get("axis_typing") or {}).get(axis, {}).get("aggregate")
            for axis in ("x", "y")
        ):
            tags.add("feature:valid_aggregate_scatter")
    if visual.get("origin") == "vql_bin" or grouping.get("sql_group_by_fields"):
        tags.add("feature:valid_grouping_pattern")
    return tags


def select_r1_sample(
    selected: List[Dict[str, Any]],
    *,
    seed: int = SEED_V2,
    size: int = 30,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Deterministic set-cover sample for independent human reviewer R1.

    Every available chart is represented. Scatter is capped at one row, while
    constraints, grouping variants, source-group sizes, score extremes, and
    repaired-rule counterparts receive explicit coverage slots.
    """
    if size <= 0:
        return [], {"size": 0, "covered_tags": [], "missing_available_tags": []}
    group_counts = collections.Counter(_group_of(record) for record in selected)
    scores = [int(record.get("quality_score", 0)) for record in selected]
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 0
    ordered = sorted(selected, key=lambda record: _hash_key(f"{seed}:r1", record["item_id"]))
    tags_by_id = {
        record["item_id"]: _feature_tags(record, group_counts, min_score, max_score)
        for record in ordered
    }
    available_tags = set().union(*(tags_by_id.values())) if tags_by_id else set()
    critical_counterpart_tags = {
        "feature:valid_identifier_count",
        "feature:valid_constraint_scope",
        "feature:valid_aggregate_scatter",
    }
    required_tags = {
        *(f"chart:{chart}" for chart in NORMALIZED_CHART_TYPES if any(
            record.get("chart_type") == chart for record in selected
        )),
        *(tag for tag in available_tags if tag.startswith("feature:")),
        *critical_counterpart_tags,
    }

    chosen: List[Dict[str, Any]] = []
    chosen_ids: set = set()
    covered: set = set()

    def add(record: Dict[str, Any]) -> bool:
        if record["item_id"] in chosen_ids or len(chosen) >= size:
            return False
        if record.get("chart_type") == "scatter" and any(
            item.get("chart_type") == "scatter" for item in chosen
        ):
            return False
        chosen.append(record)
        chosen_ids.add(record["item_id"])
        covered.update(tags_by_id[record["item_id"]])
        return True

    # First guarantee every chart, choosing the row with maximum feature gain.
    for chart in NORMALIZED_CHART_TYPES:
        candidates = [record for record in ordered if record.get("chart_type") == chart]
        if candidates:
            candidates.sort(key=lambda record: (
                0 if (
                    chart != "scatter"
                    or "feature:valid_aggregate_scatter" in tags_by_id[record["item_id"]]
                ) else 1,
                -len(tags_by_id[record["item_id"]] & required_tags),
                _hash_key(f"{seed}:r1-chart:{chart}", record["item_id"]),
            ))
            add(candidates[0])

    # Greedy set cover for every feature that is actually available.
    while len(chosen) < size and (required_tags - covered):
        unseen_databases = {_db_of(record) for record in selected} - {_db_of(record) for record in chosen}
        candidates = [
            record for record in ordered
            if record["item_id"] not in chosen_ids
            and not (record.get("chart_type") == "scatter" and any(
                item.get("chart_type") == "scatter" for item in chosen
            ))
        ]
        if not candidates:
            break
        best = min(candidates, key=lambda record: (
            -len(tags_by_id[record["item_id"]] & (required_tags - covered)),
            0 if _db_of(record) in unseen_databases else 1,
            _hash_key(f"{seed}:r1-cover", record["item_id"]),
        ))
        if not (tags_by_id[best["item_id"]] & (required_tags - covered)):
            break
        add(best)

    # Fill remaining positions round-robin across non-Scatter charts and prefer
    # unseen databases. Scatter stays at one to avoid artificial emphasis.
    pointers = {chart: 0 for chart in NON_SCATTER_CHARTS}
    pools = {
        chart: [record for record in ordered if record.get("chart_type") == chart]
        for chart in NON_SCATTER_CHARTS
    }
    while len(chosen) < size:
        progressed = False
        seen_databases = {_db_of(record) for record in chosen}
        for chart in NON_SCATTER_CHARTS:
            pool = pools[chart]
            remaining = [record for record in pool if record["item_id"] not in chosen_ids]
            if not remaining:
                continue
            unseen = [record for record in remaining if _db_of(record) not in seen_databases]
            candidate = (unseen or remaining)[0]
            pointers[chart] += 1
            if add(candidate):
                progressed = True
                seen_databases.add(_db_of(candidate))
            if len(chosen) >= size:
                break
        if not progressed:
            break

    chosen.sort(key=lambda record: record["item_id"])
    covered = set().union(*(tags_by_id[record["item_id"]] for record in chosen)) if chosen else set()
    coverage = {
        "size": len(chosen),
        "seed": seed,
        "chart_counts": dict(sorted(collections.Counter(
            record.get("chart_type") for record in chosen
        ).items())),
        "database_count": len({_db_of(record) for record in chosen}),
        "covered_tags": sorted(covered),
        "required_available_tags": sorted(required_tags),
        "missing_available_tags": sorted(required_tags - covered),
        "scatter_count": sum(record.get("chart_type") == "scatter" for record in chosen),
    }
    return chosen[:size], coverage


def snapshot_tree(root: Path) -> Dict[str, Dict[str, Any]]:
    """Full relative-path/size/SHA-256 snapshot for immutability gates."""
    snapshot: Dict[str, Dict[str, Any]] = {}
    if not root.is_dir():
        return snapshot
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        snapshot[path.relative_to(root).as_posix()] = {"size": path.stat().st_size, "sha256": digest}
    return snapshot
