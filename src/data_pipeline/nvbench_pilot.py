"""Build and validate the versioned staging nvBench pilot (nothing is frozen).

Pure, reusable logic: builds the deterministic 100-item pilot record set and runs
structural, semantic, duplicate and leakage validation plus distribution analysis.
All source metadata is read from the actual nested locations
(``brief.extra.provenance`` / ``brief.extra.lineage`` / ``brief.extra.task_inference``)
— never assumed at the record top level.

I/O is confined to the thin orchestrator CLI; functions here take/return plain
Python structures so they can be unit-tested without touching disk.
"""

from __future__ import annotations

import collections
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from src.data_pipeline.benchmark_validation import item_brief
from src.data_pipeline.builders.leakage import fingerprint
from src.data_pipeline.builders.nvbench_builder import NvBenchBuilder
from src.data_pipeline.frozen_validation import find_duplicate_ids, validate_record
from src.data_pipeline.leakage_similarity import brief_text, char_ngrams, jaccard, near_duplicate_pairs
from src.data_pipeline.nvbench_extract import (
    check_aggregate_intent_conflict,
    check_chart_query_conflict,
    extract_base_field,
    extract_nested,
)
from src.data_pipeline.nvbench_source import (
    apply_limit,
    item_chart,
    item_group_id,
    map_chart,
    parse_aggregate,
    select_one_per_group,
)

NEAR_DUP_THRESHOLD = 0.8  # char-3gram Jaccard; matches experiments/scripts/check_dataset_leakage.py
_LINEAGE_FIELDS = (
    "chart_type",
    "encoding",
    "goal",
    "kpi",
    "task_type",
    "layout",
    "styling",
    "interactions",
    "rationales",
)


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def _record(item) -> dict:
    return {
        "item_id": item.item_id,
        "split": item.split,
        "brief": item.brief.model_dump(mode="json"),
        "recommendation": item.recommendation.model_dump(mode="json"),
    }


DEFAULT_TARGET_PER_CHART = 20
DEFAULT_DB_CAP = 10


def _hash_key(seed: int, item_id: str) -> str:
    return hashlib.md5(f"{seed}:{item_id}".encode("utf-8")).hexdigest()


def _norm_goal_text(item) -> str:
    goals = item.brief.goals or [""]
    return " ".join(str(goals[0]).strip().lower().split())


def select_pilot_v3(
    items: List[Any],
    *,
    seed: int = 42,
    target_per_chart: int = DEFAULT_TARGET_PER_CHART,
    db_cap: int = DEFAULT_DB_CAP,
    near_dup_threshold: float = NEAR_DUP_THRESHOLD,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Deterministic, near-duplicate-aware, database-capped pilot sampling.

    Pipeline: one query per source group -> drop exact normalized-goal duplicate
    clusters (keep the lowest deterministic hash) -> for each normalized chart
    type, greedily admit up to ``target_per_chart`` candidates in deterministic
    hash order, skipping any candidate whose brief is a near-duplicate (Jaccard
    >= ``near_dup_threshold``) of an already-admitted item or that would push a
    database over ``db_cap``. If a chart bucket cannot reach its target under the
    cap, the cap is relaxed for that bucket only in a second deterministic pass,
    and every such admission is recorded in the returned fallback report.

    Returns ``(selected, report)`` where ``report`` documents chart/database
    counts and any fallback admissions (never silent).
    """
    pool = select_one_per_group(items, seed=seed)

    # Drop exact normalized-goal duplicate clusters, keep the lowest-hash item.
    clusters: Dict[str, List[Any]] = collections.defaultdict(list)
    for it in pool:
        clusters[_norm_goal_text(it)].append(it)
    deduped = []
    dropped_goal_dups = 0
    for cluster in clusters.values():
        cluster.sort(key=lambda it: _hash_key(seed, it.item_id))
        deduped.append(cluster[0])
        dropped_goal_dups += len(cluster) - 1
    deduped.sort(key=lambda it: it.item_id)

    charts = sorted({item_chart(it) for it in deduped})
    buckets = {c: sorted([it for it in deduped if item_chart(it) == c],
                         key=lambda it: _hash_key(seed, it.item_id)) for c in charts}

    admitted: List[Any] = []
    admitted_ids: set = set()
    admitted_ngrams: List[frozenset] = []
    db_counts: Dict[str, int] = collections.defaultdict(int)
    chart_counts: Dict[str, int] = {c: 0 for c in charts}
    fallbacks: List[Dict[str, Any]] = []

    def db_of(it) -> str:
        return it.brief.extra.get("provenance", {}).get("db_id", "?")

    def is_near_dup(ngrams: frozenset) -> bool:
        return any(jaccard(ngrams, other) >= near_dup_threshold for other in admitted_ngrams)

    def try_admit(it, *, respect_cap: bool) -> bool:
        if it.item_id in admitted_ids:
            return False
        if respect_cap and db_counts[db_of(it)] >= db_cap:
            return False
        ngrams = frozenset(char_ngrams(brief_text(it.brief.model_dump(mode="json"))))
        if is_near_dup(ngrams):
            return False
        admitted.append(it)
        admitted_ids.add(it.item_id)
        admitted_ngrams.append(ngrams)
        db_counts[db_of(it)] += 1
        return True

    # Primary pass: strict per-database cap.
    for chart in charts:
        for it in buckets[chart]:
            if chart_counts[chart] >= target_per_chart:
                break
            if try_admit(it, respect_cap=True):
                chart_counts[chart] += 1

    # Fallback pass: relax the database cap only for buckets left short.
    for chart in charts:
        if chart_counts[chart] >= target_per_chart:
            continue
        for it in buckets[chart]:
            if chart_counts[chart] >= target_per_chart:
                break
            if try_admit(it, respect_cap=False):
                chart_counts[chart] += 1
                fallbacks.append({
                    "chart": chart, "item_id": it.item_id, "db_id": db_of(it),
                    "reason": "db_cap_relaxed",
                    "detail": f"database cap ({db_cap}) exhausted for chart '{chart}'; "
                              "admitted over cap to reach the per-chart target.",
                })

    admitted.sort(key=lambda it: it.item_id)
    report = {
        "target_per_chart": target_per_chart,
        "db_cap": db_cap,
        "near_dup_threshold": near_dup_threshold,
        "dropped_exact_goal_duplicates": dropped_goal_dups,
        "chart_counts": chart_counts,
        "db_counts": dict(db_counts),
        "fallbacks": fallbacks,
    }
    return admitted, report


def build_pilot_records(
    nvbench_json: str,
    cache_root: Optional[str],
    mapping_path: str,
    *,
    limit: Optional[int],
    seed: int,
    one_per_group: bool,
    stratify: bool,
    target_per_chart: Optional[int] = None,
    db_cap: int = DEFAULT_DB_CAP,
) -> Tuple[List[dict], List[dict], Dict[str, Any]]:
    """Build the versioned staging pilot records deterministically.

    Returns ``(records, rejections, build_meta)``. When ``target_per_chart`` is
    given, selection uses :func:`select_pilot_v3` (one-per-group, exact-goal-dedup,
    chart-balanced, database-capped, near-duplicate-aware). Otherwise it falls
    back to the simpler one-per-group + chart-stratified-limit selection used by
    pilot v1/v2, so both remain reproducible from this single code path.
    """
    builder = NvBenchBuilder(nvbench_json, cache_root=cache_root, mapping_path=mapping_path)
    result = builder.build()

    items = result.accepted
    sampling_report: Optional[Dict[str, Any]] = None
    if target_per_chart is not None:
        items, sampling_report = select_pilot_v3(
            items, seed=seed, target_per_chart=target_per_chart, db_cap=db_cap,
        )
    else:
        if one_per_group:
            items = select_one_per_group(items, seed=seed)
        items = apply_limit(items, limit, stratify_by_chart=stratify, seed=seed)
    records = [_record(it) for it in items]

    build_meta = {
        "seed": seed,
        "selection": {
            "limit": limit,
            "one_query_per_group": one_per_group,
            "stratify_by_chart": stratify,
            "target_per_chart": target_per_chart,
            "db_cap": db_cap,
        },
        "counts": {
            "accepted_total": result.stats["n_accepted"],
            "rejected_total": result.stats["n_rejected"],
            "selected": len(records),
        },
        "mapping_version": result.stats["mapping_version"],
        "task_rule_version": result.stats["task_rule_version"],
        "db_metadata_available": result.stats["db_metadata_available"],
        "rejection_reasons": result.stats["rejection_reasons"],
        "sampling_fallback": sampling_report,
    }
    return records, result.rejections, build_meta


# --------------------------------------------------------------------------- #
# nested accessors (never assume top-level provenance)
# --------------------------------------------------------------------------- #
def _extra(rec: dict) -> dict:
    return (rec.get("brief") or {}).get("extra") or {}


def _prov(rec: dict) -> dict:
    return _extra(rec).get("provenance") or {}


def _lineage(rec: dict) -> dict:
    return _extra(rec).get("lineage") or {}


def _task(rec: dict) -> dict:
    return _extra(rec).get("task_inference") or {}


def _mapping0(rec: dict) -> dict:
    maps = (rec.get("recommendation") or {}).get("kpi_chart_mapping") or []
    return maps[0] if maps else {}


def _check(name: str, passed: bool, severity: str, item_ids=None, detail: str = "") -> dict:
    ids = sorted(set(item_ids or []))
    return {"check": name, "passed": passed, "severity": severity, "n": len(ids), "item_ids": ids, "detail": detail}


# --------------------------------------------------------------------------- #
# structural validation
# --------------------------------------------------------------------------- #
def structural_checks(records: List[dict], expected: int = 100) -> List[dict]:
    checks: List[dict] = []

    checks.append(_check("count_exactly_expected", len(records) == expected, "mandatory",
                         detail=f"{len(records)} records (expected {expected})"))

    # Pydantic + enum + non-empty (reuse validate_record on read-back records).
    schema_bad = [r["item_id"] for r in records if validate_record(r)]
    checks.append(_check("pydantic_schema_valid", not schema_bad, "mandatory", schema_bad,
                         "records failing schema/enum/non-empty validation"))

    dup_ids = find_duplicate_ids(records)
    checks.append(_check("unique_item_ids", not dup_ids, "mandatory", dup_ids, "duplicate item_id"))

    srids = [_prov(r).get("source_record_id") for r in records]
    dup_srids = [s for s, c in collections.Counter(srids).items() if s and c > 1]
    checks.append(_check("unique_source_record_ids", not dup_srids, "mandatory", dup_srids,
                         "duplicate source_record_id"))

    gids = [_prov(r).get("source_group_id") for r in records]
    n_groups = len({g for g in gids if g})
    checks.append(_check("unique_source_group_count", n_groups == expected, "mandatory",
                         detail=f"{n_groups} unique source_group_id (expected {expected})"))

    # stable ids present + well-formed
    missing_ids = [r.get("item_id", "<none>") for r in records
                   if not r.get("item_id") or not _prov(r).get("source_record_id") or not _prov(r).get("source_group_id")]
    checks.append(_check("stable_ids_present", not missing_ids, "mandatory", missing_ids,
                         "records missing item_id/source_record_id/source_group_id"))

    # group-aware split: no source group straddles train/val
    group_splits: Dict[str, set] = collections.defaultdict(set)
    for r in records:
        group_splits[_prov(r).get("source_group_id")].add(r.get("split"))
    straddling = sorted(g for g, s in group_splits.items() if len(s) > 1)
    bad_split_ids = [r["item_id"] for r in records if _prov(r).get("source_group_id") in set(straddling)]
    checks.append(_check("no_cross_split_group_leakage", not straddling, "mandatory", bad_split_ids,
                         f"groups split across train/val: {straddling}"))

    splits = {r.get("split") for r in records}
    checks.append(_check("splits_train_val_only", splits <= {"train", "val"}, "mandatory",
                         detail=f"splits present: {sorted(splits)}"))
    return checks


# --------------------------------------------------------------------------- #
# semantic validation
# --------------------------------------------------------------------------- #
def semantic_checks(records: List[dict], mapping: dict, resolver=None) -> Tuple[List[dict], List[dict]]:
    """Semantic + axis-typing/KPI/grouping validation.

    ``resolver`` (optional ``DbMetadataResolver``) enables the live
    non-aggregate-dtype-agrees-with-database check; when omitted that check is
    skipped (recorded as not-applicable).
    """
    bad: Dict[str, list] = collections.defaultdict(list)
    warnings: List[dict] = []
    scatter_records = 0
    db_checked = 0

    for r in records:
        iid = r.get("item_id", "<none>")
        brief = r.get("brief") or {}
        m = _mapping0(r)
        enc = m.get("encoding") or {}
        prov = _prov(r)
        lin = _lineage(r)
        task = _task(r)
        col_names = {str(c.get("name")) for c in (brief.get("columns") or [])}
        axis_typing = prov.get("axis_typing") or {}
        kpi_sel = prov.get("kpi_selection") or {}
        grouping = prov.get("grouping") or {}
        label = prov.get("original_chart_label", "")

        # surface builder warnings (single source of warnings)
        for w in prov.get("build_warnings") or []:
            warnings.append({"item_id": iid, **w})

        # 1) recommended KPI exists in brief.kpis
        if m.get("kpi") not in (brief.get("kpis") or []):
            bad["kpi_present_in_brief"].append(iid)

        # 2) encoding x/y refer to a known column or an aggregate expression
        for axis in ("x", "y"):
            val = enc.get(axis)
            if val in (None, ""):
                continue
            if val not in col_names and not parse_aggregate(str(val)):
                bad["encoding_refers_known_column_or_aggregate"].append(iid)
                break

        # 3) normalized chart type matches the source chart
        try:
            if m.get("chart_type") != map_chart(label, mapping)["chart_type"].value:
                bad["chart_type_matches_source"].append(iid)
        except Exception:
            bad["chart_type_matches_source"].append(iid)

        # 4) original chart label preserved
        if not label:
            bad["original_chart_label_preserved"].append(iid)

        # 5) grouping/classify preserved for grouping charts
        if label in ("Grouping Line", "Grouping Scatter"):
            if not enc.get("grouped") or not grouping.get("classify"):
                bad["grouping_classify_preserved"].append(iid)

        # 6/7) task rule-derived + metadata present
        if task.get("derivation_status") != "rule-derived":
            bad["task_type_rule_derived"].append(iid)
        if not (task.get("rule_version") and task.get("evidence") is not None
                and isinstance(task.get("confidence"), (int, float))):
            bad["task_meta_present"].append(iid)

        # 8) template-derived fields not labeled source-provided
        if any(lin.get(f) != "template-derived" for f in ("layout", "styling", "interactions", "rationales")):
            bad["template_fields_not_source_provided"].append(iid)

        # 9) no LLM-generated field
        if any(str(v) == "LLM-generated" for v in lin.values()):
            bad["no_llm_generated_fields"].append(iid)

        # --- axis typing / KPI / grouping strengthened checks ---
        for axis in ("x", "y"):
            t = axis_typing.get(axis) or {}
            if t.get("aggregate"):
                if t.get("dtype") != "number":
                    bad["aggregate_not_categorical"].append(iid)
                if t.get("role") != "measure":
                    bad["aggregate_not_dimension"].append(iid)
            else:
                # 12) non-aggregate dtype agrees with database metadata (if available)
                if resolver is not None and resolver.available:
                    db_dtype = resolver.dtype_of(prov.get("db_id", ""), t.get("name", ""))
                    if db_dtype is not None:
                        db_checked += 1
                        if db_dtype != t.get("dtype"):
                            bad["nonagg_dtype_agrees_db"].append(iid)

        # 13) y not forced numeric without evidence
        yt = axis_typing.get("y") or {}
        if not yt.get("aggregate") and yt.get("dtype") == "number" and yt.get("dtype_source") != "source-provided(db)":
            bad["y_not_forced_numeric"].append(iid)

        # 14/15) scatter: both axes typed independently; categorical axes warned
        if m.get("chart_type") == "scatter":
            scatter_records += 1
            if not (axis_typing.get("x") and axis_typing.get("y")):
                bad["scatter_axes_typed_independently"].append(iid)
            wtypes = {(w.get("axis"), w.get("type")) for w in (prov.get("build_warnings") or [])}
            for axis in ("x", "y"):
                if (axis_typing.get(axis) or {}).get("dtype") == "categorical" \
                        and (axis, "scatter_non_numeric_axis") not in wtypes:
                    bad["categorical_scatter_warned"].append(iid)

        # 16) KPI policy consistent
        if kpi_sel.get("primary_kpi") != m.get("kpi"):
            bad["kpi_policy_consistent"].append(iid)
        elif kpi_sel.get("aggregate_axes") and not parse_aggregate(str(m.get("kpi", ""))):
            bad["kpi_policy_consistent"].append(iid)

        # 17) recovered grouping field exists in columns
        sf = grouping.get("series_field")
        if sf and sf not in col_names:
            bad["recovered_group_field_in_columns"].append(iid)

        # 18) source chart + raw encoding unchanged
        raw = prov.get("raw_encoding") or {}
        vo = prov.get("vis_obj") or {}
        if (raw.get("x_name") != vo.get("x_name") or raw.get("y_name") != vo.get("y_name")
                or raw.get("classify") != vo.get("classify")):
            bad["raw_encoding_unchanged"].append(iid)

        # 19) no aggregate expression appears in brief.columns (raw columns only)
        if any(parse_aggregate(str(c.get("name"))) for c in (brief.get("columns") or [])):
            bad["no_aggregate_in_columns"].append(iid)

        # SQL identifiers are case-insensitive; a field can be written in a
        # different case in the WHERE/ORDER BY/BIN clause than in the SELECT
        # clause that produced brief.columns, so compare case-insensitively.
        col_names_lower = {c.lower() for c in col_names}

        # 20) every aggregate's base field is present in brief.columns
        for axis in ("x", "y"):
            t = axis_typing.get(axis) or {}
            if not t.get("aggregate"):
                continue
            base = extract_base_field(t.get("name", ""))
            if base and base.lower() not in col_names_lower:
                bad["aggregate_base_field_in_columns"].append(iid)

        # 21) every group/filter/sort/time-bin field is present in brief.columns
        constraints = prov.get("constraints") or {}
        required_fields = []
        if grouping.get("series_field"):
            required_fields.append(grouping["series_field"])
        for f in constraints.get("filters") or []:
            required_fields.append(f.get("field"))
        sort_c = constraints.get("sort")
        if sort_c:
            sort_expr = sort_c.get("field", "")
            # COUNT(*)-style sort keys have no base field and are correctly not
            # added to brief.columns (same rule as x/y aggregates); only require
            # a field here when one is actually expected.
            sort_field = extract_base_field(sort_expr) if parse_aggregate(sort_expr) else sort_expr
            if sort_field:
                required_fields.append(sort_field)
        tg = constraints.get("time_grain")
        if tg:
            required_fields.append(tg.get("field"))
        if any(f and f.lower() not in col_names_lower for f in required_fields):
            bad["constraint_fields_in_columns"].append(iid)

        # 22) every accepted stacked bar has a valid, present group field
        if m.get("chart_type") == "stacked_bar":
            gf = enc.get("group_field")
            if not gf or gf not in col_names:
                bad["stacked_bar_has_group_field"].append(iid)

        # 23) every accepted scatter has two numeric axes
        if m.get("chart_type") == "scatter":
            if not all((axis_typing.get(a) or {}).get("dtype") == "number" for a in ("x", "y")):
                bad["scatter_two_numeric_axes"].append(iid)

        # 24) no nested aggregate remains in the final encoded expression
        if any(extract_nested(str(enc.get(a) or "")) for a in ("x", "y")):
            bad["no_nested_aggregate_remains"].append(iid)

        # 25) query-requested chart vs. source chart consistency (defense-in-depth
        #    re-check of a builder-time reject condition on the serialized record)
        if check_chart_query_conflict(label, [prov.get("nl_query", "")]):
            bad["query_chart_consistent"].append(iid)

        # 26) query-aggregate intent vs. encoded aggregate agreement
        for axis in ("x", "y"):
            expr = enc.get(axis)
            if expr and parse_aggregate(str(expr)):
                if check_aggregate_intent_conflict(str(expr), [prov.get("nl_query", "")], resolver, prov.get("db_id", "")):
                    bad["query_aggregate_agrees"].append(iid)

    def mk(name, detail, severity="mandatory"):
        ids = bad.get(name, [])
        return _check(name, not ids, severity, ids, detail)

    checks = [
        mk("kpi_present_in_brief", "recommended KPI missing from brief.kpis"),
        mk("encoding_refers_known_column_or_aggregate", "encoding x/y not a column nor aggregate"),
        mk("chart_type_matches_source", "normalized chart type != nvBench source chart"),
        mk("original_chart_label_preserved", "provenance.original_chart_label empty"),
        mk("grouping_classify_preserved", "grouping chart missing grouped flag or classify"),
        mk("task_type_rule_derived", "task derivation_status != rule-derived"),
        mk("task_meta_present", "task missing rule_version/confidence/evidence"),
        mk("template_fields_not_source_provided", "template fields mislabeled"),
        mk("no_llm_generated_fields", "a lineage field is LLM-generated"),
        mk("aggregate_not_categorical", "aggregate axis has non-number dtype"),
        mk("aggregate_not_dimension", "aggregate axis has dimension role"),
        mk("nonagg_dtype_agrees_db", f"non-aggregate axis dtype disagrees with DB ({db_checked} checked)"),
        mk("y_not_forced_numeric", "y forced to number without DB evidence"),
        mk("scatter_axes_typed_independently", f"scatter axes not both typed ({scatter_records} scatter records)"),
        mk("categorical_scatter_warned", "categorical scatter axis without warning"),
        mk("kpi_policy_consistent", "KPI selection violates documented policy"),
        mk("recovered_group_field_in_columns", "recovered grouping field missing from columns"),
        mk("raw_encoding_unchanged", "raw encoding differs from source vis_obj"),
        mk("no_aggregate_in_columns", "an aggregate expression appears in brief.columns"),
        mk("aggregate_base_field_in_columns", "aggregate base field missing from brief.columns"),
        mk("constraint_fields_in_columns", "a group/filter/sort/time-bin field is missing from brief.columns"),
        mk("stacked_bar_has_group_field", "accepted stacked_bar without a valid group_field"),
        mk("scatter_two_numeric_axes", "accepted scatter without two numeric axes"),
        mk("no_nested_aggregate_remains", "a nested aggregate expression remains in the encoding"),
        mk("query_chart_consistent", "query explicitly requests a chart different from the source chart"),
        mk("query_aggregate_agrees", "query aggregate intent conflicts with the encoded aggregate"),
    ]
    return checks, warnings


# --------------------------------------------------------------------------- #
# duplicate validation (within the pilot)
# --------------------------------------------------------------------------- #
def _norm_goal(goal: str) -> str:
    return " ".join(str(goal).strip().lower().split())


def duplicate_checks(records: List[dict], *, strict: bool = False) -> Tuple[List[dict], List[dict]]:
    """Duplicate detection within the pilot.

    ``strict=True`` (pilot v3) promotes exact normalized-goal duplicates and
    within-pilot near-duplicates to mandatory failures, since v3's selection
    (:func:`select_pilot_v3`) is designed to exclude both by construction; a
    violation indicates a sampling defect, not merely a reviewable warning.
    ``strict=False`` (pilot v1/v2 compatibility) keeps both as warnings.
    """
    findings: List[dict] = []
    dup_severity = "mandatory" if strict else "warning"

    dup_ids = find_duplicate_ids(records)
    dup_srids = [s for s, c in collections.Counter(
        _prov(r).get("source_record_id") for r in records).items() if s and c > 1]
    dup_gids = [g for g, c in collections.Counter(
        _prov(r).get("source_group_id") for r in records).items() if g and c > 1]

    # exact brief fingerprint duplicates
    fp_seen: Dict[str, str] = {}
    fp_dups: List[str] = []
    for r in records:
        fp = fingerprint(r.get("brief") or {})
        if fp in fp_seen:
            fp_dups.append(r["item_id"])
        else:
            fp_seen[fp] = r["item_id"]

    # normalized goal duplicates (warning)
    goal_map: Dict[str, List[str]] = collections.defaultdict(list)
    for r in records:
        for g in (r.get("brief") or {}).get("goals") or []:
            goal_map[_norm_goal(g)].append(r["item_id"])
    goal_dups = sorted({iid for ids in goal_map.values() if len(ids) > 1 for iid in ids})

    # near-duplicate pairs within pilot (warning); store each unordered pair once.
    pilot_pairs = [(r["item_id"], r.get("brief") or {}) for r in records]
    raw_near = near_duplicate_pairs(pilot_pairs, pilot_pairs, threshold=NEAR_DUP_THRESHOLD)
    seen_pair: set = set()
    near: List[dict] = []
    for p in raw_near:
        if p["left_id"] == p["right_id"]:
            continue
        key = tuple(sorted((p["left_id"], p["right_id"])))
        if key in seen_pair:
            continue
        seen_pair.add(key)
        a, b = key
        near.append({"left_id": a, "right_id": b, "similarity": p["similarity"]})

    checks = [
        _check("no_duplicate_item_ids", not dup_ids, "mandatory", dup_ids),
        _check("no_duplicate_source_record_ids", not dup_srids, "mandatory", dup_srids),
        _check("no_duplicate_source_group_ids", not dup_gids, "mandatory", dup_gids),
        _check("no_exact_duplicate_briefs", not fp_dups, "mandatory", fp_dups, "exact brief fingerprint duplicates"),
        _check("no_normalized_duplicate_goals", not goal_dups, dup_severity, goal_dups, "normalized goal text repeated"),
        _check("near_duplicate_within_pilot", not near, dup_severity,
               detail=f"{len(near)} pairs >= {NEAR_DUP_THRESHOLD} char-3gram Jaccard"),
    ]

    for iid in dup_ids:
        findings.append({"type": "duplicate_item_id", "item_id": iid, "severity": "mandatory"})
    for iid in dup_srids:
        findings.append({"type": "duplicate_source_record_id", "item_id": iid, "severity": "mandatory"})
    for iid in fp_dups:
        findings.append({"type": "duplicate_brief_fingerprint", "item_id": iid, "severity": "mandatory"})
    for iid in goal_dups:
        findings.append({"type": "normalized_duplicate_goal", "item_id": iid, "severity": dup_severity})
    for p in near:
        findings.append({"type": "near_duplicate_pair", "severity": dup_severity,
                         "left_id": p["left_id"], "right_id": p["right_id"], "similarity": p["similarity"]})
    return checks, findings


# --------------------------------------------------------------------------- #
# leakage validation (pilot vs independent evaluation artifacts)
# --------------------------------------------------------------------------- #
def _adapt(records: List[dict], kind: str) -> List[Tuple[str, dict]]:
    out: List[Tuple[str, dict]] = []
    for rec in records:
        if kind == "top":  # brief fields at top level
            out.append((rec.get("item_id", ""), rec))
        elif kind == "nested":  # brief under rec["brief"]
            out.append((rec.get("item_id", ""), rec.get("brief") or {}))
        elif kind == "benchmark":  # flat benchmark record
            out.append((rec.get("benchmark_id", ""), item_brief(rec)))
    return out


def leakage_checks(
    records: List[dict],
    eval_sources: List[Dict[str, Any]],
) -> Tuple[List[dict], List[dict]]:
    """Compare pilot vs each eval source. ``eval_sources`` items:
    ``{"name", "records": [...], "kind": "top"|"nested"|"benchmark", "present": bool}``.
    """
    findings: List[dict] = []
    checks: List[dict] = []

    pilot_ids = {r["item_id"] for r in records}
    pilot_srids = {_prov(r).get("source_record_id") for r in records}
    pilot_fp = {fingerprint(r.get("brief") or {}): r["item_id"] for r in records}
    pilot_pairs = [(r["item_id"], r.get("brief") or {}) for r in records]

    any_id_overlap: List[str] = []
    any_srid_overlap: List[str] = []
    any_fp_overlap: List[str] = []
    any_near: List[dict] = []

    for src in eval_sources:
        name = src["name"]
        if not src.get("present"):
            findings.append({"type": "eval_source_skipped", "source": name, "severity": "info",
                             "detail": "artifact not present; skipped"})
            continue
        pairs = _adapt(src["records"], src["kind"])
        eval_ids = {i for i, _ in pairs}
        eval_fp = {fingerprint(b) for _, b in pairs}

        id_over = sorted(pilot_ids & eval_ids)
        srid_over = sorted({s for s in pilot_srids if s in eval_ids})
        fp_over = sorted({pilot_fp[f] for f in (set(pilot_fp) & eval_fp)})
        near = [p for p in near_duplicate_pairs(pilot_pairs, pairs, threshold=NEAR_DUP_THRESHOLD)]

        any_id_overlap += id_over
        any_srid_overlap += srid_over
        any_fp_overlap += fp_over
        any_near += [{**p, "source": name} for p in near]

        for iid in id_over:
            findings.append({"type": "exact_item_id_overlap", "source": name, "item_id": iid, "severity": "mandatory"})
        for iid in srid_over:
            findings.append({"type": "exact_source_record_overlap", "source": name, "item_id": iid, "severity": "mandatory"})
        for iid in fp_over:
            findings.append({"type": "exact_fingerprint_overlap", "source": name, "item_id": iid, "severity": "mandatory"})
        for p in near:
            findings.append({"type": "near_duplicate", "source": name, "severity": "warning",
                             "left_id": p["left_id"], "right_id": p["right_id"], "similarity": p["similarity"]})

    checks = [
        _check("no_exact_item_id_overlap", not any_id_overlap, "mandatory", any_id_overlap),
        _check("no_exact_source_record_overlap", not any_srid_overlap, "mandatory", any_srid_overlap),
        _check("no_exact_fingerprint_overlap", not any_fp_overlap, "mandatory", any_fp_overlap),
        _check("no_near_duplicate_eval_overlap", not any_near, "warning",
               detail=f"{len(any_near)} near-duplicate pairs >= {NEAR_DUP_THRESHOLD} (manual review)"),
    ]
    return checks, findings


# --------------------------------------------------------------------------- #
# distribution analysis
# --------------------------------------------------------------------------- #
def distribution_rows(records: List[dict]) -> List[Tuple[str, str, int]]:
    """Long-format ``(dimension, value, count)`` rows."""
    counters: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in records:
        prov = _prov(r)
        m = _mapping0(r)
        lin = _lineage(r)
        counters["source_chart_label"][prov.get("original_chart_label", "?")] += 1
        counters["normalized_chart_type"][m.get("chart_type", "?")] += 1
        counters["inferred_task_type"][_task(r).get("task_type", m.get("task_type", "?"))] += 1
        counters["split"][r.get("split", "?")] += 1
        counters["database"][prov.get("db_id", "?")] += 1
        # dtype origin counted per axis (x and y independently typed)
        axis_typing = prov.get("axis_typing") or {}
        for axis in ("x", "y"):
            counters["column_dtype_origin"][(axis_typing.get(axis) or {}).get("dtype_source", "?")] += 1
        counters["grouping_recovery"][(prov.get("grouping") or {}).get("recovery_status", "?")] += 1
        counters["grouped"]["grouped" if (m.get("encoding") or {}).get("grouped") else "non_grouped"] += 1
        for f in _LINEAGE_FIELDS:
            counters["field_lineage"][f"{f}={lin.get(f, '?')}"] += 1

        # Source-constraint coverage: how many accepted records preserve each
        # kind of constraint (filters/sort/time-bin/grouping).
        constraints = prov.get("constraints") or {}
        counters["coverage"]["filters_present" if constraints.get("filters") else "filters_absent"] += 1
        counters["coverage"]["sort_present" if constraints.get("sort") else "sort_absent"] += 1
        counters["coverage"]["time_grain_present" if constraints.get("time_grain") else "time_grain_absent"] += 1
        counters["coverage"]["grouping_present" if (prov.get("grouping") or {}).get("is_grouped") else "grouping_absent"] += 1

    rows: List[Tuple[str, str, int]] = []
    for dim in ("source_chart_label", "normalized_chart_type", "inferred_task_type",
                "split", "database", "field_lineage", "column_dtype_origin",
                "grouping_recovery", "grouped", "coverage"):
        for value, count in sorted(counters[dim].items(), key=lambda kv: (-kv[1], kv[0])):
            rows.append((dim, value, count))
    return rows
