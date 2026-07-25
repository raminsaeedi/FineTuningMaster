"""Phase 2: select the maximum valid Tier-A nvBench corpus.

Consumes Phase 1's enriched output (``nvbench_quality_pool_final/
tier_a_candidates.jsonl``) as the sole source of truth -- no re-tiering, no
rule changes. Every function here operates on that enriched dict shape
directly: ``{item_id, source_group_id, source_record_id, db_id, chart_type,
quality_tier, quality_score, failed_rules, warnings, rule_version, evidence,
record}`` where ``record`` is the original ``{item_id, split, brief,
recommendation}`` builder output.

Selection policy (never weakens a quality rule to hit a quota):
1. Admit every survivable Scatter record (one-per-group, exact-goal-dedup,
   near-dup-aware, leakage-excluded) -- there is no fixed Scatter target here,
   just "take everything that survives".
2. Split the remaining budget 40/20/20/20 across bar/line/pie/stacked_bar
   (bar capped at 50% of the total requested size).
3. If a chart's real supply falls short of its quota, redistribute the deficit
   to charts with remaining untried supply, deterministically, and keep
   iterating until either the target is met or every chart's supply is
   exhausted. Only then does the caller see ``insufficient_unique_tier_a_candidates``.
"""

from __future__ import annotations

import collections
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from src.data_pipeline.builders.leakage import fingerprint
from src.data_pipeline.leakage_similarity import brief_text, char_ngrams, jaccard
from src.data_pipeline.nvbench_extract import extract_base_field
from src.data_pipeline.nvbench_source import parse_aggregate

NORMALIZED_CHART_TYPES = ("bar", "line", "pie", "scatter", "stacked_bar")
NON_SCATTER_CHARTS = ("bar", "line", "pie", "stacked_bar")
_QUOTA_SHARES = {"bar": 0.40, "line": 0.20, "pie": 0.20, "stacked_bar": 0.20}


# --------------------------------------------------------------------------- #
# accessors on the Phase-1 enriched dict shape
# --------------------------------------------------------------------------- #
def _chart_of(rec: Dict[str, Any]) -> str:
    return rec["chart_type"]


def _group_of(rec: Dict[str, Any]) -> str:
    return rec["source_group_id"]


def _db_of(rec: Dict[str, Any]) -> str:
    return rec["db_id"]


def _brief_of(rec: Dict[str, Any]) -> Dict[str, Any]:
    return (rec.get("record") or {}).get("brief") or {}


def _prov_of(rec: Dict[str, Any]) -> Dict[str, Any]:
    return _brief_of(rec).get("extra", {}).get("provenance", {}) or {}


def _mapping0_of(rec: Dict[str, Any]) -> Dict[str, Any]:
    maps = ((rec.get("record") or {}).get("recommendation") or {}).get("kpi_chart_mapping") or []
    return maps[0] if maps else {}


# --------------------------------------------------------------------------- #
# Step 2 -- deterministic semantic signature (2B)
# --------------------------------------------------------------------------- #
def semantic_signature(rec: Dict[str, Any]) -> Tuple[Any, ...]:
    """Analytical-content signature: KPI base field, aggregate function, x/y
    fields, grouping field, filters, sort, time grain, database ID, chart type.

    Deliberately excludes the raw natural-language goal text: two records with
    an identical signature but differently-worded goals are paraphrases (the
    wording-similarity gate in :func:`_second_from_group` catches those
    separately); two records with a *different* signature are analytically
    distinct regardless of how similar their wording happens to be.
    """
    prov = _prov_of(rec)
    m = _mapping0_of(rec)
    enc = m.get("encoding") or {}
    axis_typing = prov.get("axis_typing") or {}
    kpi_sel = prov.get("kpi_selection") or {}
    primary_kpi = kpi_sel.get("primary_kpi") or m.get("kpi") or ""
    agg_func = parse_aggregate(primary_kpi)
    base_field = extract_base_field(primary_kpi) if agg_func else primary_kpi

    x = (axis_typing.get("x") or {}).get("name")
    y = (axis_typing.get("y") or {}).get("name")
    group_field = enc.get("group_field")

    constraints = prov.get("constraints") or {}
    filters = tuple(sorted(
        (f.get("field"), f.get("operator"), f.get("value")) for f in (constraints.get("filters") or [])
    ))
    sort = constraints.get("sort")
    sort_tuple = (sort.get("field"), sort.get("direction")) if sort else None
    tg = constraints.get("time_grain")
    tg_tuple = (tg.get("field"), tg.get("grain")) if tg else None

    return (rec.get("chart_type"), base_field, agg_func, x, y, group_field, filters, sort_tuple, tg_tuple,
           rec.get("db_id"))


_SIGNATURE_LABELS = ("chart_type", "kpi_base_field", "aggregate", "x_field", "y_field",
                    "group_field", "filters", "sort", "time_grain", "db_id")


def signature_diff(sig_a: Tuple[Any, ...], sig_b: Tuple[Any, ...]) -> List[str]:
    """Human-readable list of components that differ between two signatures."""
    return [label for label, a, b in zip(_SIGNATURE_LABELS, sig_a, sig_b) if a != b]


def _norm_goal(rec: Dict[str, Any]) -> str:
    goals = _brief_of(rec).get("goals") or [""]
    return " ".join(str(goals[0]).strip().lower().split())


def _hash_key(seed: int, item_id: str) -> str:
    return hashlib.md5(f"{seed}:{item_id}".encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Step 1 -- validate Phase-1 input
# --------------------------------------------------------------------------- #
def validate_phase1_input(manifest: Dict[str, Any], hashes: Dict[str, str],
                          recomputed_hashes: Dict[str, str], tier_a_records: List[Dict[str, Any]]) -> List[dict]:
    """Consistency checks on the Phase-1 quality-pool-final output.

    Returns a list of ``{check, passed, severity, n, item_ids, detail}`` dicts,
    the same shape used throughout the nvBench pipeline's reports.
    """
    def _c(name, ids, detail, severity="mandatory"):
        ids = sorted(set(ids))
        return {"check": name, "passed": not ids, "severity": severity, "n": len(ids), "item_ids": ids, "detail": detail}

    checks = []
    checks.append({"check": "phase1_status_pass", "passed": bool(manifest.get("passed")), "severity": "mandatory",
                   "n": 0 if manifest.get("passed") else 1, "item_ids": [],
                   "detail": f"Phase-1 manifest.passed={manifest.get('passed')}"})

    hash_mismatches = [k for k in ("tier_a_candidates", "quality_pool_summary")
                       if hashes.get(k) != recomputed_hashes.get(k)]
    checks.append({"check": "phase1_hashes_valid", "passed": not hash_mismatches, "severity": "mandatory",
                   "n": len(hash_mismatches), "item_ids": [],
                   "detail": f"mismatched files: {hash_mismatches}" if hash_mismatches else "all hashes match"})

    below_min = [r["item_id"] for r in tier_a_records if r.get("quality_score", 0) < 90]
    checks.append(_c("phase1_all_tier_a_score_at_least_90", below_min, "Tier-A record scored below 90"))

    mandatory_fail = [r["item_id"] for r in tier_a_records if r.get("failed_rules")]
    checks.append(_c("phase1_no_tier_a_mandatory_failure", mandatory_fail,
                     "Tier-A record has a non-empty failed_rules list"))

    bad_pie = []
    for r in tier_a_records:
        if _chart_of(r) != "pie":
            continue
        axis_typing = ((r.get("record") or {}).get("brief") or {}).get("extra", {}).get("provenance", {}).get("axis_typing", {})
        y_agg = (axis_typing.get("y") or {}).get("aggregate")
        if y_agg and y_agg.upper() not in ("COUNT", "SUM"):
            bad_pie.append(r["item_id"])
    checks.append(_c("phase1_no_tier_a_pie_avg_min_max", bad_pie, "Tier-A pie uses a non-additive aggregate"))

    return checks


# --------------------------------------------------------------------------- #
# Step 2 -- real availability (before any selection is attempted)
# --------------------------------------------------------------------------- #
def _leakage_banned_ids(eval_sources: List[Dict[str, Any]]) -> Tuple[set, set]:
    """Exact item_id/source_record_id and brief-fingerprint sets from eval sources."""
    from src.data_pipeline.nvbench_pilot import _adapt  # reuse existing adapter

    banned_ids: set = set()
    banned_fps: set = set()
    for src in eval_sources:
        if not src.get("present"):
            continue
        for iid, brief in _adapt(src["records"], src["kind"]):
            if iid:
                banned_ids.add(iid)
            banned_fps.add(fingerprint(brief))
    return banned_ids, banned_fps


def compute_availability(
    tier_a_records: List[Dict[str, Any]],
    eval_sources: List[Dict[str, Any]],
    *,
    seed: int = 42,
    near_dup_threshold: float = 0.8,
) -> Dict[str, Any]:
    """Per-chart real availability: records, unique groups, unique databases,
    exact-goal-dedup losses, group-uniqueness losses, near-dup losses (computed
    by actually running the chart-local admission, not estimated), and
    eval-leakage exclusions. Never assumes a distribution before measuring it.
    """
    banned_ids, banned_fps = _leakage_banned_ids(eval_sources)

    def excluded_by_leakage(rec) -> bool:
        r = rec.get("record") or {}
        if rec["item_id"] in banned_ids or rec.get("source_record_id") in banned_ids:
            return True
        return fingerprint(_brief_of(rec)) in banned_fps

    by_chart: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in tier_a_records:
        by_chart[_chart_of(r)].append(r)

    availability: Dict[str, Any] = {}
    for chart in NORMALIZED_CHART_TYPES:
        recs = by_chart.get(chart, [])
        n_records = len(recs)
        n_leaked = sum(1 for r in recs if excluded_by_leakage(r))
        surviving = [r for r in recs if not excluded_by_leakage(r)]

        groups: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in surviving:
            groups[_group_of(r)].append(r)
        n_unique_groups = len(groups)
        n_dropped_by_group_uniqueness = len(surviving) - n_unique_groups

        one_per_group = []
        for cluster in groups.values():
            cluster.sort(key=lambda r: _hash_key(seed, r["item_id"]))
            one_per_group.append(cluster[0])

        goal_clusters: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in one_per_group:
            goal_clusters[_norm_goal(r)].append(r)
        deduped = []
        n_dropped_exact_goal = 0
        for cluster in goal_clusters.values():
            cluster.sort(key=lambda r: _hash_key(seed, r["item_id"]))
            deduped.append(cluster[0])
            n_dropped_exact_goal += len(cluster) - 1
        deduped.sort(key=lambda r: r["item_id"])

        # Chart-local near-dup simulation (true max survivable within this
        # chart alone, ignoring cross-chart interaction -- selection itself
        # checks near-dup globally across all admitted items).
        admitted_ngrams: List[frozenset] = []
        n_near_dup_dropped = 0
        survivable = 0
        for r in sorted(deduped, key=lambda r: _hash_key(seed, r["item_id"])):
            ngrams = frozenset(char_ngrams(brief_text(_brief_of(r))))
            if any(jaccard(ngrams, other) >= near_dup_threshold for other in admitted_ngrams):
                n_near_dup_dropped += 1
                continue
            admitted_ngrams.append(ngrams)
            survivable += 1

        unique_databases = len({_db_of(r) for r in recs})
        availability[chart] = {
            "tier_a_records": n_records,
            "unique_source_groups": n_unique_groups,
            "unique_databases": unique_databases,
            "excluded_by_eval_leakage": n_leaked,
            "dropped_by_group_uniqueness": n_dropped_by_group_uniqueness,
            "dropped_by_exact_goal_dedup": n_dropped_exact_goal,
            "dropped_by_near_duplicate": n_near_dup_dropped,
            "max_survivable_chart_local": survivable,
        }
    return availability


# --------------------------------------------------------------------------- #
# Step 3 -- deterministic selection with redistribution
# --------------------------------------------------------------------------- #
def _quota_split(remainder: int, bar_cap: Optional[int] = None) -> Dict[str, int]:
    """Largest-remainder 40/20/20/20 split of ``remainder`` across the 4
    non-Scatter chart types, with bar capped at ``bar_cap`` if given."""
    raw = {c: remainder * share for c, share in _QUOTA_SHARES.items()}
    base = {c: int(v) for c, v in raw.items()}
    used = sum(base.values())
    leftover = remainder - used
    # Largest-remainder method, alphabetical tie-break for determinism.
    order = sorted(NON_SCATTER_CHARTS, key=lambda c: (-(raw[c] - base[c]), c))
    for c in order[:leftover]:
        base[c] += 1
    if bar_cap is not None and base["bar"] > bar_cap:
        overflow = base["bar"] - bar_cap
        base["bar"] = bar_cap
        # Redistribute the overflow evenly (largest-remainder again) across
        # the other three -- deterministic, never dropped silently.
        others = [c for c in NON_SCATTER_CHARTS if c != "bar"]
        add, extra = divmod(overflow, len(others))
        for c in others:
            base[c] += add
        for c in sorted(others)[:extra]:
            base[c] += 1
    return base


def select_large_v1(
    tier_a_records: List[Dict[str, Any]],
    eval_sources: List[Dict[str, Any]],
    *,
    seed: int = 42,
    total: int = 2000,
    db_cap: int = 100,
    near_dup_threshold: float = 0.8,
    max_per_group: int = 1,
    minimum_acceptable: Optional[int] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    """Select up to ``total`` records per the policy in the module docstring.

    ``max_per_group`` (1 by default; Phase 2B passes 2) allows a second record
    from the same source group only when it clears BOTH gates: its normalized
    goal is NOT a near-duplicate of the first-selected record's goal (Jaccard
    <= ``near_dup_threshold`` -- so a sort-direction variant with 95%-identical
    wording still fails here), AND its :func:`semantic_signature` genuinely
    differs (KPI/aggregate/axis/grouping/filter/sort/time-grain) -- so passing
    the wording gate alone is never sufficient. At most one secondary record is
    ever taken per group, in deterministic hash order.

    ``minimum_acceptable`` supports Phase 2C's documented maximum-valid-corpus
    policy. When the preferred ``total`` cannot be reached but the deterministic
    selection reaches that minimum, the maximum valid selection is returned
    without changing any quality, duplicate, leakage, database-cap, or
    source-group rule. The default remains strict target-or-fail behavior.

    Returns ``(selected_or_None, report)``. ``selected`` is a list of the
    Phase-1 enriched dicts (never modified). On insufficient supply, returns
    ``(None, report)`` with ``report["status"] == "insufficient_unique_tier_a_candidates"``
    and a full per-chart breakdown of what was achievable -- never a Tier-B
    fallback, never a relaxed near-dup threshold, never fewer than the
    requested unique-group guarantee silently accepted as success.
    """
    # Defense in depth: never draw from anything but Tier A, even if the
    # caller's input list is contaminated -- this function alone must never be
    # the reason a Tier-B record ends up in the dataset.
    tier_a_records = [r for r in tier_a_records if r.get("quality_tier") == "A"]

    banned_ids, banned_fps = _leakage_banned_ids(eval_sources)

    def excluded_by_leakage(rec) -> bool:
        if rec["item_id"] in banned_ids or rec.get("source_record_id") in banned_ids:
            return True
        return fingerprint(_brief_of(rec)) in banned_fps

    by_chart: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in tier_a_records:
        by_chart[_chart_of(r)].append(r)

    multi_record_evidence: List[Dict[str, Any]] = []

    def group_selection(cluster: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cluster = sorted(cluster, key=lambda r: _hash_key(seed, r["item_id"]))
        primary = cluster[0]
        chosen = [primary]
        if max_per_group >= 2 and len(cluster) > 1:
            primary_ngrams = frozenset(char_ngrams(brief_text(_brief_of(primary))))
            primary_sig = semantic_signature(primary)
            for cand in cluster[1:]:
                cand_ngrams = frozenset(char_ngrams(brief_text(_brief_of(cand))))
                sim = jaccard(primary_ngrams, cand_ngrams)
                if sim > near_dup_threshold:
                    continue  # too similar in wording -- a paraphrase/sort-variant, not distinct
                cand_sig = semantic_signature(cand)
                if cand_sig == primary_sig:
                    continue  # identical analytical content regardless of wording -- reject
                diff = signature_diff(primary_sig, cand_sig)
                chosen.append(cand)
                multi_record_evidence.append({
                    "source_group_id": _group_of(primary),
                    "item_ids": [primary["item_id"], cand["item_id"]],
                    "normalized_goals": [_norm_goal(primary), _norm_goal(cand)],
                    "signatures": [str(primary_sig), str(cand_sig)],
                    "differing_components": diff,
                    "goal_similarity": round(sim, 4),
                    "justification": (
                        f"differs in {', '.join(diff) or 'no field'}; "
                        f"goal-text similarity {sim:.3f} <= threshold {near_dup_threshold}"
                    ),
                })
                break  # at most one secondary per group
        return chosen

    def chart_pool(chart: str) -> List[Dict[str, Any]]:
        """Up to max_per_group per group -> exact-goal-dedup -> leakage-excluded -> hash-sorted."""
        recs = [r for r in by_chart.get(chart, []) if not excluded_by_leakage(r)]
        groups: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in recs:
            groups[_group_of(r)].append(r)
        from_groups: List[Dict[str, Any]] = []
        for cluster in groups.values():
            from_groups.extend(group_selection(cluster))
        goal_clusters: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in from_groups:
            goal_clusters[_norm_goal(r)].append(r)
        deduped = []
        for cluster in goal_clusters.values():
            cluster.sort(key=lambda r: _hash_key(seed, r["item_id"]))
            deduped.append(cluster[0])
        return sorted(deduped, key=lambda r: _hash_key(seed, r["item_id"]))

    pools = {c: chart_pool(c) for c in NORMALIZED_CHART_TYPES}

    admitted: List[Dict[str, Any]] = []
    admitted_ids: set = set()
    admitted_goals: set = set()
    admitted_ngrams: List[frozenset] = []
    db_counts: Dict[str, int] = collections.defaultdict(int)
    db_cap_skipped: int = 0

    def try_admit(rec) -> bool:
        nonlocal db_cap_skipped
        if rec["item_id"] in admitted_ids:
            return False
        if db_counts[_db_of(rec)] >= db_cap:
            db_cap_skipped += 1
            return False
        normalized_goal = _norm_goal(rec)
        if normalized_goal and normalized_goal in admitted_goals:
            return False
        ngrams = frozenset(char_ngrams(brief_text(_brief_of(rec))))
        if any(jaccard(ngrams, other) >= near_dup_threshold for other in admitted_ngrams):
            return False
        admitted.append(rec)
        admitted_ids.add(rec["item_id"])
        if normalized_goal:
            admitted_goals.add(normalized_goal)
        admitted_ngrams.append(ngrams)
        db_counts[_db_of(rec)] += 1
        return True

    # Step 1: admit every survivable Scatter record -- no fixed target.
    scatter_pos = 0
    for rec in pools["scatter"]:
        scatter_pos += 1
        try_admit(rec)
    n_scatter_admitted = sum(1 for r in admitted if _chart_of(r) == "scatter")

    remainder = max(0, total - n_scatter_admitted)
    quotas = _quota_split(remainder, bar_cap=int(total * 0.5))
    nominal_quotas = dict(quotas)  # the 40/20/20/20 split before any redistribution
    bucket_pos = {c: 0 for c in NON_SCATTER_CHARTS}
    admitted_counts = {c: 0 for c in NON_SCATTER_CHARTS}

    while True:
        for chart in NON_SCATTER_CHARTS:
            pool = pools[chart]
            while admitted_counts[chart] < quotas[chart] and bucket_pos[chart] < len(pool):
                rec = pool[bucket_pos[chart]]
                bucket_pos[chart] += 1
                if try_admit(rec):
                    admitted_counts[chart] += 1

        # Freeze any chart whose pool is now fully exhausted below its quota:
        # its deficit is redistributed exactly once here, never re-injected in
        # a later round (that was the bug -- recomputing shortfall from the
        # still-short-but-permanently-stuck chart every iteration caused it to
        # be "redistributed" over and over, wildly over-admitting elsewhere).
        newly_frozen_deficit = 0
        for chart in NON_SCATTER_CHARTS:
            if bucket_pos[chart] >= len(pools[chart]) and admitted_counts[chart] < quotas[chart]:
                newly_frozen_deficit += quotas[chart] - admitted_counts[chart]
                quotas[chart] = admitted_counts[chart]

        if newly_frozen_deficit == 0:
            break  # every chart either met its quota or still has untried supply -- done

        donors = [c for c in NON_SCATTER_CHARTS if bucket_pos[c] < len(pools[c])]
        if not donors:
            break  # every chart's pool is fully exhausted; cannot redistribute further
        remaining_shortfall = newly_frozen_deficit
        idx = 0
        while remaining_shortfall > 0:
            c = donors[idx % len(donors)]
            quotas[c] += 1
            remaining_shortfall -= 1
            idx += 1

    admitted.sort(key=lambda r: r["item_id"])
    achieved_per_chart = {c: admitted_counts[c] for c in NON_SCATTER_CHARTS}
    achieved_per_chart["scatter"] = n_scatter_admitted

    admitted_ids_final = {r["item_id"] for r in admitted}
    kept_multi_record_evidence = [
        e for e in multi_record_evidence
        if e["item_ids"][0] in admitted_ids_final and e["item_ids"][1] in admitted_ids_final
    ]
    admitted_group_counts = collections.Counter(_group_of(r) for r in admitted)
    groups_with_one_record = sum(1 for c in admitted_group_counts.values() if c == 1)
    groups_with_two_records = sum(1 for c in admitted_group_counts.values() if c == 2)

    report = {
        "requested_total": total,
        "achieved_total": len(admitted),
        "requested_distribution": {**{c: quotas[c] for c in NON_SCATTER_CHARTS}, "scatter": n_scatter_admitted},
        "nominal_requested_distribution": {**nominal_quotas, "scatter": n_scatter_admitted},
        "achieved_distribution": achieved_per_chart,
        "max_achievable_per_chart": {c: len(pools[c]) for c in NORMALIZED_CHART_TYPES},
        "db_cap": db_cap,
        "db_cap_skipped_admissions": db_cap_skipped,
        "near_dup_threshold": near_dup_threshold,
        "max_per_group": max_per_group,
        "seed": seed,
        "preferred_target": total,
        "minimum_acceptable": minimum_acceptable,
        "unique_source_groups_selected": len(admitted_group_counts),
        "groups_with_one_record": groups_with_one_record,
        "groups_with_two_records": groups_with_two_records,
        "multi_record_groups": kept_multi_record_evidence,
    }

    if len(admitted) < total:
        report["deficit"] = total - len(admitted)
        if minimum_acceptable is not None and len(admitted) >= minimum_acceptable:
            report["status"] = "maximum_valid_corpus_accepted"
            return admitted, report
        report["status"] = "insufficient_unique_tier_a_candidates"
        return None, report

    report["status"] = "ok"
    return admitted, report


# --------------------------------------------------------------------------- #
# Step 4 -- deterministic, chart-stratified train/validation split
# --------------------------------------------------------------------------- #
def split_train_val(
    selected: List[Dict[str, Any]], *, seed: int = 42, val_fraction: float = 0.10,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Group-aware 90/10 split: the split unit is the SOURCE GROUP, not the
    record. When two records share a group (Phase 2B's controlled two-per-group
    policy), both are always assigned to the same split -- this is mandatory
    even though it means a two-record group counts as one unit for the ratio.
    Stratified by chart type (each chart's groups hashed and split
    independently) so the chart-type ratio is preserved in both splits.
    """
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    per_chart_report: Dict[str, Any] = {}

    by_chart_groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for r in selected:
        by_chart_groups[_chart_of(r)][_group_of(r)].append(r)

    for chart, groups in by_chart_groups.items():
        group_ids = sorted(groups, key=lambda gid: _hash_key(f"{seed}:split", gid))
        n_val_groups = round(len(group_ids) * val_fraction)
        val_group_ids = set(group_ids[:n_val_groups])
        chart_train = chart_val = 0
        for gid in group_ids:
            recs = groups[gid]
            if gid in val_group_ids:
                val.extend(recs)
                chart_val += len(recs)
            else:
                train.extend(recs)
                chart_train += len(recs)
        per_chart_report[chart] = {
            "train": chart_train, "val": chart_val, "total": chart_train + chart_val,
            "train_groups": len(group_ids) - len(val_group_ids), "val_groups": len(val_group_ids),
        }

    train.sort(key=lambda r: r["item_id"])
    val.sort(key=lambda r: r["item_id"])

    train_groups = {_group_of(r) for r in train}
    val_groups = {_group_of(r) for r in val}
    report = {
        "seed": seed, "val_fraction": val_fraction, "split_algorithm_version": "nvbench_large_v1_split_v2_group_safe",
        "train_count": len(train), "val_count": len(val),
        "per_chart": per_chart_report,
        "cross_split_group_overlap": sorted(train_groups & val_groups),
    }
    return train, val, report


# --------------------------------------------------------------------------- #
# Phase 2C -- deterministic group-aware 70/15/15 train/val/test split
# --------------------------------------------------------------------------- #
def split_train_val_test(
    selected: List[Dict[str, Any]], *, seed: int = 42, val_fraction: float = 0.15, test_fraction: float = 0.15,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Deterministic source-group-aware train/validation/test split.

    Groups are formed globally before chart stratification, so even a source
    group containing multiple chart types cannot straddle splits. The dominant
    chart places each group in one deterministic stratum; exact row ratios stay
    secondary to source-group integrity.
    """
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be non-negative and sum to less than 1")

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    all_groups: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for rec in selected:
        all_groups[_group_of(rec)].append(rec)

    by_chart_groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = collections.defaultdict(dict)
    for group_id, records in all_groups.items():
        chart_counts = collections.Counter(_chart_of(rec) for rec in records)
        stratum = sorted(chart_counts, key=lambda chart: (-chart_counts[chart], chart))[0]
        by_chart_groups[stratum][group_id] = records

    for groups in by_chart_groups.values():
        group_ids = sorted(groups, key=lambda group_id: _hash_key(f"{seed}:split3", group_id))
        n_groups = len(group_ids)
        n_test_groups = round(n_groups * test_fraction)
        n_val_groups = round(n_groups * val_fraction)
        test_ids = set(group_ids[:n_test_groups])
        val_ids = set(group_ids[n_test_groups:n_test_groups + n_val_groups])
        for group_id in group_ids:
            records = groups[group_id]
            if group_id in test_ids:
                test.extend(records)
            elif group_id in val_ids:
                val.extend(records)
            else:
                train.extend(records)

    train.sort(key=lambda rec: rec["item_id"])
    val.sort(key=lambda rec: rec["item_id"])
    test.sort(key=lambda rec: rec["item_id"])

    buckets = {"train": train, "val": val, "test": test}
    split_groups = {
        name: {_group_of(rec) for rec in records}
        for name, records in buckets.items()
    }
    charts = sorted({_chart_of(rec) for rec in selected})
    databases = sorted({_db_of(rec) for rec in selected})
    per_chart_report = {}
    for chart in charts:
        counts = {
            name: sum(1 for rec in records if _chart_of(rec) == chart)
            for name, records in buckets.items()
        }
        group_counts = {
            name: len({_group_of(rec) for rec in records if _chart_of(rec) == chart})
            for name, records in buckets.items()
        }
        per_chart_report[chart] = {
            **counts,
            "total": sum(counts.values()),
            "train_groups": group_counts["train"],
            "val_groups": group_counts["val"],
            "test_groups": group_counts["test"],
        }
    per_database_report = {
        database: {
            name: sum(1 for rec in records if _db_of(rec) == database)
            for name, records in buckets.items()
        }
        for database in databases
    }

    total = len(selected)
    overlap = (
        (split_groups["train"] & split_groups["val"])
        | (split_groups["train"] & split_groups["test"])
        | (split_groups["val"] & split_groups["test"])
    )
    report = {
        "seed": seed,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "split_algorithm_version": "nvbench_large_v1_split_v3_train_val_test",
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "actual_percentages": {
            name: round(100 * len(records) / total, 6) if total else 0.0
            for name, records in buckets.items()
        },
        "unique_source_groups": {
            name: len(groups) for name, groups in split_groups.items()
        },
        "per_chart": per_chart_report,
        "per_database": per_database_report,
        "cross_split_group_overlap": sorted(overlap),
        "test_membership_sha256": hashlib.sha256(
            "\n".join(rec["item_id"] for rec in test).encode("utf-8")
        ).hexdigest(),
    }
    return train, val, test, report

# --------------------------------------------------------------------------- #
# Step 7 -- deterministic 30-item stratified spot-check sample
# --------------------------------------------------------------------------- #
def _has_filters(rec: Dict[str, Any]) -> bool:
    prov = (rec.get("record") or {}).get("brief", {}).get("extra", {}).get("provenance", {})
    return bool((prov.get("constraints") or {}).get("filters"))


def _has_sort(rec: Dict[str, Any]) -> bool:
    prov = (rec.get("record") or {}).get("brief", {}).get("extra", {}).get("provenance", {})
    return bool((prov.get("constraints") or {}).get("sort"))


def _has_grouping(rec: Dict[str, Any]) -> bool:
    prov = (rec.get("record") or {}).get("brief", {}).get("extra", {}).get("provenance", {})
    return bool((prov.get("grouping") or {}).get("is_grouped"))


def _has_time_grain(rec: Dict[str, Any]) -> bool:
    prov = (rec.get("record") or {}).get("brief", {}).get("extra", {}).get("provenance", {})
    return bool((prov.get("constraints") or {}).get("time_grain"))


def select_spotcheck_sample(
    selected: List[Dict[str, Any]], *, seed: int = 42, size: int = 30,
) -> List[Dict[str, Any]]:
    """Deterministic stratified sample covering chart types, multiple databases,
    filters/sort/grouping/time-grain presence, score range, and Scatter coverage.
    """
    ordered_all = sorted(selected, key=lambda r: _hash_key(f"{seed}:spot", r["item_id"]))
    chosen: List[Dict[str, Any]] = []
    chosen_ids: set = set()

    def add(rec) -> None:
        if rec["item_id"] not in chosen_ids and len(chosen) < size:
            chosen.append(rec)
            chosen_ids.add(rec["item_id"])

    scatter = [r for r in ordered_all if _chart_of(r) == "scatter"]
    if len(scatter) <= 10:
        for r in scatter:
            add(r)
    else:
        by_score = sorted(scatter, key=lambda r: (r.get("quality_score", 0), r["item_id"]))
        step = max(1, len(by_score) // 10)
        for r in by_score[::step][:10]:
            add(r)

    # Coverage slots: one example each of filters / sort / grouping / time_grain,
    # plus the lowest- and highest-scoring Tier-A records.
    by_score_all = sorted(ordered_all, key=lambda r: (r.get("quality_score", 0), r["item_id"]))
    if by_score_all:
        add(by_score_all[0])
        add(by_score_all[-1])
    for predicate in (_has_filters, _has_sort, _has_grouping, _has_time_grain):
        match = next((r for r in ordered_all if predicate(r)), None)
        if match:
            add(match)

    # Round-robin across the remaining (non-scatter) chart types for broad
    # coverage, preferring databases not yet represented in the sample.
    remaining_charts = [c for c in NON_SCATTER_CHARTS]
    idx = 0
    seen_dbs: set = {_db_of(r) for r in chosen}
    per_chart_pool = {c: [r for r in ordered_all if _chart_of(r) == c] for c in remaining_charts}
    per_chart_ptr = {c: 0 for c in remaining_charts}
    while len(chosen) < size:
        progressed = False
        for _ in range(len(remaining_charts)):
            chart = remaining_charts[idx % len(remaining_charts)]
            idx += 1
            pool = per_chart_pool[chart]
            ptr = per_chart_ptr[chart]
            # Prefer a candidate from an unseen database first.
            pick = None
            for j in range(ptr, len(pool)):
                cand = pool[j]
                if cand["item_id"] in chosen_ids:
                    continue
                if _db_of(cand) not in seen_dbs:
                    pick = (j, cand)
                    break
            if pick is None:
                for j in range(ptr, len(pool)):
                    cand = pool[j]
                    if cand["item_id"] not in chosen_ids:
                        pick = (j, cand)
                        break
            if pick is not None:
                j, cand = pick
                per_chart_ptr[chart] = j + 1
                seen_dbs.add(_db_of(cand))
                add(cand)
                progressed = True
                if len(chosen) >= size:
                    break
        if not progressed:
            break

    chosen.sort(key=lambda r: r["item_id"])
    return chosen[:size]


# --------------------------------------------------------------------------- #
# Phase 2C -- deterministic 40-item stratified human-eval subset (test split only)
# --------------------------------------------------------------------------- #
def select_human_eval_sample(
    test_records: List[Dict[str, Any]], *, seed: int = 42, size: int = 40,
) -> List[Dict[str, Any]]:
    """Like :func:`select_spotcheck_sample`, plus explicit coverage of both
    one-record and two-record source groups (Phase 2B's controlled pairing
    policy must be visible in the human-eval input set, not just the pilot).
    Input briefs and source evidence only -- no ratings, no model outputs.
    """
    ordered_all = sorted(test_records, key=lambda r: _hash_key(f"{seed}:humaneval", r["item_id"]))
    chosen: List[Dict[str, Any]] = []
    chosen_ids: set = set()

    def add(rec) -> None:
        if rec["item_id"] not in chosen_ids and len(chosen) < size:
            chosen.append(rec)
            chosen_ids.add(rec["item_id"])

    scatter = [r for r in ordered_all if _chart_of(r) == "scatter"]
    if len(scatter) <= 10:
        for r in scatter:
            add(r)
    else:
        by_score = sorted(scatter, key=lambda r: (r.get("quality_score", 0), r["item_id"]))
        step = max(1, len(by_score) // 10)
        for r in by_score[::step][:10]:
            add(r)

    group_counts = collections.Counter(_group_of(r) for r in ordered_all)
    two_record_group = next((g for g, c in group_counts.items() if c == 2), None)
    if two_record_group:
        for r in ordered_all:
            if _group_of(r) == two_record_group:
                add(r)
    one_record_group = next((g for g, c in group_counts.items() if c == 1), None)
    if one_record_group:
        match = next((r for r in ordered_all if _group_of(r) == one_record_group), None)
        if match:
            add(match)

    by_score_all = sorted(ordered_all, key=lambda r: (r.get("quality_score", 0), r["item_id"]))
    if by_score_all:
        add(by_score_all[0])
        add(by_score_all[-1])
    for predicate in (_has_filters, _has_sort, _has_grouping, _has_time_grain):
        match = next((r for r in ordered_all if predicate(r)), None)
        if match:
            add(match)

    remaining_charts = [c for c in NON_SCATTER_CHARTS]
    idx = 0
    seen_dbs: set = {_db_of(r) for r in chosen}
    per_chart_pool = {c: [r for r in ordered_all if _chart_of(r) == c] for c in remaining_charts}
    per_chart_ptr = {c: 0 for c in remaining_charts}
    while len(chosen) < size:
        progressed = False
        for _ in range(len(remaining_charts)):
            chart = remaining_charts[idx % len(remaining_charts)]
            idx += 1
            pool = per_chart_pool[chart]
            ptr = per_chart_ptr[chart]
            pick = None
            for j in range(ptr, len(pool)):
                cand = pool[j]
                if cand["item_id"] in chosen_ids:
                    continue
                if _db_of(cand) not in seen_dbs:
                    pick = (j, cand)
                    break
            if pick is None:
                for j in range(ptr, len(pool)):
                    cand = pool[j]
                    if cand["item_id"] not in chosen_ids:
                        pick = (j, cand)
                        break
            if pick is not None:
                j, cand = pick
                per_chart_ptr[chart] = j + 1
                seen_dbs.add(_db_of(cand))
                add(cand)
                progressed = True
                if len(chosen) >= size:
                    break
        if not progressed:
            break

    # Complete small or unusual chart mixtures without weakening prior coverage.
    for rec in ordered_all:
        add(rec)

    chosen.sort(key=lambda r: r["item_id"])
    return chosen[:size]
