"""Pilot v6: quality-constrained, unbalanced 100-record sampling.

Pilot v5's quality gate demoted enough scatter candidates (mixed-aggregate KPI
ambiguity + near-duplicate exclusion) that only ~10 unique, high-confidence
scatter groups remain in the whole nvBench corpus -- verified below, not
assumed. Rather than reject the pilot outright (v5's correct behavior under a
strict 20-per-chart contract) or weaken the quality gate to manufacture more
scatter candidates, this module replaces the *balance* requirement with a
quality-constrained, deterministic per-chart target: take every available
high-confidence scatter group (capped at 10) and redistribute the remaining
budget evenly across the other four chart types, so the pilot still totals
exactly 100 Tier-A records with zero degradation of quality rules.

The admission algorithm itself is unchanged from ``select_pilot_v3``/
``select_pilot_v4`` (one-per-group -> exact-goal-dedup -> near-duplicate-aware,
database-capped admission) -- only the *target* is now a per-chart dict instead
of one scalar shared by every chart, so this cannot be expressed by reusing
those functions unmodified and is implemented here as their generalized sibling.
"""

from __future__ import annotations

import collections
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from src.data_pipeline.leakage_similarity import brief_text, char_ngrams, jaccard
from src.data_pipeline.nvbench_pilot_v5 import _corpus_quality_metrics_v5
from src.data_pipeline.nvbench_source import item_chart, select_one_per_group

NORMALIZED_CHART_TYPES = ("bar", "line", "pie", "scatter", "stacked_bar")
DEFAULT_TOTAL = 100


def _hash_key(seed: int, item_id: str) -> str:
    return hashlib.md5(f"{seed}:{item_id}".encode("utf-8")).hexdigest()


def _norm_goal_text(item) -> str:
    goals = item.brief.goals or [""]
    return " ".join(str(goals[0]).strip().lower().split())


def compute_target_distribution(
    tier_a_items: List[Any],
    *,
    seed: int = 42,
    total: int = DEFAULT_TOTAL,
    reserved_chart: str = "scatter",
    reserved_cap: int = 10,
    other_charts: Tuple[str, ...] = ("bar", "line", "pie", "stacked_bar"),
) -> Dict[str, int]:
    """Deterministically derive the per-chart target: reserve up to
    ``reserved_cap`` for ``reserved_chart`` from its actual available unique
    source-group count (never more than what genuinely exists), then split the
    remainder as evenly as possible across ``other_charts`` (remainder-first,
    alphabetical, for a stable tie-break).
    """
    pool = select_one_per_group(tier_a_items, seed=seed)
    available = collections.Counter(item_chart(it) for it in pool)
    reserved = min(reserved_cap, available.get(reserved_chart, 0))
    remaining = total - reserved
    n_other = len(other_charts)
    base, extra = divmod(remaining, n_other)
    target = {c: base for c in other_charts}
    for c in sorted(other_charts)[:extra]:
        target[c] += 1
    target[reserved_chart] = reserved
    return target


def select_pilot_v6(
    tier_a_items: List[Any],
    *,
    seed: int = 42,
    target_per_chart: Dict[str, int],
    db_cap: int = 10,
    near_dup_threshold: float = 0.8,
) -> Tuple[Optional[List[Any]], Dict[str, Any]]:
    """Quality-constrained sampler: per-chart targets instead of one uniform target.

    Same admission contract as ``select_pilot_v3``: one query per source group,
    exact normalized-goal dedup, near-duplicate exclusion (Jaccard >=
    ``near_dup_threshold``), and a hard per-database cap -- but with an explicit,
    non-uniform ``target_per_chart``. Never falls back over the database cap and
    never admits a chart short of its recorded target: returns ``(None, report)``
    with a diagnosable ``report["status"]`` in either case.
    """
    pool = select_one_per_group(tier_a_items, seed=seed)
    availability_raw: Dict[str, int] = dict(collections.Counter(item_chart(it) for it in tier_a_items))
    unique_groups_per_chart: Dict[str, int] = dict(collections.Counter(item_chart(it) for it in pool))

    clusters: Dict[str, List[Any]] = collections.defaultdict(list)
    for it in pool:
        clusters[_norm_goal_text(it)].append(it)
    deduped = []
    dropped_exact_goal_dups_by_chart: Dict[str, int] = collections.defaultdict(int)
    for cluster in clusters.values():
        cluster.sort(key=lambda it: _hash_key(seed, it.item_id))
        deduped.append(cluster[0])
        if len(cluster) > 1:
            dropped_exact_goal_dups_by_chart[item_chart(cluster[0])] += len(cluster) - 1
    deduped.sort(key=lambda it: it.item_id)

    charts = sorted(target_per_chart)
    buckets = {c: sorted([it for it in deduped if item_chart(it) == c],
                         key=lambda it: _hash_key(seed, it.item_id)) for c in charts}

    admitted: List[Any] = []
    admitted_ids: set = set()
    admitted_ngrams: List[frozenset] = []
    db_counts: Dict[str, int] = collections.defaultdict(int)
    chart_counts: Dict[str, int] = {c: 0 for c in charts}
    near_dup_removed_by_chart: Dict[str, int] = collections.defaultdict(int)
    fallbacks: List[Dict[str, Any]] = []

    def db_of(it) -> str:
        return it.brief.extra.get("provenance", {}).get("db_id", "?")

    def is_near_dup(ngrams: frozenset) -> bool:
        return any(jaccard(ngrams, other) >= near_dup_threshold for other in admitted_ngrams)

    def try_admit(it, *, respect_cap: bool) -> str:
        if it.item_id in admitted_ids:
            return "already_admitted"
        if respect_cap and db_counts[db_of(it)] >= db_cap:
            return "db_cap"
        ngrams = frozenset(char_ngrams(brief_text(it.brief.model_dump(mode="json"))))
        if is_near_dup(ngrams):
            return "near_duplicate"
        admitted.append(it)
        admitted_ids.add(it.item_id)
        admitted_ngrams.append(ngrams)
        db_counts[db_of(it)] += 1
        return "admitted"

    for chart in charts:
        target = target_per_chart[chart]
        for it in buckets[chart]:
            if chart_counts[chart] >= target:
                break
            reason = try_admit(it, respect_cap=True)
            if reason == "admitted":
                chart_counts[chart] += 1
            elif reason == "near_duplicate":
                near_dup_removed_by_chart[chart] += 1

    for chart in charts:
        target = target_per_chart[chart]
        if chart_counts[chart] >= target:
            continue
        for it in buckets[chart]:
            if chart_counts[chart] >= target:
                break
            reason = try_admit(it, respect_cap=False)
            if reason == "admitted":
                chart_counts[chart] += 1
                fallbacks.append({
                    "chart": chart, "item_id": it.item_id, "db_id": db_of(it),
                    "reason": "db_cap_relaxed",
                    "detail": f"database cap ({db_cap}) exhausted for chart '{chart}'; "
                              "admitted over cap to reach the recorded target.",
                })
            elif reason == "near_duplicate":
                near_dup_removed_by_chart[chart] += 1

    admitted.sort(key=lambda it: it.item_id)
    report = {
        "target_per_chart": dict(target_per_chart),
        "db_cap": db_cap,
        "near_dup_threshold": near_dup_threshold,
        "availability_per_chart_raw": availability_raw,
        "unique_groups_per_chart": unique_groups_per_chart,
        "dropped_exact_goal_duplicates_by_chart": dict(dropped_exact_goal_dups_by_chart),
        "dropped_exact_goal_duplicates_total": sum(dropped_exact_goal_dups_by_chart.values()),
        "near_dup_removed_by_chart": dict(near_dup_removed_by_chart),
        "chart_counts": chart_counts,
        "db_counts": dict(db_counts),
        "fallbacks": fallbacks,
    }

    short = {c: n for c, n in chart_counts.items() if n != target_per_chart[c]}
    if fallbacks:
        over_cap_charts = sorted({f["chart"] for f in fallbacks})
        report["status"] = "insufficient_tier_a_candidates_within_db_cap"
        report["short_charts"] = over_cap_charts
        return None, report
    if short:
        report["status"] = "insufficient_tier_a_candidates_after_dedup"
        report["short_charts"] = sorted(short)
        return None, report

    report["status"] = "ok"
    return admitted, report


def before_after_v5_v6(v5_records, v6_records, mapping, resolver, profiler, cfg) -> Dict[str, Any]:
    """Corpus-level v5-vs-v6 comparison under the (unchanged) v5 quality gate.

    ``v5_records`` is v5's corrected Tier-A pool (v5 itself shipped no
    ``accepted.jsonl``); ``v6_records`` is v6's selected 100. Never compares by
    item id.
    """
    return {
        "note": (
            "Corpus-level comparison under the unchanged v5 quality gate. v5 had no "
            "accepted pilot (insufficient scatter supply under a uniform 20-per-chart "
            "target); v6 compares against v5's corrected Tier-A pool."
        ),
        "v5": _corpus_quality_metrics_v5(v5_records, mapping, resolver, profiler, cfg),
        "v6": _corpus_quality_metrics_v5(v6_records, mapping, resolver, profiler, cfg),
    }
