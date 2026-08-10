"""Rebuild the FINAL nvBench quality pool (not a pilot): enriched per-record
evidence, full manifest/hash/validation artifact set.

Reruns ``build_quality_pool`` (unchanged) over the full technically-valid
nvBench candidate pool. Unlike the plain tier_a/b/c dumps written by
``rebuild_nvbench_quality_pool.py``, every record here is enriched with its
quality tier, score, component scores, mandatory failures, warnings, rule
version, and full KPI/chart/constraint evidence -- so a reviewer never has to
cross-reference a separate summary file to see *why* a record landed where it
did.

No pilot sampling. No ``accepted.jsonl``. No manual-audit template. Writes only
to ``--out`` (default: a standalone location, never a pilot version directory).

Usage:
    python experiments/scripts/rebuild_nvbench_quality_pool_final.py \
        --out data/staging/dashboard_v3/nvbench_quality_pool_final
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.builders.nvbench_builder import NvBenchBuilder  # noqa: E402
from src.data_pipeline.frozen_validation import sha256_of_file  # noqa: E402
from src.data_pipeline.nvbench_pilot import _mapping0, _prov, _record  # noqa: E402
from src.data_pipeline.nvbench_profile import DbProfiler  # noqa: E402
from src.data_pipeline.nvbench_quality import build_quality_pool, load_quality_config  # noqa: E402
from src.data_pipeline.nvbench_source import DbMetadataResolver, item_chart, item_group_id, load_mapping  # noqa: E402
from src.utils.io import write_json  # noqa: E402

DEFAULT_NVBENCH_JSON = "data/raw_external/nvbench/extracted/nvBench-main/NVBench.json"
DEFAULT_CACHE_ROOT = "data/cache_external/nvbench/databases"
DEFAULT_MAPPING = "src/config/data/nvbench_mapping.yaml"
DEFAULT_QUALITY_CONFIG = "src/config/data/nvbench_quality_rules.yaml"
DEFAULT_PROFILE_CACHE = "data/cache_external/nvbench/field_profiles.json"

_SCATTER_AXIS_RULES = ("scatter_identifier_axis", "invalid_scatter_axes")
_KPI_SQL_CONFLICT_RULES = ("kpi_sql_aggregation_conflict", "mixed_aggregate_ambiguous_kpi")
_IDENTIFIER_MEASURE_RULES = ("identifier_as_measure", "identifier_as_continuous_kpi",
                            "meaningless_identifier_aggregation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild the final, enriched nvBench quality pool (no pilot sampling).")
    p.add_argument("--out", default="data/staging/dashboard_v3/nvbench_quality_pool_final")
    p.add_argument("--nvbench-json", default=DEFAULT_NVBENCH_JSON)
    p.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    p.add_argument("--mapping", default=DEFAULT_MAPPING)
    p.add_argument("--quality-config", default=DEFAULT_QUALITY_CONFIG)
    p.add_argument("--profile-cache", default=DEFAULT_PROFILE_CACHE)
    return p.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _write_jsonl(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def _enriched_record(record: dict, quality: dict) -> dict:
    prov = _prov(record)
    m = _mapping0(record)
    return {
        "item_id": record.get("item_id"),
        "split": record.get("split"),
        "source_record_id": prov.get("source_record_id"),
        "source_group_id": prov.get("source_group_id"),
        "db_id": prov.get("db_id"),
        "chart_type": m.get("chart_type"),
        "quality_tier": quality.get("tier"),
        "quality_score": quality.get("quality_score"),
        "component_scores": quality.get("component_scores"),
        "failed_rules": quality.get("failed_rules"),
        "warnings": quality.get("warnings"),
        "rule_version": quality.get("rule_version"),
        "evidence": {
            "kpi_suitability": quality.get("kpi_suitability"),
            "chart_suitability": quality.get("chart_suitability"),
            "constraint_suitability": quality.get("constraint_suitability"),
            "source_consistency": quality.get("source_consistency"),
            "source_fidelity_failed_checks": quality.get("fidelity_failed"),
        },
        "record": record,
    }


def _check(name, ids, detail, severity="mandatory"):
    ids = sorted(set(ids))
    return {"check": name, "passed": not ids, "severity": severity, "n": len(ids), "item_ids": ids, "detail": detail}


def _validate(tier_a_items, quality_by_id: dict) -> list:
    below_min, mandatory_fail, bad_pie_agg, id_measure, bad_scatter, kpi_conflict, missing_constraint = (
        [], [], [], [], [], [], [])
    pie_count_sum_seen = False
    for it in tier_a_items:
        iid = it.item_id
        q = quality_by_id.get(iid, {})
        if q.get("quality_score", 0) < 90:
            below_min.append(iid)
        if q.get("failed_rules"):
            mandatory_fail.append(iid)
        chart = item_chart(it)
        if chart == "pie":
            y_agg = (it.brief.extra.get("provenance", {}).get("axis_typing", {}).get("y") or {}).get("aggregate")
            if y_agg:
                if y_agg.upper() not in ("COUNT", "SUM"):
                    bad_pie_agg.append(iid)
                else:
                    pie_count_sum_seen = True
        failed = set(q.get("failed_rules", []))
        if failed & set(_IDENTIFIER_MEASURE_RULES):
            id_measure.append(iid)
        if failed & set(_SCATTER_AXIS_RULES):
            bad_scatter.append(iid)
        if failed & set(_KPI_SQL_CONFLICT_RULES):
            kpi_conflict.append(iid)
        if failed & {
            "missing_required_time_grain", "missing_required_grouping",
            "missing_required_dimension", "missing_aggregate_condition",
            "time_grain_source_conflict", "source_conflict",
        }:
            missing_constraint.append(iid)

    checks = [
        _check("all_tier_a_score_at_least_90", below_min, "Tier-A record scored below 90"),
        _check("no_tier_a_mandatory_failure", mandatory_fail, "Tier-A record has a non-empty failed_rules list"),
        _check("no_tier_a_pie_avg_min_max", bad_pie_agg, "Tier-A pie uses a non-additive aggregate"),
        {"check": "valid_pie_count_sum_remain_eligible", "passed": pie_count_sum_seen, "severity": "mandatory",
         "n": 0 if pie_count_sum_seen else 1, "item_ids": [],
         "detail": "at least one Tier-A pie with COUNT/SUM aggregate" if pie_count_sum_seen
         else "no Tier-A pie with COUNT/SUM found"},
        _check("no_identifier_as_measure_tier_a", id_measure, "Tier-A record uses an identifier as a measure"),
        _check("no_invalid_scatter_axis_tier_a", bad_scatter, "Tier-A scatter has an identifier/categorical axis"),
        _check("no_kpi_sql_conflict_tier_a", kpi_conflict, "Tier-A record has a KPI/SQL aggregation conflict"),
        _check("no_missing_required_constraint_tier_a", missing_constraint,
              "Tier-A record is missing a required time-grain/grouping constraint"),
    ]
    return checks


def _write_summary_md(summary: dict, path: Path) -> None:
    L = ["# nvBench Final Quality Pool — Summary", ""]
    L.append(f"- rule_version: {summary['rule_version']}")
    L.append(f"- total_candidates: {summary['total_candidates']}")
    L.append(f"- Tier A: {summary['tier_a_count']}  Tier B: {summary['tier_b_count']}  Tier C: {summary['tier_c_count']}")
    sd = summary["score_distribution"]
    L.append(f"- score distribution: min={sd['min']} max={sd['max']} mean={sd['mean']} median={sd['median']}")
    L.append("")
    L.append("## Tier A by chart")
    for chart, counts in sorted(summary["tier_by_chart"].items()):
        L.append(f"- `{chart}`: {counts}")
    L.append("")
    L.append("## Rule failure counts")
    for rule, count in sorted(summary["rule_failure_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        L.append(f"- `{rule}`: {count}")
    path.write_text("\n".join(L), encoding="utf-8")


def _write_validation_md(report: dict, path: Path) -> None:
    L = ["# nvBench Final Quality Pool — Validation Report", ""]
    L.append(f"- built_at: {report['built_at']}")
    L.append(f"- rule_version: {report['rule_version']}")
    L.append(f"## Result: **{'PASS' if report['passed'] else 'FAIL'}**")
    L.append("")
    for c in report["checks"]:
        flag = "PASS" if c["passed"] else "FAIL"
        L.append(f"- [{flag}] `{c['check']}` (n={c['n']}) {c['detail']}")
    path.write_text("\n".join(L), encoding="utf-8")


def _write_rule_failures_csv(rule_failure_counts: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule", "count"])
        for rule, count in sorted(rule_failure_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            w.writerow([rule, count])


def main() -> None:
    args = parse_args()
    out_dir = _resolve(args.out)
    if "frozen" in out_dir.parts:
        raise SystemExit(f"refusing to write into a frozen path: {out_dir}")
    if any(p.startswith("nvbench_pilot_v") for p in out_dir.parts):
        raise SystemExit(f"refusing to write into a pilot version directory: {out_dir}")

    mapping = load_mapping(_resolve(args.mapping))
    cfg = load_quality_config(str(_resolve(args.quality_config)))
    cache_root = _resolve(args.cache_root)
    resolver_cache_root = str(cache_root) if cache_root.exists() else None
    resolver = DbMetadataResolver(resolver_cache_root)
    profiler = DbProfiler(resolver, cache_path=_resolve(args.profile_cache))

    builder = NvBenchBuilder(
        str(_resolve(args.nvbench_json)), cache_root=resolver_cache_root, mapping_path=str(_resolve(args.mapping)),
    )
    build_result = builder.build()
    pool = build_quality_pool(build_result.accepted, mapping, resolver, profiler, cfg)
    profiler.save_cache()

    out_dir.mkdir(parents=True, exist_ok=True)
    quality_by_id = pool["quality_by_id"]

    tier_a_enriched = [_enriched_record(_record(it), quality_by_id[it.item_id]) for it in pool["tier_a"]]
    tier_b_enriched = [_enriched_record(_record(it), quality_by_id[it.item_id]) for it in pool["tier_b"]]
    tier_c_enriched = [_enriched_record(_record(it), quality_by_id[it.item_id]) for it in pool["tier_c"]]

    tier_a_path = out_dir / "tier_a_candidates.jsonl"
    tier_b_path = out_dir / "tier_b_diagnostics.jsonl"
    tier_c_path = out_dir / "tier_c_rejected.jsonl"
    summary_path = out_dir / "quality_pool_summary.json"

    _write_jsonl(tier_a_enriched, tier_a_path)
    _write_jsonl(tier_b_enriched, tier_b_path)
    _write_jsonl(tier_c_enriched, tier_c_path)
    write_json(pool["summary"], summary_path)
    _write_summary_md(pool["summary"], out_dir / "quality_pool_summary.md")
    _write_rule_failures_csv(pool["summary"]["rule_failure_counts"], out_dir / "quality_rule_failures.csv")

    built_at = datetime.now(timezone.utc).isoformat()
    validation_checks = _validate(pool["tier_a"], quality_by_id)
    passed = all(c["passed"] for c in validation_checks)
    validation_report = {
        "built_at": built_at, "rule_version": cfg.get("rule_version"), "passed": passed, "checks": validation_checks,
    }
    write_json(validation_report, out_dir / "validation_report.json")
    _write_validation_md(validation_report, out_dir / "validation_report.md")

    unique_groups_per_chart: dict = collections.defaultdict(set)
    for it in pool["tier_a"]:
        unique_groups_per_chart[item_chart(it)].add(item_group_id(it))
    unique_groups_per_chart = {c: len(g) for c, g in unique_groups_per_chart.items()}

    demoted_by_pie_rule = sum(
        1 for q in quality_by_id.values() if "pie_non_additive_kpi" in q.get("failed_rules", [])
    )
    scores = [q["quality_score"] for q in quality_by_id.values()]

    manifest = {
        "built_at": built_at,
        "source": "nvbench",
        "kind": "quality_pool_final",
        "rule_version": cfg.get("rule_version"),
        "mapping_version": mapping.get("mapping_version"),
        "passed": passed,
        "total_technically_valid_candidates": len(build_result.accepted),
        "counts": {
            "tier_a": pool["summary"]["tier_a_count"],
            "tier_b": pool["summary"]["tier_b_count"],
            "tier_c": pool["summary"]["tier_c_count"],
        },
        "tier_a_by_chart": {c: v.get("A", 0) for c, v in pool["summary"]["tier_by_chart"].items()},
        "unique_tier_a_groups_by_chart": unique_groups_per_chart,
        "pie_non_additive_kpi_demoted_count": demoted_by_pie_rule,
        "quality_score_range": {"min": min(scores) if scores else None, "max": max(scores) if scores else None},
        "pilots_untouched": ["nvbench_pilot_v1", "nvbench_pilot_v2", "nvbench_pilot_v3",
                            "nvbench_pilot_v4", "nvbench_pilot_v5", "nvbench_pilot_v6"],
    }
    write_json(manifest, out_dir / "manifest.json")

    hashes = {
        "tier_a_candidates": sha256_of_file(tier_a_path),
        "tier_b_diagnostics": sha256_of_file(tier_b_path),
        "tier_c_rejected": sha256_of_file(tier_c_path),
        "quality_pool_summary": sha256_of_file(summary_path),
    }
    write_json(hashes, out_dir / "hashes.json")
    # manifest.json is written above and hashed last (it doesn't hash itself).
    hashes["manifest"] = sha256_of_file(out_dir / "manifest.json")
    write_json(hashes, out_dir / "hashes.json")

    print(json.dumps({"manifest": manifest, "hashes": hashes, "validation_passed": passed}, indent=2))
    print(f"[rebuild] wrote final quality pool to {out_dir}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
