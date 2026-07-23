"""Build and validate Pilot v4: the dashboard-suitability quality layer.

Runs the full nvBench technically-valid candidate pool through the quality
tiering layer (identifier detection, KPI suitability, chart suitability,
explainable scoring), selects 100 records (20 per normalized chart type) from
Tier-A candidates only, and validates the result against the v3 structural/
semantic/duplicate/leakage checks plus new dashboard-suitability checks.

Writes only to the given ``--out`` staging directory. Refuses any path
containing ``frozen``. Nothing under nvbench_pilot_v1/v2/v3 is read for
anything other than the read-only before/after comparison.

Usage:
    python experiments/scripts/run_nvbench_pilot_v4.py \
        --out data/staging/dashboard_v3/nvbench_pilot_v4 \
        --target-per-chart 20 --db-cap 10 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.builders.nvbench_builder import NvBenchBuilder  # noqa: E402
from src.data_pipeline.frozen_validation import read_jsonl_strict, sha256_of_file  # noqa: E402
from src.data_pipeline.nvbench_pilot import (  # noqa: E402
    NEAR_DUP_THRESHOLD,
    _mapping0,
    _prov,
    _record,
    _task,
    distribution_rows,
    duplicate_checks,
    leakage_checks,
    semantic_checks,
    structural_checks,
)
from src.data_pipeline.nvbench_pilot_v4 import (  # noqa: E402
    NORMALIZED_CHART_TYPES,
    before_after_v3_v4,
    select_pilot_v4,
)
from src.data_pipeline.nvbench_profile import DbProfiler  # noqa: E402
from src.data_pipeline.nvbench_quality import build_quality_pool, load_quality_config  # noqa: E402
from src.data_pipeline.nvbench_source import DbMetadataResolver, load_mapping  # noqa: E402
from src.utils.io import write_json  # noqa: E402

DEFAULT_NVBENCH_JSON = "data/raw_external/nvbench/extracted/nvBench-main/NVBench.json"
DEFAULT_CACHE_ROOT = "data/cache_external/nvbench/databases"
DEFAULT_MAPPING = "src/config/data/nvbench_mapping.yaml"
DEFAULT_QUALITY_CONFIG = "src/config/data/nvbench_quality_rules.yaml"
DEFAULT_PROFILE_CACHE = "data/cache_external/nvbench/field_profiles.json"
DEFAULT_V3_PILOT = "data/staging/dashboard_v3/nvbench_pilot_v3/accepted.jsonl"

# Independent evaluation artifacts the pilot must never leak into (same list v3 uses).
EVAL_ARTIFACTS = [
    ("real_briefs_v1", "data/eval/real_briefs_v1.jsonl", "top"),
    ("real_briefs_seed", "data/eval/real_briefs/items.jsonl", "top"),
    ("benchmark_v1", "data/eval/benchmark_v1.jsonl", "benchmark"),
    ("benchmark_v1_infer", "data/eval/benchmark_v1_infer.jsonl", "nested"),
    ("internal_test", "data/frozen/dashboard_v2/internal_test.jsonl", "nested"),
    ("test_paraphrased", "data/frozen/dashboard_v2/test_paraphrased.jsonl", "nested"),
    ("test_missing_info", "data/frozen/dashboard_v2/test_missing_info.jsonl", "nested"),
    ("human_eval_items", "experiments/results/human_eval/items.jsonl", "nested"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build and validate Pilot v4 (dashboard-suitability quality layer).")
    p.add_argument("--out", default="data/staging/dashboard_v3/nvbench_pilot_v4")
    p.add_argument("--nvbench-json", default=DEFAULT_NVBENCH_JSON)
    p.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    p.add_argument("--mapping", default=DEFAULT_MAPPING)
    p.add_argument("--quality-config", default=DEFAULT_QUALITY_CONFIG)
    p.add_argument("--profile-cache", default=DEFAULT_PROFILE_CACHE)
    p.add_argument("--v3-pilot", default=DEFAULT_V3_PILOT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-per-chart", type=int, default=20)
    p.add_argument("--db-cap", type=int, default=10)
    p.add_argument(
        "--allow-partial-charts", action="store_true",
        help="Accept a chart bucket below --target-per-chart when its Tier-A supply genuinely "
             "tops out after mandatory dedup/near-duplicate exclusion (never a rule relaxation, "
             "never a database-cap fallback -- that is still always refused). The shortfall is "
             "recorded explicitly in manifest.json and the validation report, never hidden.",
    )
    return p.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _jsonl_bytes(records) -> bytes:
    return ("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)).encode("utf-8")


def _write_jsonl(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def _write_distribution_csv(rows, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dimension", "value", "count"])
        w.writerows(rows)


def _dict_check(name, errors) -> dict:
    return {"check": name, "passed": False, "severity": "mandatory", "n": len(errors),
            "item_ids": [], "detail": "; ".join(errors[:5])}


def _load_eval_sources() -> list:
    sources = []
    for name, rel, kind in EVAL_ARTIFACTS:
        path = _resolve(rel)
        if not path.exists():
            sources.append({"name": name, "records": [], "kind": kind, "present": False})
            continue
        records, _errors = read_jsonl_strict(path)
        sources.append({"name": name, "records": records, "kind": kind, "present": True})
    return sources


# --------------------------------------------------------------------------- #
# manual audit template (Phase 11)
# --------------------------------------------------------------------------- #
_AUDIT_SOURCE_COLS = [
    "item_id", "split", "db_id", "source_group_id", "source_record_id", "original_query",
    "source_sql", "source_vql", "original_chart_label", "normalized_chart_type",
    "inferred_task_type", "raw_columns", "KPI", "encoding_x", "encoding_y", "aggregation",
    "filters", "sort", "time_grain", "group_field", "classify_values",
    "quality_tier", "quality_score", "quality_component_scores",
    "identifier_flags", "identifier_evidence", "field_profiles",
    "chart_suitability_rules", "kpi_suitability_rules", "source_fidelity_rules",
    "failed_rules", "warnings",
]
_AUDIT_REVIEW_COLS = [
    "reviewer_id", "query_goal_fidelity", "raw_columns_correct", "kpi_correct",
    "source_chart_preserved", "query_chart_consistent", "chart_appropriate",
    "encoding_correct", "constraints_preserved", "task_type_plausible",
    "source_fidelity", "design_validity", "accept", "error_category", "review_comment",
]


def _write_audit_template(records, quality_by_id: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_AUDIT_SOURCE_COLS + _AUDIT_REVIEW_COLS)
        w.writeheader()
        for r in records:
            prov = _prov(r)
            m = _mapping0(r)
            brief = r.get("brief") or {}
            enc = m.get("encoding") or {}
            constraints = prov.get("constraints") or {}
            q = quality_by_id.get(r.get("item_id", ""), {})
            row = {
                "item_id": r.get("item_id", ""),
                "split": r.get("split", ""),
                "db_id": prov.get("db_id", ""),
                "source_group_id": prov.get("source_group_id", ""),
                "source_record_id": prov.get("source_record_id", ""),
                "original_query": prov.get("nl_query", ""),
                "source_sql": ((prov.get("vis_query") or {}).get("data_part") or {}).get("sql_part", ""),
                "source_vql": (prov.get("vis_query") or {}).get("VQL", ""),
                "original_chart_label": prov.get("original_chart_label", ""),
                "normalized_chart_type": m.get("chart_type", ""),
                "inferred_task_type": _task(r).get("task_type", m.get("task_type", "")),
                "raw_columns": json.dumps(brief.get("columns", []), ensure_ascii=False),
                "KPI": json.dumps(brief.get("kpis", []), ensure_ascii=False),
                "encoding_x": enc.get("x", ""),
                "encoding_y": enc.get("y", ""),
                "aggregation": enc.get("aggregate") or "",
                "filters": json.dumps(constraints.get("filters", []), ensure_ascii=False),
                "sort": json.dumps(constraints.get("sort"), ensure_ascii=False),
                "time_grain": json.dumps(constraints.get("time_grain"), ensure_ascii=False),
                "group_field": enc.get("group_field") or "",
                "classify_values": json.dumps(enc.get("classify", []), ensure_ascii=False),
                "quality_tier": q.get("tier", ""),
                "quality_score": q.get("quality_score", ""),
                "quality_component_scores": json.dumps(q.get("component_scores", {}), ensure_ascii=False, default=str),
                "identifier_flags": json.dumps({
                    "kpi": (q.get("kpi_suitability", {}).get("evidence", {}) or {}).get("identifier"),
                }, ensure_ascii=False, default=str),
                "identifier_evidence": json.dumps(
                    q.get("kpi_suitability", {}).get("evidence", {}), ensure_ascii=False, default=str),
                "field_profiles": json.dumps(
                    q.get("chart_suitability", {}).get("evidence", {}), ensure_ascii=False, default=str),
                "chart_suitability_rules": json.dumps(
                    q.get("chart_suitability", {}).get("failed_rules", []), ensure_ascii=False),
                "kpi_suitability_rules": json.dumps(
                    q.get("kpi_suitability", {}).get("failed_rules", []), ensure_ascii=False),
                "source_fidelity_rules": json.dumps(q.get("fidelity_failed", []), ensure_ascii=False),
                "failed_rules": json.dumps(q.get("failed_rules", []), ensure_ascii=False),
                "warnings": json.dumps(q.get("warnings", []), ensure_ascii=False),
            }
            for c in _AUDIT_REVIEW_COLS:
                row[c] = ""  # human-review fields intentionally empty; never pre-filled
            w.writerow(row)


def _write_quality_rule_failures_csv(rule_failure_counts: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule", "count"])
        for rule, count in sorted(rule_failure_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            w.writerow([rule, count])


# --------------------------------------------------------------------------- #
# markdown reports
# --------------------------------------------------------------------------- #
def _write_validation_md(report: dict, path: Path) -> None:
    L = ["# Pilot v4 — Validation Report", ""]
    L.append(f"- pilot: `{report['pilot_dir']}`")
    L.append(f"- built_at: {report['built_at']}")
    L.append(f"- selected: {report['counts']['selected']}  "
              f"tier_a: {report['counts']['tier_a']}  tier_b: {report['counts']['tier_b']}  "
              f"tier_c: {report['counts']['tier_c']}")
    L.append(f"- technical_rejected: {report['counts']['technical_rejected']}  "
              f"accepted_total_technical: {report['counts']['accepted_total_technical']}")
    L.append(f"- mapping_version: {report['mapping_version']}  quality_rule_version: {report['quality_rule_version']}")
    L.append(f"- deterministic rebuild byte-identical: **{report['determinism']['byte_identical']}**")
    L.append(f"- near-duplicate threshold: {NEAR_DUP_THRESHOLD} (char-3gram Jaccard)")
    L.append("")
    L.append(f"## Result: **{'PASS' if report['passed'] else 'FAIL'}**")
    if not report["passed"]:
        L.append("")
        L.append("### Failed mandatory criteria")
        for c in report["failed_mandatory"]:
            L.append(f"- `{c['check']}` — {c['detail']} (n={c['n']}) {c['item_ids'][:10]}")
    for section in ("structural", "semantic", "duplicate", "leakage", "quality"):
        L.append("")
        L.append(f"## {section.capitalize()} checks")
        for c in report["checks"][section]:
            flag = "PASS" if c["passed"] else ("WARN" if c["severity"] == "warning" else "FAIL")
            L.append(f"- [{flag}] `{c['check']}` (n={c['n']}) {c['detail']}")
    L.append("")
    L.append("## Distributions")
    cur = None
    for dim, value, count in report["distributions"]:
        if dim != cur:
            L.append(f"### {dim}")
            cur = dim
        L.append(f"- `{value}`: {count}")
    L.append("")
    L.append(f"## Warnings: {report['n_warnings']} (see warnings.jsonl)")
    path.write_text("\n".join(L), encoding="utf-8")


def _write_quality_md(summary: dict, path: Path) -> None:
    L = ["# Pilot v4 — Quality Pool Report", ""]
    L.append(f"- rule_version: {summary['rule_version']}")
    L.append(f"- total_candidates: {summary['total_candidates']}")
    L.append(f"- Tier A: {summary['tier_a_count']}  Tier B: {summary['tier_b_count']}  Tier C: {summary['tier_c_count']}")
    sd = summary["score_distribution"]
    L.append(f"- score distribution: min={sd['min']} max={sd['max']} mean={sd['mean']} median={sd['median']}")
    L.append("")
    L.append("## Tier by chart")
    for chart, counts in sorted(summary["tier_by_chart"].items()):
        L.append(f"- `{chart}`: {counts}")
    L.append("")
    L.append("## Tier by database")
    for db, counts in sorted(summary["tier_by_database"].items()):
        L.append(f"- `{db}`: {counts}")
    L.append("")
    L.append("## Rule failure counts")
    for rule, count in sorted(summary["rule_failure_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        L.append(f"- `{rule}`: {count}")
    path.write_text("\n".join(L), encoding="utf-8")


def _write_before_after_md(before_after: dict, path: Path) -> None:
    L = ["# Pilot v3 vs Pilot v4 — Before/After Comparison", ""]
    if "error" in before_after:
        L.append(f"- {before_after['error']}")
        path.write_text("\n".join(L), encoding="utf-8")
        return
    L.append("AI-precheck problems this quality layer targets:")
    for p in before_after["ai_precheck_problems_targeted"]:
        L.append(f"- {p}")
    L.append("")
    metrics = [
        "n", "quality_tier_distribution", "identifier_as_measure_count", "meaningless_kpi_count",
        "chart_inappropriate_count", "source_conflict_count", "missing_constraint_count",
        "uncertain_pie_count", "invalid_scatter_count", "invalid_line_count",
        "invalid_stacked_bar_count", "invalid_bar_count", "chart_distribution",
        "database_distribution", "duplicate_findings",
    ]
    L.append("| metric | v3 | v4 |")
    L.append("|---|---|---|")
    for m in metrics:
        L.append(f"| {m} | {before_after['v3'].get(m)} | {before_after['v4'].get(m)} |")
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    out_dir = _resolve(args.out)
    if "frozen" in out_dir.parts:
        raise SystemExit(f"refusing to write into a frozen path: {out_dir}")

    mapping = load_mapping(_resolve(args.mapping))
    cfg = load_quality_config(str(_resolve(args.quality_config)))
    cache_root = _resolve(args.cache_root)
    resolver_cache_root = str(cache_root) if cache_root.exists() else None
    resolver = DbMetadataResolver(resolver_cache_root)
    profiler = DbProfiler(resolver, cache_path=_resolve(args.profile_cache))

    def _build_pool():
        builder = NvBenchBuilder(
            str(_resolve(args.nvbench_json)), cache_root=resolver_cache_root, mapping_path=str(_resolve(args.mapping)),
        )
        build_result = builder.build()
        pool = build_quality_pool(build_result.accepted, mapping, resolver, profiler, cfg)
        return build_result, pool

    build_result, pool = _build_pool()
    profiler.save_cache()

    selected, sampling_report = select_pilot_v4(
        pool["tier_a"], seed=args.seed, target_per_chart=args.target_per_chart, db_cap=args.db_cap,
        allow_partial=args.allow_partial_charts,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    quality_pool_dir = out_dir / "quality_pool"
    quality_pool_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl([_record(it) for it in pool["tier_a"]], quality_pool_dir / "tier_a_candidates.jsonl")
    _write_jsonl([_record(it) for it in pool["tier_b"]], quality_pool_dir / "tier_b_diagnostics.jsonl")
    _write_jsonl([_record(it) for it in pool["tier_c"]], quality_pool_dir / "tier_c_rejected.jsonl")
    write_json(pool["summary"], quality_pool_dir / "quality_pool_summary.json")
    _write_jsonl(build_result.rejections, out_dir / "rejected.jsonl")

    built_at = datetime.now(timezone.utc).isoformat()

    if selected is None:
        write_json({
            "built_at": built_at, "source": "nvbench", "usage_tier": "train_aug",
            "status": sampling_report["status"], "passed": False,
            "sampling_report": sampling_report, "quality_pool_summary": pool["summary"],
            "quality_rule_version": cfg.get("rule_version"),
        }, out_dir / "manifest.json")
        write_json(pool["summary"], reports_dir / "quality_report.json")
        _write_quality_md(pool["summary"], reports_dir / "quality_report.md")
        _write_quality_rule_failures_csv(pool["summary"]["rule_failure_counts"], reports_dir / "quality_rule_failures.csv")
        # No valid Pilot v4 selection exists: remove any stale per-selection
        # reports from a previous run rather than leave them describing a
        # selection that no longer exists.
        for stale in ("validation_report.json", "validation_report.md", "duplicate_report.jsonl",
                      "leakage_report.jsonl", "warnings.jsonl", "distribution_report.csv",
                      "manual_audit_template.csv", "before_after_v3_v4.json", "before_after_v3_v4.md"):
            stale_path = reports_dir / stale
            if stale_path.exists():
                stale_path.unlink()
        write_json({
            "pilot_dir": out_dir.as_posix(), "built_at": built_at, "passed": False,
            "status": sampling_report["status"], "sampling_report": sampling_report,
            "quality_pool_summary": pool["summary"],
            "detail": (
                f"Pilot v4 selection refused: chart type(s) {sampling_report.get('short_charts')} "
                f"cannot reach {args.target_per_chart} Tier-A records "
                f"({sampling_report.get('selected_per_chart') or sampling_report.get('available_per_chart')}) "
                "without weakening quality rules or accepting a database-cap fallback. "
                "No accepted.jsonl was written; see quality_pool/ for the full Tier-A/B/C diagnostic pools."
            ),
        }, reports_dir / "validation_report.json")
        _write_before_after_md({"error": "no Pilot v4 selection to compare (insufficient Tier-A supply)"},
                                reports_dir / "before_after_v3_v4.md")
        print(f"[FAIL] {sampling_report['status']}: {sampling_report}")
        print(f"[reports] {reports_dir}")
        raise SystemExit(1)

    records = [_record(it) for it in selected]
    acc_bytes = _jsonl_bytes(records)
    acc_path = out_dir / "accepted.jsonl"

    if acc_path.exists():
        existing = acc_path.read_bytes()
        if existing == acc_bytes:
            print("[idempotent] accepted.jsonl byte-identical; re-running validation + reports only.")
        else:
            print("[conflict] existing Pilot v4 differs from a fresh rebuild; not overwriting.")
            raise SystemExit(1)
    else:
        acc_path.write_bytes(acc_bytes)
        print(f"[build] wrote {len(records)} Pilot v4 records to {out_dir}")

    # Deterministic rebuild check: rerun the whole build+tier+select pipeline once more.
    _, pool2 = _build_pool()
    selected2, sampling_report2 = select_pilot_v4(
        pool2["tier_a"], seed=args.seed, target_per_chart=args.target_per_chart, db_cap=args.db_cap,
        allow_partial=args.allow_partial_charts,
    )
    records2 = [_record(it) for it in selected2] if selected2 is not None else []
    tmp_dir = out_dir / ".rebuild_check"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "accepted.jsonl").write_bytes(_jsonl_bytes(records2))
    determinism = {
        "accepted_sha256_main": sha256_of_file(acc_path),
        "accepted_sha256_rebuild": sha256_of_file(tmp_dir / "accepted.jsonl"),
    }
    determinism["byte_identical"] = determinism["accepted_sha256_main"] == determinism["accepted_sha256_rebuild"]
    (tmp_dir / "accepted.jsonl").unlink()
    tmp_dir.rmdir()

    rb_records, acc_errors = read_jsonl_strict(acc_path)
    json_valid = not acc_errors

    is_partial = sampling_report.get("status") == "ok_partial_documented_shortfall"
    # A documented, user-approved partial pilot has a different -- but still
    # exact and deterministic -- expected count than the nominal target.
    expected_total = len(records) if is_partial else args.target_per_chart * len(NORMALIZED_CHART_TYPES)
    s_checks = structural_checks(rb_records, expected=expected_total)
    sem_checks, warnings = semantic_checks(rb_records, mapping, resolver=resolver)
    dup_checks, dup_findings = duplicate_checks(rb_records, strict=True)
    lk_checks, lk_findings = leakage_checks(rb_records, _load_eval_sources())
    dist = distribution_rows(rb_records)

    quality_by_id = pool["quality_by_id"]
    tier_a_min_score = cfg["scoring"]["tier_a_min_score"]
    not_tier_a, below_min, mandatory_fail = [], [], []
    for r in rb_records:
        q = quality_by_id.get(r["item_id"])
        if q is None or q["tier"] != "A":
            not_tier_a.append(r["item_id"])
        if q is None or q["quality_score"] < tier_a_min_score:
            below_min.append(r["item_id"])
        if q and q["failed_rules"]:
            mandatory_fail.append(r["item_id"])

    quality_checks = [
        {"check": "all_selected_are_tier_a", "passed": not not_tier_a, "severity": "mandatory",
         "n": len(not_tier_a), "item_ids": not_tier_a, "detail": "selected record not Tier A"},
        {"check": "all_scores_at_least_min", "passed": not below_min, "severity": "mandatory",
         "n": len(below_min), "item_ids": below_min, "detail": f"quality_score below {tier_a_min_score}"},
        {"check": "no_mandatory_quality_failures", "passed": not mandatory_fail, "severity": "mandatory",
         "n": len(mandatory_fail), "item_ids": mandatory_fail, "detail": "selected record has a failed quality rule"},
    ]

    all_checks = {"structural": s_checks, "semantic": sem_checks, "duplicate": dup_checks,
                  "leakage": lk_checks, "quality": quality_checks}
    failed_mandatory = [c for group in all_checks.values() for c in group
                        if c["severity"] == "mandatory" and not c["passed"]]

    if not json_valid:
        failed_mandatory.append(_dict_check("json_valid_readback", acc_errors))
    if not determinism["byte_identical"]:
        failed_mandatory.append({"check": "deterministic_rebuild", "passed": False, "severity": "mandatory",
                                 "n": 0, "item_ids": [], "detail": "rebuild not byte-identical"})
    short = {c: n for c, n in sampling_report.get("chart_counts", {}).items() if n != args.target_per_chart}
    per_chart_target_check = None
    if short and is_partial:
        # User-approved, documented deviation from the nominal per-chart target
        # (verified genuine Tier-A scarcity after mandatory dedup/near-dup
        # exclusion, not a rule relaxation) -- reported, never silently hidden,
        # but not counted as a mandatory failure of the accepted pilot.
        per_chart_target_check = {
            "check": "exact_per_chart_target", "passed": True, "severity": "info",
            "n": len(short), "item_ids": [],
            "detail": f"documented partial pilot: chart counts != {args.target_per_chart}: {short} "
                      f"(user-approved; see sampling_report.shortfall)",
        }
    elif short:
        failed_mandatory.append({"check": "exact_per_chart_target", "passed": False, "severity": "mandatory",
                                 "n": len(short), "item_ids": [],
                                 "detail": f"chart counts != {args.target_per_chart}: {short}"})
    else:
        per_chart_target_check = {"check": "exact_per_chart_target", "passed": True, "severity": "mandatory",
                                   "n": 0, "item_ids": [], "detail": f"all charts == {args.target_per_chart}"}
    if per_chart_target_check:
        all_checks.setdefault("quality", []).append(per_chart_target_check)

    passed = not failed_mandatory

    report = {
        "pilot_dir": out_dir.as_posix(), "built_at": built_at, "passed": passed,
        "json_valid_readback": json_valid,
        "counts": {
            "selected": len(rb_records), "tier_a": pool["summary"]["tier_a_count"],
            "tier_b": pool["summary"]["tier_b_count"], "tier_c": pool["summary"]["tier_c_count"],
            "technical_rejected": len(build_result.rejections),
            "accepted_total_technical": build_result.stats["n_accepted"],
        },
        "mapping_version": mapping.get("mapping_version"), "quality_rule_version": cfg.get("rule_version"),
        "determinism": determinism, "checks": all_checks, "failed_mandatory": failed_mandatory,
        "distributions": dist, "n_warnings": len(warnings), "sampling_report": sampling_report,
    }

    write_json(report, reports_dir / "validation_report.json")
    _write_validation_md(report, reports_dir / "validation_report.md")
    write_json(pool["summary"], reports_dir / "quality_report.json")
    _write_quality_md(pool["summary"], reports_dir / "quality_report.md")
    _write_quality_rule_failures_csv(pool["summary"]["rule_failure_counts"], reports_dir / "quality_rule_failures.csv")
    _write_distribution_csv(dist, reports_dir / "distribution_report.csv")
    _write_jsonl(dup_findings, reports_dir / "duplicate_report.jsonl")
    _write_jsonl(lk_findings, reports_dir / "leakage_report.jsonl")
    _write_jsonl(warnings, reports_dir / "warnings.jsonl")
    _write_audit_template(rb_records, quality_by_id, reports_dir / "manual_audit_template.csv")

    v3_path = _resolve(args.v3_pilot)
    if v3_path.exists():
        v3_records, _v3_errors = read_jsonl_strict(v3_path)
        before_after = before_after_v3_v4(v3_records, rb_records, mapping, resolver, profiler, cfg)
    else:
        before_after = {"error": f"v3 pilot not found at {v3_path}"}
    write_json(before_after, reports_dir / "before_after_v3_v4.json")
    _write_before_after_md(before_after, reports_dir / "before_after_v3_v4.md")

    write_json({
        "built_at": built_at, "source": "nvbench", "usage_tier": "train_aug",
        "status": "ok" if passed else "quality_gate_failed",
        "seed": args.seed,
        "selection_policy": {
            "target_per_chart": args.target_per_chart, "db_cap": args.db_cap,
            "near_duplicate_threshold": NEAR_DUP_THRESHOLD,
        },
        "partial_pilot": is_partial,
        "chart_shortfall": short if is_partial else None,
        "quality_rule_version": cfg.get("rule_version"), "mapping_version": mapping.get("mapping_version"),
        "counts": report["counts"], "determinism": determinism, "passed": passed,
    }, out_dir / "manifest.json")
    write_json({
        "n_items": len(rb_records),
        "distributions": [{"dimension": d, "value": v, "count": c} for d, v, c in dist],
    }, out_dir / "distribution_report.json")

    print(f"[validate] json_valid={json_valid} byte_identical={determinism['byte_identical']} warnings={len(warnings)}")
    print(f"[result] {'PASS' if passed else 'FAIL'}")
    if not passed:
        for c in failed_mandatory:
            print(f"  FAILED {c['check']}: {c['detail']} ids={c['item_ids'][:10]}")
    print(f"[reports] {reports_dir}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
