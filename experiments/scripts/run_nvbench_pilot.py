"""Build and validate the versioned staging nvBench pilot (nothing is frozen).

Builds the deterministic 100-item nvBench pilot into a new versioned staging
directory, verifies a byte-identical rebuild, runs the full validation gate
(structural, semantic, duplicate, leakage, distribution) on the
serialized-and-read-back records, and writes seven report artifacts.

Writes only to the given ``--out`` staging directory. Refuses any path containing
``frozen``. On an idempotent rerun with byte-identical candidate files it does not
rewrite them (validation + reports only); it exits 1 if the rebuilt bytes differ.

Usage:
    python experiments/scripts/run_nvbench_pilot.py \
        --out data/staging/dashboard_v3/nvbench_pilot_v1 \
        --limit 100 --seed 42 --one-query-per-group --stratify-by-chart
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

from src.data_pipeline.frozen_validation import read_jsonl_strict, sha256_of_file  # noqa: E402
from src.data_pipeline.nvbench_pilot import (  # noqa: E402
    NEAR_DUP_THRESHOLD,
    build_pilot_records,
    distribution_rows,
    duplicate_checks,
    leakage_checks,
    semantic_checks,
    structural_checks,
    _lineage,
    _mapping0,
    _prov,
    _task,
)
from src.data_pipeline.nvbench_source import DbMetadataResolver, load_mapping  # noqa: E402
from src.utils.io import write_json  # noqa: E402

DEFAULT_NVBENCH_JSON = "data/raw_external/nvbench/extracted/nvBench-main/NVBench.json"
DEFAULT_CACHE_ROOT = "data/cache_external/nvbench/databases"
DEFAULT_MAPPING = "src/config/data/nvbench_mapping.yaml"

# Independent evaluation artifacts the pilot must never leak into.
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
    p = argparse.ArgumentParser(description="Build and validate the versioned staging nvBench pilot.")
    p.add_argument("--out", required=True)
    p.add_argument("--nvbench-json", default=DEFAULT_NVBENCH_JSON)
    p.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    p.add_argument("--mapping", default=DEFAULT_MAPPING)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--one-query-per-group", action="store_true")
    p.add_argument("--stratify-by-chart", action="store_true")
    p.add_argument("--target-per-chart", type=int, default=None,
                    help="Use the v3 sampler: exactly this many records per normalized "
                         "chart type, database-capped and near-duplicate-aware.")
    p.add_argument("--db-cap", type=int, default=10,
                    help="Max records from one database (v3 sampler only); relaxed "
                         "per-bucket only if a chart target cannot otherwise be reached.")
    return p.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _jsonl_bytes(records) -> bytes:
    return ("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)).encode("utf-8")


def _build(args):
    cache_root = _resolve(args.cache_root)
    return build_pilot_records(
        str(_resolve(args.nvbench_json)),
        str(cache_root) if cache_root.exists() else None,
        str(_resolve(args.mapping)),
        limit=args.limit,
        seed=args.seed,
        one_per_group=args.one_query_per_group,
        stratify=args.stratify_by_chart,
        target_per_chart=args.target_per_chart,
        db_cap=args.db_cap,
    )


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
# report writers
# --------------------------------------------------------------------------- #
def _write_distribution_csv(rows, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dimension", "value", "count"])
        w.writerows(rows)


_AUDIT_SOURCE_COLS = [
    "item_id", "source_group_id", "source_record_id", "original_query",
    "original_chart_label", "normalized_chart_type", "inferred_task_type",
    "task_confidence", "KPI", "columns", "encoding", "grouping_information",
    "generated_goal", "generated_recommendation",
    # Pilot v3 additions (machine-filled from provenance).
    "split", "db_id", "warning_flags", "filters", "sort", "time_grain", "group_field",
]
_AUDIT_REVIEW_COLS = [
    "reviewer_id", "query_goal_fidelity", "kpi_correct", "columns_correct",
    "source_chart_preserved", "encoding_correct", "task_type_plausible",
    "template_fields_coherent", "accept", "error_category", "review_comment",
    # Pilot v3 additions (left empty; human review only).
    "raw_columns_correct", "query_chart_consistent", "chart_appropriate",
    "constraints_preserved", "source_fidelity", "design_validity",
]


def _write_audit_template(records, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_AUDIT_SOURCE_COLS + _AUDIT_REVIEW_COLS)
        w.writeheader()
        for r in records:
            prov = _prov(r)
            m = _mapping0(r)
            brief = r.get("brief") or {}
            enc = m.get("encoding") or {}
            constraints = prov.get("constraints") or {}
            row = {
                "item_id": r.get("item_id", ""),
                "source_group_id": prov.get("source_group_id", ""),
                "source_record_id": prov.get("source_record_id", ""),
                "original_query": prov.get("nl_query", ""),
                "original_chart_label": prov.get("original_chart_label", ""),
                "normalized_chart_type": m.get("chart_type", ""),
                "inferred_task_type": _task(r).get("task_type", m.get("task_type", "")),
                "task_confidence": _task(r).get("confidence", ""),
                "KPI": json.dumps(brief.get("kpis", []), ensure_ascii=False),
                "columns": json.dumps(brief.get("columns", []), ensure_ascii=False),
                "encoding": json.dumps(enc, ensure_ascii=False),
                "grouping_information": json.dumps(prov.get("grouping", {}), ensure_ascii=False),
                "generated_goal": json.dumps(brief.get("goals", []), ensure_ascii=False),
                "generated_recommendation": json.dumps(r.get("recommendation", {}), ensure_ascii=False),
                "split": r.get("split", ""),
                "db_id": prov.get("db_id", ""),
                "warning_flags": json.dumps(prov.get("build_warnings", []), ensure_ascii=False),
                "filters": json.dumps(constraints.get("filters", []), ensure_ascii=False),
                "sort": json.dumps(constraints.get("sort"), ensure_ascii=False),
                "time_grain": json.dumps(constraints.get("time_grain"), ensure_ascii=False),
                "group_field": enc.get("group_field") or "",
            }
            for c in _AUDIT_REVIEW_COLS:
                row[c] = ""  # human-review fields intentionally empty
            w.writerow(row)


def _write_md(report: dict, path: Path) -> None:
    L = ["# nvBench Versioned Staging Pilot — Validation Report", ""]
    L.append(f"- pilot: `{report['pilot_dir']}`")
    L.append(f"- built_at: {report['built_at']}")
    L.append(f"- accepted: {report['counts']['selected']}  rejected: {report['counts']['rejected_total']}")
    L.append(f"- mapping_version: {report['mapping_version']}  task_rule_version: {report['task_rule_version']}")
    L.append(f"- deterministic rebuild byte-identical: **{report['determinism']['byte_identical']}**")
    L.append(f"- near-duplicate threshold: {NEAR_DUP_THRESHOLD} (char-3gram Jaccard)")
    sampling = report.get("sampling_fallback")
    if sampling:
        L.append(f"- sampling: chart_counts={sampling['chart_counts']} db_cap={sampling['db_cap']} "
                 f"dropped_exact_goal_duplicates={sampling['dropped_exact_goal_duplicates']}")
        if sampling["fallbacks"]:
            L.append(f"- **sampling fallback used** ({len(sampling['fallbacks'])} admissions over the database cap):")
            for f in sampling["fallbacks"]:
                L.append(f"  - `{f['item_id']}` (chart={f['chart']}, db={f['db_id']}): {f['detail']}")
        else:
            L.append("- sampling fallback: none needed (database cap satisfied for every chart)")
    L.append("")
    L.append(f"## Result: **{'PASS' if report['passed'] else 'FAIL'}**")
    if not report["passed"]:
        L.append("")
        L.append("### Failed mandatory criteria")
        for c in report["failed_mandatory"]:
            L.append(f"- `{c['check']}` — {c['detail']} (n={c['n']}) {c['item_ids'][:10]}")
    for section in ("structural", "semantic", "duplicate", "leakage"):
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
    L.append("## Warnings & limitations")
    L.append(f"- warnings emitted: {report['n_warnings']} (see warnings.jsonl)")
    L.append("- Grouping series-field names are not recoverable from nvBench; raw classify "
             "values are preserved in provenance rather than inventing a column name.")
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    out_dir = _resolve(args.out)
    if "frozen" in out_dir.parts:
        raise SystemExit(f"refusing to write into a frozen path: {out_dir}")

    records, rejections, meta = _build(args)
    acc_bytes = _jsonl_bytes(records)
    rej_bytes = _jsonl_bytes(rejections)
    acc_path = out_dir / "accepted.jsonl"
    rej_path = out_dir / "rejected.jsonl"

    # Conflict guard / idempotency.
    if acc_path.exists():
        existing_acc = acc_path.read_bytes()
        existing_rej = rej_path.read_bytes() if rej_path.exists() else b""
        if existing_acc == acc_bytes and existing_rej == rej_bytes:
            print("[idempotent] candidate files byte-identical; re-running validation + reports only.")
        else:
            print("[conflict] existing pilot differs from a fresh rebuild; not overwriting.")
            print(f"  conflicting files: {acc_path}" + (f", {rej_path}" if existing_rej != rej_bytes else ""))
            raise SystemExit(1)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        acc_path.write_bytes(acc_bytes)
        rej_path.write_bytes(rej_bytes)
        print(f"[build] wrote {len(records)} accepted, {len(rejections)} rejected to {out_dir}")

    # Deterministic rebuild into a temp dir; compare file hashes.
    tmp_dir = out_dir / ".rebuild_check"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rec2, rej2, _ = _build(args)
    (tmp_dir / "accepted.jsonl").write_bytes(_jsonl_bytes(rec2))
    (tmp_dir / "rejected.jsonl").write_bytes(_jsonl_bytes(rej2))
    determinism = {
        "accepted_sha256_main": sha256_of_file(acc_path),
        "accepted_sha256_rebuild": sha256_of_file(tmp_dir / "accepted.jsonl"),
        "rejected_sha256_main": sha256_of_file(rej_path),
        "rejected_sha256_rebuild": sha256_of_file(tmp_dir / "rejected.jsonl"),
    }
    determinism["byte_identical"] = (
        determinism["accepted_sha256_main"] == determinism["accepted_sha256_rebuild"]
        and determinism["rejected_sha256_main"] == determinism["rejected_sha256_rebuild"]
    )
    for p in (tmp_dir / "accepted.jsonl", tmp_dir / "rejected.jsonl"):
        p.unlink()
    tmp_dir.rmdir()

    # Read back from disk; validate the serialized-and-read-back records.
    rb_records, acc_errors = read_jsonl_strict(acc_path)
    rb_rejections, rej_errors = read_jsonl_strict(rej_path)
    json_valid = not acc_errors and not rej_errors

    mapping = load_mapping(_resolve(args.mapping))
    cache_root = _resolve(args.cache_root)
    resolver = DbMetadataResolver(str(cache_root) if cache_root.exists() else None)
    s_checks = structural_checks(rb_records, expected=args.limit)
    sem_checks, warnings = semantic_checks(rb_records, mapping, resolver=resolver)
    # v3 sampling (target_per_chart set) is designed to exclude exact-goal and
    # near-duplicates by construction, so those become mandatory in that mode.
    dup_checks, dup_findings = duplicate_checks(rb_records, strict=args.target_per_chart is not None)
    lk_checks, lk_findings = leakage_checks(rb_records, _load_eval_sources())
    dist = distribution_rows(rb_records)

    all_checks = {"structural": s_checks, "semantic": sem_checks, "duplicate": dup_checks, "leakage": lk_checks}
    failed_mandatory = [c for group in all_checks.values() for c in group
                        if c["severity"] == "mandatory" and not c["passed"]]

    sampling = meta.get("sampling_fallback")
    if args.target_per_chart is not None and sampling:
        short = {c: n for c, n in sampling["chart_counts"].items() if n != args.target_per_chart}
        if short:
            failed_mandatory.append({
                "check": "exact_per_chart_target", "passed": False, "severity": "mandatory",
                "n": len(short), "item_ids": [],
                "detail": f"chart counts != {args.target_per_chart}: {short}",
            })
        over_cap = {db: n for db, n in sampling["db_counts"].items() if n > args.db_cap}
        if over_cap and not sampling["fallbacks"]:
            failed_mandatory.append({
                "check": "database_cap_satisfied", "passed": False, "severity": "mandatory",
                "n": len(over_cap), "item_ids": [],
                "detail": f"databases over cap ({args.db_cap}) without a documented fallback: {over_cap}",
            })

    if not json_valid:
        failed_mandatory.append(_dict_check("json_valid_readback", acc_errors + rej_errors))
    if not determinism["byte_identical"]:
        failed_mandatory.append({"check": "deterministic_rebuild", "passed": False, "severity": "mandatory",
                                 "n": 0, "item_ids": [], "detail": "rebuild not byte-identical"})
    passed = not failed_mandatory

    built_at = datetime.now(timezone.utc).isoformat()
    report = {
        "pilot_dir": out_dir.as_posix(),
        "built_at": built_at,
        "passed": passed,
        "json_valid_readback": json_valid,
        "counts": {"selected": len(rb_records), "rejected_total": len(rb_rejections),
                   "accepted_total": meta["counts"]["accepted_total"]},
        "mapping_version": meta["mapping_version"],
        "task_rule_version": meta["task_rule_version"],
        "db_metadata_available": meta["db_metadata_available"],
        "determinism": determinism,
        "checks": all_checks,
        "failed_mandatory": failed_mandatory,
        "distributions": dist,
        "n_warnings": len(warnings),
        "sampling_fallback": sampling,
    }

    # Write artifacts.
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(report, reports_dir / "validation_report.json")
    _write_md(report, reports_dir / "validation_report.md")
    _write_jsonl(dup_findings, reports_dir / "duplicate_report.jsonl")
    _write_jsonl(lk_findings, reports_dir / "leakage_report.jsonl")
    _write_jsonl(warnings, reports_dir / "warnings.jsonl")
    _write_distribution_csv(dist, reports_dir / "distribution_report.csv")
    _write_audit_template(rb_records, reports_dir / "manual_audit_template.csv")

    # Also refresh the top-level machine-readable summaries.
    write_json({
        "built_at": built_at, "source": "nvbench", "usage_tier": "train_aug",
        **meta, "determinism": determinism, "passed": passed,
    }, out_dir / "manifest.json")
    write_json({"n_items": len(rb_records),
                "distributions": [{"dimension": d, "value": v, "count": c} for d, v, c in dist]},
               out_dir / "distribution_report.json")

    print(f"[validate] json_valid={json_valid} byte_identical={determinism['byte_identical']} "
          f"warnings={len(warnings)}")
    print(f"[result] {'PASS' if passed else 'FAIL'}")
    if not passed:
        for c in failed_mandatory:
            print(f"  FAILED {c['check']}: {c['detail']} ids={c['item_ids'][:10]}")
    print(f"[reports] {reports_dir}")
    raise SystemExit(0 if passed else 1)


def _write_jsonl(records, path: Path) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def _dict_check(name, errors):
    return {"check": name, "passed": False, "severity": "mandatory", "n": len(errors),
            "item_ids": [], "detail": "; ".join(errors[:5])}


if __name__ == "__main__":
    main()
