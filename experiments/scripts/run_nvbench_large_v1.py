"""Phase 2C: finalize maximum valid Tier-A nvBench corpus.

Consumes Phase 1 quality pool without re-tiering or weakening rules. Applies
controlled maximum-two-per-source-group selection, accepts documented 1,819-row
maximum valid corpus, creates deterministic source-group-disjoint 70/15/15
train/validation/held-out-test splits, and writes validation, provenance,
human-evaluation-input, and hash artifacts under ``--out``.

Held-out test remains in-domain nvBench evidence. Literature-derived L1 file is
separate limited external evidence and is never modified or used for training.
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

from src.data_pipeline.frozen_validation import read_jsonl_strict, sha256_of_file  # noqa: E402
from src.data_pipeline.nvbench_large_v1 import (  # noqa: E402
    NORMALIZED_CHART_TYPES,
    _brief_of,
    compute_availability,
    select_human_eval_sample,
    select_large_v1,
    select_spotcheck_sample,
    split_train_val_test,
    validate_phase1_input,
)
from src.data_pipeline.nvbench_pilot import (  # noqa: E402
    _mapping0,
    _prov,
    distribution_rows,
    duplicate_checks,
    leakage_checks,
    semantic_checks,
    structural_checks,
)
from src.data_pipeline.nvbench_source import DbMetadataResolver, load_mapping  # noqa: E402
from src.utils.io import write_json  # noqa: E402

DEFAULT_QUALITY_POOL_DIR = "data/staging/dashboard_v3/nvbench_quality_pool_final"
DEFAULT_MAPPING = "src/config/data/nvbench_mapping.yaml"
DEFAULT_CACHE_ROOT = "data/cache_external/nvbench/databases"
DEFAULT_EXTERNAL_L1 = "data/eval/l1_chart_effectiveness_v1.csv"
PREFERRED_TARGET = 2000
MINIMUM_ACCEPTABLE = 1800
EXPECTED_ACTUAL = 1819
FINAL_STATUS = "PASS_MAXIMUM_VALID_CORPUS_WITH_HELD_OUT_TEST"

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
    p = argparse.ArgumentParser(description="Phase 2C: finalize the 1,819-record Tier-A nvBench corpus.")
    p.add_argument("--out", default="data/staging/dashboard_v3/nvbench_large_v1")
    p.add_argument("--quality-pool-dir", default=DEFAULT_QUALITY_POOL_DIR)
    p.add_argument("--mapping", default=DEFAULT_MAPPING)
    p.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    p.add_argument("--external-l1", default=DEFAULT_EXTERNAL_L1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preferred-target", "--total", dest="preferred_target", type=int, default=PREFERRED_TARGET)
    p.add_argument("--minimum-acceptable", type=int, default=MINIMUM_ACCEPTABLE)
    p.add_argument("--expected-actual", type=int, default=EXPECTED_ACTUAL)
    p.add_argument("--db-cap", type=int, default=100)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument(
        "--max-per-group", type=int, default=2,
        help="Controlled Phase-2C policy. Must remain exactly 2.",
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


def _load_phase1(quality_pool_dir: Path):
    manifest = json.loads((quality_pool_dir / "manifest.json").read_text(encoding="utf-8"))
    hashes = json.loads((quality_pool_dir / "hashes.json").read_text(encoding="utf-8"))
    tier_a_path = quality_pool_dir / "tier_a_candidates.jsonl"
    summary_path = quality_pool_dir / "quality_pool_summary.json"
    tier_a_records = [json.loads(l) for l in tier_a_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    recomputed = {
        "tier_a_candidates": sha256_of_file(tier_a_path),
        "quality_pool_summary": sha256_of_file(summary_path),
    }
    return manifest, hashes, recomputed, tier_a_records


_SPOT_SOURCE_COLS = [
    "item_id", "source_group_id", "source_record_id", "db_id", "original_query",
    "source_sql", "source_vql", "original_chart_label", "normalized_chart_type",
    "kpi", "raw_columns", "encoding", "filters", "sort", "grouping", "time_grain",
    "quality_score", "quality_evidence", "mandatory_failures", "warnings",
]
_SPOT_REVIEW_COLS = [
    "reviewer_id", "goal_fidelity", "raw_columns_correct", "kpi_correct",
    "chart_appropriate", "encoding_correct", "constraints_preserved",
    "source_fidelity", "design_validity", "accept", "error_category", "review_comment",
]


_MULTI_RECORD_COLS = [
    "source_group_id", "item_id_1", "item_id_2", "normalized_goal_1", "normalized_goal_2",
    "signature_1", "signature_2", "differing_components", "goal_similarity", "justification",
]


def _write_multi_record_groups_csv(pairs, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_MULTI_RECORD_COLS)
        w.writeheader()
        for p in pairs:
            w.writerow({
                "source_group_id": p["source_group_id"],
                "item_id_1": p["item_ids"][0], "item_id_2": p["item_ids"][1],
                "normalized_goal_1": p["normalized_goals"][0], "normalized_goal_2": p["normalized_goals"][1],
                "signature_1": p["signatures"][0], "signature_2": p["signatures"][1],
                "differing_components": ";".join(p["differing_components"]),
                "goal_similarity": p["goal_similarity"], "justification": p["justification"],
            })


def _write_multi_record_groups_md(pairs, path: Path) -> None:
    L = ["# Multi-Record Source Groups (Phase 2B controlled two-per-group policy)", ""]
    L.append(f"Total groups contributing 2 records: {len(pairs)}")
    L.append("")
    for p in pairs:
        L.append(f"## group `{p['source_group_id']}`")
        L.append(f"- items: `{p['item_ids'][0]}` / `{p['item_ids'][1]}`")
        L.append(f"- goal 1: {p['normalized_goals'][0]}")
        L.append(f"- goal 2: {p['normalized_goals'][1]}")
        L.append(f"- goal similarity: {p['goal_similarity']}")
        L.append(f"- differing components: {', '.join(p['differing_components']) or 'none'}")
        L.append(f"- justification: {p['justification']}")
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def _write_spotcheck_csv(sample, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SPOT_SOURCE_COLS + _SPOT_REVIEW_COLS)
        w.writeheader()
        for rec in sample:
            r = rec.get("record") or {}
            prov = _prov(r)
            m = _mapping0(r)
            brief = _brief_of(rec)
            enc = m.get("encoding") or {}
            constraints = prov.get("constraints") or {}
            row = {
                "item_id": rec["item_id"], "source_group_id": rec["source_group_id"],
                "source_record_id": rec["source_record_id"], "db_id": rec["db_id"],
                "original_query": prov.get("nl_query", ""),
                "source_sql": ((prov.get("vis_query") or {}).get("data_part") or {}).get("sql_part", ""),
                "source_vql": (prov.get("vis_query") or {}).get("VQL", ""),
                "original_chart_label": prov.get("original_chart_label", ""),
                "normalized_chart_type": rec["chart_type"],
                "kpi": json.dumps(brief.get("kpis", []), ensure_ascii=False),
                "raw_columns": json.dumps(brief.get("columns", []), ensure_ascii=False),
                "encoding": json.dumps(enc, ensure_ascii=False),
                "filters": json.dumps(constraints.get("filters", []), ensure_ascii=False),
                "sort": json.dumps(constraints.get("sort"), ensure_ascii=False),
                "grouping": json.dumps(prov.get("grouping", {}), ensure_ascii=False),
                "time_grain": json.dumps(constraints.get("time_grain"), ensure_ascii=False),
                "quality_score": rec.get("quality_score", ""),
                "quality_evidence": json.dumps(rec.get("evidence", {}), ensure_ascii=False, default=str),
                "mandatory_failures": json.dumps(rec.get("failed_rules", []), ensure_ascii=False),
                "warnings": json.dumps(rec.get("warnings", []), ensure_ascii=False),
            }
            for c in _SPOT_REVIEW_COLS:
                row[c] = ""  # human-review fields intentionally empty
            w.writerow(row)


_HUMAN_REVIEW_COLS = [
    "reviewer_id", "goal_fidelity", "chart_appropriate", "encoding_correct",
    "source_fidelity", "overall_rating", "error_category", "review_comment",
]


def _project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(_PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _record_with_split(rec, split: str) -> dict:
    return {**(rec.get("record") or {}), "split": split}


def _human_eval_item(rec) -> dict:
    record = rec.get("record") or {}
    provenance = _prov(record)
    evidence = {
        "source_record_id": rec.get("source_record_id"),
        "source_group_id": rec.get("source_group_id"),
        "db_id": rec.get("db_id"),
        "original_query": provenance.get("nl_query", ""),
        "source_sql": ((provenance.get("vis_query") or {}).get("data_part") or {}).get("sql_part", ""),
        "source_vql": (provenance.get("vis_query") or {}).get("VQL", ""),
        "original_chart_label": provenance.get("original_chart_label", ""),
        "normalized_chart_type": rec.get("chart_type"),
        "quality_tier": rec.get("quality_tier"),
        "quality_score": rec.get("quality_score"),
        "quality_evidence": rec.get("evidence") or {},
    }
    return {
        "item_id": rec["item_id"],
        "input_brief": record.get("brief") or {},
        "source_evidence": evidence,
        "provenance": provenance,
        "review": {field: "" for field in _HUMAN_REVIEW_COLS},
    }


def _write_human_eval_files(sample, jsonl_path: Path, csv_path: Path) -> None:
    rows = [_human_eval_item(rec) for rec in sample]
    _write_jsonl(rows, jsonl_path)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["item_id", "input_brief", "source_evidence", "provenance"] + _HUMAN_REVIEW_COLS
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {
                "item_id": row["item_id"],
                "input_brief": json.dumps(row["input_brief"], ensure_ascii=False),
                "source_evidence": json.dumps(row["source_evidence"], ensure_ascii=False),
                "provenance": json.dumps(row["provenance"], ensure_ascii=False),
            }
            csv_row.update(row["review"])
            writer.writerow(csv_row)


def _distribution_summary(selected, train, val, test) -> dict:
    buckets = {"all": selected, "train": train, "val": val, "test": test}
    summary = {}
    for name, records in buckets.items():
        summary[name] = {
            "count": len(records),
            "unique_source_groups": len({rec["source_group_id"] for rec in records}),
            "chart_type": dict(sorted(collections.Counter(rec["chart_type"] for rec in records).items())),
            "database": dict(sorted(collections.Counter(rec["db_id"] for rec in records).items())),
        }
    return summary


def _write_independent_reference(path: Path, external_l1: Path) -> dict:
    reference = {
        "held_out_nvbench_test": {
            "path": "test.jsonl",
            "domain": "nvBench",
            "classification": "in-domain group-disjoint held-out test",
            "fully_external": False,
            "training_use": "prohibited",
            "validation_use": "prohibited",
            "enrichment_target_use": "prohibited",
            "prompt_selection_use": "prohibited",
            "hyperparameter_selection_use": "prohibited",
            "retrieval_example_use": "prohibited",
        },
        "literature_based_human_effectiveness_gold": {
            "path": _project_relative(external_l1),
            "sha256": sha256_of_file(external_l1),
            "classification": "external literature-based L1 chart-effectiveness evidence",
            "fully_external": True,
            "coverage": "limited to covered chart-selection task/data-shape cells",
            "training_use": "prohibited",
        },
    }
    write_json(reference, path)
    return reference


def _hash_outputs(out_dir: Path, relative_paths: list[str]) -> dict:
    return {
        relative: sha256_of_file(out_dir / relative)
        for relative in sorted(relative_paths)
    }
def main() -> None:
    args = parse_args()
    fixed_policy = {
        "seed": 42,
        "preferred_target": PREFERRED_TARGET,
        "minimum_acceptable": MINIMUM_ACCEPTABLE,
        "expected_actual": EXPECTED_ACTUAL,
        "max_per_group": 2,
        "val_fraction": 0.15,
        "test_fraction": 0.15,
    }
    actual_policy = {
        "seed": args.seed,
        "preferred_target": args.preferred_target,
        "minimum_acceptable": args.minimum_acceptable,
        "expected_actual": args.expected_actual,
        "max_per_group": args.max_per_group,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
    }
    if actual_policy != fixed_policy:
        raise SystemExit(f"Phase 2C policy is fixed; expected {fixed_policy}, got {actual_policy}")

    out_dir = _resolve(args.out)
    if "frozen" in out_dir.parts:
        raise SystemExit(f"refusing to write into a frozen path: {out_dir}")

    quality_pool_dir = _resolve(args.quality_pool_dir)
    external_l1 = _resolve(args.external_l1)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    built_at = datetime.now(timezone.utc).isoformat()

    if not external_l1.is_file():
        status = "MISSING_EXTERNAL_L1_FILE"
        write_json({
            "built_at": built_at,
            "passed": False,
            "status": status,
            "requested_external_l1": str(external_l1),
        }, out_dir / "manifest.json")
        print(f"[FAIL] {status}: {external_l1}")
        raise SystemExit(1)

    manifest1, hashes1, recomputed1, tier_a_records = _load_phase1(quality_pool_dir)
    phase1_checks = validate_phase1_input(manifest1, hashes1, recomputed1, tier_a_records)
    phase1_ok = all(check["passed"] for check in phase1_checks)
    write_json({
        "quality_pool_dir": str(quality_pool_dir),
        "phase1_manifest": manifest1,
        "phase1_hashes": hashes1,
        "phase1_input_checks": phase1_checks,
    }, out_dir / "quality_pool_reference.json")

    if not phase1_ok:
        status = "PHASE1_INPUT_INVALID"
        write_json({
            "built_at": built_at,
            "passed": False,
            "status": status,
            "phase1_checks": phase1_checks,
        }, out_dir / "manifest.json")
        print(f"[FAIL] {status}: {[check for check in phase1_checks if not check['passed']]}")
        raise SystemExit(1)

    eval_sources = _load_eval_sources()
    availability = compute_availability(tier_a_records, eval_sources, seed=args.seed)
    selected, sampling_report = select_large_v1(
        tier_a_records,
        eval_sources,
        seed=args.seed,
        total=args.preferred_target,
        minimum_acceptable=args.minimum_acceptable,
        db_cap=args.db_cap,
        max_per_group=args.max_per_group,
    )

    sampling_summary = {
        key: value for key, value in sampling_report.items()
        if key != "multi_record_groups"
    }
    attrition_report = {
        "preferred_target": args.preferred_target,
        "minimum_acceptable": args.minimum_acceptable,
        "actual_selected": sampling_report.get("achieved_total"),
        "selection_status": sampling_report.get("status"),
        "availability": availability,
        "sampling_summary": sampling_summary,
    }
    write_json(attrition_report, reports_dir / "selection_attrition.json")
    lines = [
        "# Phase 2C — Selection Attrition Report",
        "",
        f"- preferred_target: {args.preferred_target}",
        f"- minimum_acceptable: {args.minimum_acceptable}",
        f"- actual_selected: {sampling_report.get('achieved_total')}",
        f"- selection_status: {sampling_report.get('status')}",
        "",
    ]
    for chart in NORMALIZED_CHART_TYPES:
        lines.append(f"## {chart}")
        for key, value in availability[chart].items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    (reports_dir / "selection_attrition.md").write_text("\n".join(lines), encoding="utf-8")

    if selected is None:
        status = "INSUFFICIENT_DISTINCT_TIER_A_RECORDS"
        validation_report = {
            "passed": False,
            "status": status,
            "sampling_report": sampling_summary,
            "checks": [],
        }
        write_json(validation_report, reports_dir / "validation_report.json")
        (reports_dir / "validation_report.md").write_text(
            f"# Phase 2C — Validation Report\n\nStatus: **FAIL** ({status})\n",
            encoding="utf-8",
        )
        write_json({
            "built_at": built_at,
            "source": "nvbench",
            "kind": "large_v1",
            "passed": False,
            "status": status,
            "sampling_report": sampling_summary,
        }, out_dir / "manifest.json")
        print(f"[FAIL] {status}: achieved={sampling_report.get('achieved_total')}")
        raise SystemExit(1)

    if len(selected) != args.expected_actual:
        status = "UNEXPECTED_MAXIMUM_VALID_CORPUS_SIZE"
        detail = (
            f"deterministic selection produced {len(selected)} records; "
            f"expected exactly {args.expected_actual}"
        )
        write_json({
            "passed": False,
            "status": status,
            "detail": detail,
            "sampling_report": sampling_summary,
        }, reports_dir / "validation_report.json")
        write_json({
            "built_at": built_at,
            "passed": False,
            "status": status,
            "detail": detail,
        }, out_dir / "manifest.json")
        print(f"[FAIL] {status}: {detail}")
        raise SystemExit(1)

    train, val, test, split_report = split_train_val_test(
        selected,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    split_of = {
        rec["item_id"]: split_name
        for split_name, records in (("train", train), ("val", val), ("test", test))
        for rec in records
    }
    all_records = [_record_with_split(rec, split_of[rec["item_id"]]) for rec in selected]
    train_records = [_record_with_split(rec, "train") for rec in train]
    val_records = [_record_with_split(rec, "val") for rec in val]
    test_records = [_record_with_split(rec, "test") for rec in test]

    all_selected_path = out_dir / "all_selected.jsonl"
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    test_path = out_dir / "test.jsonl"
    all_selected_path.write_bytes(_jsonl_bytes(all_records))
    train_path.write_bytes(_jsonl_bytes(train_records))
    val_path.write_bytes(_jsonl_bytes(val_records))
    test_path.write_bytes(_jsonl_bytes(test_records))

    mapping = load_mapping(_resolve(args.mapping))
    cache_root = _resolve(args.cache_root)
    resolver = DbMetadataResolver(str(cache_root) if cache_root.exists() else None)

    structural = [
        check for check in structural_checks(all_records, expected=args.expected_actual)
        if check["check"] not in {"unique_source_group_count", "splits_train_val_only"}
    ]
    structural.append({
        "check": "splits_train_val_test_only",
        "passed": {record.get("split") for record in all_records} == {"train", "val", "test"},
        "severity": "mandatory",
        "n": 0,
        "item_ids": [],
        "detail": f"splits present: {sorted({record.get('split') for record in all_records})}",
    })
    semantic, warnings = semantic_checks(all_records, mapping, resolver=resolver)
    duplicate, duplicate_findings = duplicate_checks(all_records, strict=True)
    duplicate = [check for check in duplicate if check["check"] != "no_duplicate_source_group_ids"]
    leakage, leakage_findings = leakage_checks(all_records, eval_sources)

    group_counts = collections.Counter(rec["source_group_id"] for rec in selected)
    over_cap_groups = sorted(group for group, count in group_counts.items() if count > args.max_per_group)
    two_record_groups = sorted(group for group, count in group_counts.items() if count == 2)
    evidenced_groups = {
        pair["source_group_id"] for pair in sampling_report.get("multi_record_groups", [])
    }
    unevidenced_groups = sorted(set(two_record_groups) - evidenced_groups)
    below_min = [rec["item_id"] for rec in selected if rec.get("quality_score", 0) < 90]
    mandatory_quality_failures = [rec["item_id"] for rec in selected if rec.get("failed_rules")]
    absent_source_goal_text = [
        rec["item_id"] for rec in selected
        if not " ".join(str((_brief_of(rec).get("goals") or [""])[0]).split())
    ]
    test_ids = {rec["item_id"] for rec in test}
    train_val_ids = {rec["item_id"] for rec in train + val}
    test_groups = {rec["source_group_id"] for rec in test}
    train_val_groups = {rec["source_group_id"] for rec in train + val}

    human_eval_sample = select_human_eval_sample(test, seed=args.seed, size=40)
    human_jsonl = reports_dir / "human_eval_test_items_40.jsonl"
    human_csv = reports_dir / "human_eval_test_items_40.csv"
    _write_human_eval_files(human_eval_sample, human_jsonl, human_csv)
    human_rows = [_human_eval_item(rec) for rec in human_eval_sample]
    review_fields_empty = all(
        all(value == "" for value in row["review"].values()) for row in human_rows
    )

    quality = [
        {
            "check": "exactly_1819_selected_records",
            "passed": len(selected) == EXPECTED_ACTUAL,
            "severity": "mandatory",
            "n": 0 if len(selected) == EXPECTED_ACTUAL else 1,
            "item_ids": [],
            "detail": f"{len(selected)} selected",
        },
        {
            "check": "maximum_valid_corpus_acceptance_policy",
            "passed": sampling_report.get("status") == "maximum_valid_corpus_accepted"
                      and len(selected) >= args.minimum_acceptable
                      and len(selected) < args.preferred_target,
            "severity": "mandatory",
            "n": 0,
            "item_ids": [],
            "detail": (
                f"preferred={args.preferred_target}; minimum={args.minimum_acceptable}; "
                f"actual={len(selected)}; selection_status={sampling_report.get('status')}"
            ),
        },
        {
            "check": "train_val_test_equals_1819",
            "passed": len(train) + len(val) + len(test) == EXPECTED_ACTUAL,
            "severity": "mandatory",
            "n": 0,
            "item_ids": [],
            "detail": f"train={len(train)} val={len(val)} test={len(test)}",
        },
        {
            "check": "all_selected_tier_a_no_fallback",
            "passed": all(rec.get("quality_tier") == "A" for rec in selected),
            "severity": "mandatory",
            "n": 0,
            "item_ids": [],
            "detail": "Tier A only; no Tier-B/Tier-C fallback",
        },
        {
            "check": "all_scores_at_least_90",
            "passed": not below_min,
            "severity": "mandatory",
            "n": len(below_min),
            "item_ids": below_min,
            "detail": "quality_score below 90",
        },
        {
            "check": "no_mandatory_quality_failure",
            "passed": not mandatory_quality_failures,
            "severity": "mandatory",
            "n": len(mandatory_quality_failures),
            "item_ids": mandatory_quality_failures,
            "detail": "non-empty failed_rules",
        },
        {
            "check": "source_records_without_nl_query_text",
            "passed": not absent_source_goal_text,
            "severity": "warning",
            "n": len(absent_source_goal_text),
            "item_ids": absent_source_goal_text,
            "detail": "missing source NL query text preserved; SQL/VQL evidence remains present",
        },
        {
            "check": "maximum_two_records_per_source_group",
            "passed": not over_cap_groups,
            "severity": "mandatory",
            "n": len(over_cap_groups),
            "item_ids": [],
            "detail": f"groups over cap: {over_cap_groups[:10]}",
        },
        {
            "check": "same_group_pairs_semantically_distinct",
            "passed": not unevidenced_groups,
            "severity": "mandatory",
            "n": len(unevidenced_groups),
            "item_ids": [],
            "detail": f"two-record groups without evidence: {unevidenced_groups[:10]}",
        },
        {
            "check": "zero_cross_split_source_group_leakage",
            "passed": not split_report["cross_split_group_overlap"],
            "severity": "mandatory",
            "n": len(split_report["cross_split_group_overlap"]),
            "item_ids": [],
            "detail": "source groups shared across train/validation/test",
        },
        {
            "check": "test_excluded_from_training_validation_and_enrichment_targets",
            "passed": not (test_ids & train_val_ids) and not (test_groups & train_val_groups),
            "severity": "mandatory",
            "n": len(test_ids & train_val_ids) + len(test_groups & train_val_groups),
            "item_ids": sorted(test_ids & train_val_ids),
            "detail": "held-out test membership is disjoint and frozen",
        },
        {
            "check": "human_eval_sample_exactly_40_test_items",
            "passed": len(human_eval_sample) == 40
                      and {rec["item_id"] for rec in human_eval_sample} <= test_ids,
            "severity": "mandatory",
            "n": 0 if len(human_eval_sample) == 40 else 1,
            "item_ids": [],
            "detail": f"{len(human_eval_sample)} source-backed test items",
        },
        {
            "check": "human_eval_review_fields_empty",
            "passed": review_fields_empty,
            "severity": "mandatory",
            "n": 0 if review_fields_empty else 1,
            "item_ids": [],
            "detail": "no model outputs or human ratings generated",
        },
        {
            "check": "external_l1_reference_present",
            "passed": external_l1.is_file(),
            "severity": "mandatory",
            "n": 0 if external_l1.is_file() else 1,
            "item_ids": [],
            "detail": _project_relative(external_l1),
        },
    ]

    checks = {
        "structural": structural,
        "semantic": semantic,
        "duplicate": duplicate,
        "leakage": leakage,
        "quality": quality,
    }
    failed_mandatory = [
        check for section in checks.values() for check in section
        if check["severity"] == "mandatory" and not check["passed"]
    ]
    passed = not failed_mandatory
    status = FINAL_STATUS if passed else "QUALITY_GATE_FAILED"

    distributions = _distribution_summary(selected, train, val, test)
    distribution_rows_all = distribution_rows(all_records)
    for split_name, records in (("train", train), ("val", val), ("test", test)):
        for chart, count in sorted(collections.Counter(rec["chart_type"] for rec in records).items()):
            distribution_rows_all.append(("chart_by_split", f"{split_name}:{chart}", count))
        for database, count in sorted(collections.Counter(rec["db_id"] for rec in records).items()):
            distribution_rows_all.append(("database_by_split", f"{split_name}:{database}", count))

    validation_report = {
        "passed": passed,
        "status": status,
        "counts": {
            "selected": len(selected),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "checks": checks,
        "failed_mandatory": failed_mandatory,
        "distributions": distributions,
        "n_warnings": len(warnings),
        "split_report": split_report,
        "sampling_summary": sampling_summary,
    }
    write_json(validation_report, reports_dir / "validation_report.json")
    lines = [
        "# Phase 2C — Validation Report",
        "",
        f"Result: **{'PASS' if passed else 'FAIL'}**",
        f"Status: `{status}`",
        "",
    ]
    for section, section_checks in checks.items():
        lines.append(f"## {section}")
        for check in section_checks:
            flag = "PASS" if check["passed"] else ("WARN" if check["severity"] == "warning" else "FAIL")
            lines.append(f"- [{flag}] `{check['check']}` (n={check['n']}) {check['detail']}")
        lines.append("")
    (reports_dir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")

    scores = [rec["quality_score"] for rec in selected]
    quality_report = {
        "rule_version": manifest1.get("rule_version"),
        "total_selected": len(selected),
        "quality_score_range": {"min": min(scores), "max": max(scores)},
        "achieved_distribution": sampling_report["achieved_distribution"],
    }
    write_json(quality_report, reports_dir / "quality_report.json")
    (reports_dir / "quality_report.md").write_text(
        "# Phase 2C — Quality Report\n\n"
        f"- rule_version: {quality_report['rule_version']}\n"
        f"- total_selected: {quality_report['total_selected']}\n"
        f"- quality_score_range: {quality_report['quality_score_range']}\n"
        f"- achieved_distribution: {quality_report['achieved_distribution']}\n",
        encoding="utf-8",
    )
    _write_distribution_csv(distribution_rows_all, reports_dir / "distribution_report.csv")
    _write_jsonl(duplicate_findings, reports_dir / "duplicate_report.jsonl")
    _write_jsonl(leakage_findings, reports_dir / "leakage_report.jsonl")
    _write_jsonl(warnings, reports_dir / "warnings.jsonl")

    spotcheck = select_spotcheck_sample(selected, seed=args.seed, size=30)
    _write_spotcheck_csv(spotcheck, reports_dir / "manual_spotcheck_template_30.csv")
    multi_record_groups = sampling_report.get("multi_record_groups", [])
    _write_multi_record_groups_csv(multi_record_groups, reports_dir / "multi_record_source_groups.csv")
    _write_multi_record_groups_md(multi_record_groups, reports_dir / "multi_record_source_groups.md")

    write_json({
        "n_items": len(all_records),
        "split_counts": validation_report["counts"],
        "distributions": distributions,
    }, out_dir / "distribution_report.json")
    independent_reference = _write_independent_reference(
        out_dir / "independent_evaluation_reference.json",
        external_l1,
    )

    hash_paths = [
        "all_selected.jsonl",
        "train.jsonl",
        "val.jsonl",
        "test.jsonl",
        "distribution_report.json",
        "independent_evaluation_reference.json",
        "quality_pool_reference.json",
        "reports/validation_report.json",
        "reports/validation_report.md",
        "reports/distribution_report.csv",
        "reports/duplicate_report.jsonl",
        "reports/leakage_report.jsonl",
        "reports/selection_attrition.json",
        "reports/selection_attrition.md",
        "reports/multi_record_source_groups.csv",
        "reports/multi_record_source_groups.md",
        "reports/manual_spotcheck_template_30.csv",
        "reports/human_eval_test_items_40.jsonl",
        "reports/human_eval_test_items_40.csv",
        "reports/quality_report.json",
        "reports/quality_report.md",
        "reports/warnings.jsonl",
    ]
    output_hashes = _hash_outputs(out_dir, hash_paths)
    write_json({"algorithm": "sha256", "files": output_hashes}, out_dir / "hashes.json")

    manifest = {
        "built_at": built_at,
        "source": "nvbench",
        "kind": "large_v1_phase_2c",
        "passed": passed,
        "status": status,
        "preferred_target": args.preferred_target,
        "minimum_acceptable": args.minimum_acceptable,
        "actual_selected": len(selected),
        "seed": args.seed,
        "max_per_group": args.max_per_group,
        "split_algorithm_version": split_report["split_algorithm_version"],
        "split_counts": validation_report["counts"],
        "split_percentages": split_report["actual_percentages"],
        "unique_source_groups_per_split": split_report["unique_source_groups"],
        "test_membership_sha256": split_report["test_membership_sha256"],
        "chart_distributions": {
            split_name: summary["chart_type"] for split_name, summary in distributions.items()
        },
        "database_distributions": {
            split_name: summary["database"] for split_name, summary in distributions.items()
        },
        "quality_rule_version": manifest1.get("rule_version"),
        "external_l1_reference_path": independent_reference["literature_based_human_effectiveness_gold"]["path"],
        "external_l1_sha256": independent_reference["literature_based_human_effectiveness_gold"]["sha256"],
        "output_hashes": output_hashes,
        "evaluation_policy": (
            "The nvBench corpus was divided into source-group-disjoint training, validation, and held-out "
            "test splits. The held-out test is an in-domain nvBench evaluation set rather than a fully "
            "external benchmark. No test source group is used for training, validation, enrichment targets, "
            "prompt selection, or hyperparameter selection."
        ),
        "external_evidence_policy": (
            "A separate literature-based human-effectiveness gold set is maintained as limited external "
            "evidence for covered chart-selection tasks."
        ),
    }
    write_json(manifest, out_dir / "manifest.json")

    print(
        f"[result] {'PASS' if passed else 'FAIL'} status={status} selected={len(selected)} "
        f"train={len(train)} val={len(val)} test={len(test)}"
    )
    print(f"[reports] {reports_dir}")
    raise SystemExit(0 if passed else 1)

if __name__ == "__main__":
    main()
