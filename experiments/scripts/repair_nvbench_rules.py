"""Rebuild and audit the corrected nvBench Phase-1/Phase-2C corpus.

This script preserves the pre-repair corpus in memory, rebuilds the quality pool,
repairs Phase 2C using Tier-A-only replacements, emits the rule-repair reports,
then performs an independent scratch rebuild and removes only that scratch tree.
It never starts Phase 3 or invokes an LLM.
"""

from __future__ import annotations

import collections
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.leakage_similarity import brief_text, char_ngrams, jaccard  # noqa: E402

QUALITY_DIR = _PROJECT_ROOT / "data/staging/dashboard_v3/nvbench_quality_pool_final"
SELECTED_DIR = _PROJECT_ROOT / "data/staging/dashboard_v3/nvbench_large_v1"
REPAIR_DIR = _PROJECT_ROOT / "data/staging/dashboard_v3/nvbench_rule_repair"
QUALITY_SCRIPT = _PROJECT_ROOT / "experiments/scripts/rebuild_nvbench_quality_pool_final.py"
SELECT_SCRIPT = _PROJECT_ROOT / "experiments/scripts/run_nvbench_large_v1.py"

SCATTER_FIXTURES = [
    "nvbench:1631:query:1",
    "nvbench:1633:query:2",
    "nvbench:1708:query:0",
    "nvbench:1709:query:2",
    "nvbench:1782:query:4",
]
VALID_SCATTER_FIXTURE = "nvbench:292:query:1"
BIN_FIXTURES = [
    "nvbench:1330@y_name@ASC:query:0",
    "nvbench:184@y_name@ASC:query:2",
    "nvbench:416@x_name@DESC:query:3",
    "nvbench:851@y_name@ASC:query:1",
    "nvbench:867:query:5",
]
TRUE_DIMENSION_CONFLICT_FIXTURES = [
    "nvbench:2164@y_name@ASC:query:1",
    "nvbench:3219:query:1",
    "nvbench:3257:query:0",
]
LIMIT_FIXTURES = ["nvbench:2782:query:4", VALID_SCATTER_FIXTURE]
TIME_CONFLICT_FIXTURE = "nvbench:3008@x_name@DESC:query:0"
PIE_FIXTURE = "nvbench:1325:query:1"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quality_rows(directory: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for filename in ("tier_a_candidates.jsonl", "tier_b_diagnostics.jsonl", "tier_c_rejected.jsonl"):
        for row in _load_jsonl(directory / filename):
            result[row["item_id"]] = row
    return result


def _constraints(row: dict | None) -> dict:
    if not row:
        return {}
    record = row.get("record") or row
    return (((record.get("brief") or {}).get("extra") or {}).get("provenance") or {}).get("constraints") or {}


def _grouping(row: dict | None) -> dict:
    if not row:
        return {}
    record = row.get("record") or row
    return (((record.get("brief") or {}).get("extra") or {}).get("provenance") or {}).get("grouping") or {}


def _chart_of_flat(record: dict) -> str:
    mappings = (record.get("recommendation") or {}).get("kpi_chart_mapping") or []
    return str((mappings[0] if mappings else {}).get("chart_type") or "")


def _group_of_flat(record: dict) -> str:
    return str((((record.get("brief") or {}).get("extra") or {}).get("provenance") or {}).get("source_group_id") or "")


def _run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=_PROJECT_ROOT, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode:
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        raise SystemExit(result.returncode)


def _fixture_state(item_id: str, rows: dict[str, dict], selected_ids: set[str]) -> dict:
    row = rows.get(item_id)
    if not row:
        return {"item_id": item_id, "present": False, "selected": False}
    constraints = _constraints(row)
    return {
        "item_id": item_id,
        "present": True,
        "tier": row.get("quality_tier"),
        "score": row.get("quality_score"),
        "failed_rules": row.get("failed_rules") or [],
        "selected": item_id in selected_ids,
        "sort": constraints.get("sort"),
        "limit": constraints.get("limit"),
        "having": constraints.get("having") or [],
        "time_grain": constraints.get("time_grain"),
        "grouping": _grouping(row),
    }


def _near_duplicate_pairs(records: list[dict], threshold: float = 0.8) -> list[dict]:
    ngrams = [frozenset(char_ngrams(brief_text(record.get("brief") or {}))) for record in records]
    pairs = []
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            similarity = jaccard(ngrams[left], ngrams[right])
            if similarity >= threshold:
                pairs.append({
                    "item_id_1": records[left]["item_id"],
                    "item_id_2": records[right]["item_id"],
                    "similarity": round(similarity, 6),
                })
    return pairs


def _independent_rebuild() -> dict:
    scratch = REPAIR_DIR / "_scratch_rebuild"
    repair_resolved = REPAIR_DIR.resolve()
    scratch_resolved = scratch.resolve()
    if scratch_resolved.parent != repair_resolved or scratch_resolved.name != "_scratch_rebuild":
        raise RuntimeError(f"unsafe scratch path: {scratch_resolved}")
    if scratch.exists():
        shutil.rmtree(scratch)
    quality_scratch = scratch / "quality"
    selected_scratch = scratch / "selected"
    try:
        _run([sys.executable, str(QUALITY_SCRIPT), "--out", str(quality_scratch)])
        _run([
            sys.executable,
            str(SELECT_SCRIPT),
            "--out", str(selected_scratch),
            "--quality-pool-dir", str(quality_scratch),
            "--baseline-selected", str(SELECTED_DIR / "all_selected.jsonl"),
        ])
        comparisons = []
        pairs = [
            (QUALITY_DIR / "tier_a_candidates.jsonl", quality_scratch / "tier_a_candidates.jsonl"),
            (QUALITY_DIR / "tier_b_diagnostics.jsonl", quality_scratch / "tier_b_diagnostics.jsonl"),
            (QUALITY_DIR / "tier_c_rejected.jsonl", quality_scratch / "tier_c_rejected.jsonl"),
            (QUALITY_DIR / "quality_pool_summary.json", quality_scratch / "quality_pool_summary.json"),
            (SELECTED_DIR / "all_selected.jsonl", selected_scratch / "all_selected.jsonl"),
            (SELECTED_DIR / "train.jsonl", selected_scratch / "train.jsonl"),
            (SELECTED_DIR / "val.jsonl", selected_scratch / "val.jsonl"),
            (SELECTED_DIR / "test.jsonl", selected_scratch / "test.jsonl"),
            (SELECTED_DIR / "distribution_report.json", selected_scratch / "distribution_report.json"),
            (
                SELECTED_DIR / "reports/manual_spotcheck_template_30_v2.csv",
                selected_scratch / "reports/manual_spotcheck_template_30_v2.csv",
            ),
            (
                SELECTED_DIR / "reports/manual_spotcheck_protocol_v2.md",
                selected_scratch / "reports/manual_spotcheck_protocol_v2.md",
            ),
        ]
        for canonical, rebuilt in pairs:
            canonical_hash = _sha256(canonical)
            rebuilt_hash = _sha256(rebuilt)
            comparisons.append({
                "file": canonical.relative_to(_PROJECT_ROOT).as_posix(),
                "canonical_sha256": canonical_hash,
                "scratch_sha256": rebuilt_hash,
                "match": canonical_hash == rebuilt_hash,
            })
        return {"passed": all(item["match"] for item in comparisons), "files": comparisons}
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def _correct_incomplete_baseline_reports() -> None:
    """Correct promotion scope when the full historical v4 pool is unavailable.

    The five BIN fixtures are verifiable AI-precheck false rejections, but the
    AI precheck is not human gold and cannot support a corpus-wide inferred
    promotion count.  Preserve selected-corpus demotion/replacement accounting
    from the completed run and replace only the over-broad promotion/affected
    artifacts with the auditable scope.
    """
    new_quality = _quality_rows(QUALITY_DIR)
    new_selected = _load_jsonl(SELECTED_DIR / "all_selected.jsonl")
    selected_ids = {record["item_id"] for record in new_selected}
    promoted = []
    for item_id in BIN_FIXTURES:
        row = new_quality.get(item_id)
        if not row or row.get("quality_tier") != "A" or _grouping(row).get("grouping_origin") != "vql_bin":
            continue
        constraints = _constraints(row)
        promoted.append({
            "item_id": item_id,
            "before_decision": "AI_PRECHECK_REJECTED_DIAGNOSTIC",
            "before_tier": None,
            "after_tier": "A",
            "after_score": row.get("quality_score"),
            "after_failed_rules": row.get("failed_rules") or [],
            "selected_after": item_id in selected_ids,
            "promotion_basis": "VQL BIN supplies source-backed implicit visual grouping",
            "grouping_after": _grouping(row),
            "parser_constraints_after": {
                key: constraints.get(key)
                for key in ("sort", "limit", "having", "time_grain", "visual_grouping")
            },
            "comparison_scope": (
                "verified named AI-precheck false rejection; diagnostic audit is not human gold"
            ),
        })

    existing_affected = _load_jsonl(REPAIR_DIR / "affected_records.jsonl")
    affected = [
        row for row in existing_affected
        if row.get("selected_before") or row.get("selected_after")
    ]
    affected_by_id = {row["item_id"]: row for row in affected}
    for row in promoted:
        affected_by_id.setdefault(row["item_id"], row)
    affected = [affected_by_id[item_id] for item_id in sorted(affected_by_id)]

    constraint_report = _load_json(REPAIR_DIR / "constraint_validation_report.json")
    constraint_report["promoted_after_parser_fix_count"] = len(promoted)
    constraint_report["promotion_count_scope"] = (
        "verified named VQL-BIN AI-precheck false rejections only"
    )
    constraint_report["historical_full_quality_baseline_available"] = False

    summary = _load_json(REPAIR_DIR / "rule_change_summary.json")
    demoted_rows = _load_jsonl(REPAIR_DIR / "demoted_after_quality_fix.jsonl")
    selected_scatter_demotions = sorted({
        row["item_id"] for row in demoted_rows
        if "scatter_identifier_axis" in (row.get("after_failed_rules") or [])
    })
    scatter_report = _load_json(REPAIR_DIR / "scatter_validation_report.json")
    summary["promoted_after_parser_fix_count"] = len(promoted)
    summary["demoted_after_scatter_repair_count"] = len(selected_scatter_demotions)
    summary["demoted_after_scatter_repair_ids"] = selected_scatter_demotions
    summary["quality_pool_identifier_scatter_demotions"] = scatter_report.get(
        "demoted_identifier_scatter_count", 0
    )
    summary["promotion_count_scope"] = (
        "verified named VQL-BIN AI-precheck false rejections only; no corpus-wide v4 inference"
    )
    summary["historical_full_quality_baseline_available"] = False

    _write_jsonl(REPAIR_DIR / "affected_records.jsonl", affected)
    _write_jsonl(REPAIR_DIR / "promoted_after_parser_fix.jsonl", promoted)
    _write_json(REPAIR_DIR / "constraint_validation_report.json", constraint_report)
    _write_json(REPAIR_DIR / "rule_change_summary.json", summary)

    lines = [
        "# nvBench Rule Repair — Final Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Rule version: `{summary['rule_version']}`",
        f"- Verified parser promotions: {summary['promoted_after_parser_fix_count']}",
        "- Promotion scope: named VQL-BIN AI-precheck false rejections only; the full historical v4 row-level pool is unavailable.",
        f"- Previously selected records demoted by Scatter identifier rules: {summary['demoted_after_scatter_repair_count']}",
        f"- Scatter/Pie quality demotions: {summary['demoted_after_scatter_or_pie_quality_fix_count']}",
        f"- True constraint-conflict demotions: {summary['demoted_for_true_constraint_conflict_count']}",
        f"- Replacements: {summary['replacement_record_count']}",
        f"- Selected: {summary['final_selected_total']}",
        f"- Splits: {summary['final_split_counts']}",
        f"- Charts: {summary['final_chart_distribution']}",
        f"- Source groups: {summary['final_source_group_count']}",
        f"- Exact duplicates: {summary['exact_duplicate_count']}",
        f"- Near duplicates at or above 0.8: {summary['near_duplicate_count_at_or_above_0_8']}",
        f"- Evaluation leakage: {summary['evaluation_leakage_count']}",
        f"- Cross-split group leakage: {summary['cross_split_source_group_leakage_count']}",
        f"- Deterministic rebuild: {'PASS' if summary['deterministic_comparison']['passed'] else 'FAIL'}",
        "- Phase 3 started: no",
        "",
        "VQL BIN is treated as source-backed implicit visual grouping. SQL grouping is not fabricated. "
        "Identifier Scatter axes and unresolved source conflicts are demoted rather than rewritten.",
    ]
    (REPAIR_DIR / "rule_change_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_files = sorted(
        path for path in REPAIR_DIR.iterdir()
        if path.is_file() and path.name != "hashes.json"
    )
    _write_json(REPAIR_DIR / "hashes.json", {
        "algorithm": "sha256",
        "files": {path.name: _sha256(path) for path in report_files},
    })
    print(json.dumps(summary, indent=2))


def main() -> None:
    REPAIR_DIR.mkdir(parents=True, exist_ok=True)
    old_quality = _quality_rows(QUALITY_DIR)
    old_summary = _load_json(QUALITY_DIR / "quality_pool_summary.json")
    old_selected = _load_jsonl(SELECTED_DIR / "all_selected.jsonl")
    old_selected_ids = {record["item_id"] for record in old_selected}
    # A prior interrupted repair may already have replaced the Phase-1 v4 pool.
    # The original Phase-2C corpus is still intact and every one of its rows was
    # documented Tier A, so reconstruct the minimal pre-repair evidence needed
    # for selected-row demotion/replacement accounting rather than mislabeling
    # the interrupted v5 pool as the baseline.
    if old_summary.get("rule_version") != "nvbench_quality_v4":
        old_summary = {
            "rule_version": "nvbench_quality_v4",
            "total_candidates": 20986,
            "tier_a_count": 12905,
            "tier_b_count": 8062,
            "tier_c_count": 19,
        }
        old_quality = {}
        for record in old_selected:
            provenance = (((record.get("brief") or {}).get("extra") or {}).get("provenance") or {})
            old_quality[record["item_id"]] = {
                "item_id": record["item_id"],
                "quality_tier": "A",
                "quality_score": 100,
                "failed_rules": [],
                "chart_type": _chart_of_flat(record),
                "db_id": provenance.get("db_id"),
                "source_group_id": provenance.get("source_group_id"),
                "record": record,
            }

    _run([sys.executable, str(QUALITY_SCRIPT), "--out", str(QUALITY_DIR)])
    _run([
        sys.executable,
        str(SELECT_SCRIPT),
        "--out", str(SELECTED_DIR),
        "--quality-pool-dir", str(QUALITY_DIR),
        "--baseline-selected", str(SELECTED_DIR / "all_selected.jsonl"),
    ])

    new_quality = _quality_rows(QUALITY_DIR)
    new_summary = _load_json(QUALITY_DIR / "quality_pool_summary.json")
    new_selected = _load_jsonl(SELECTED_DIR / "all_selected.jsonl")
    new_selected_ids = {record["item_id"] for record in new_selected}
    new_selected_enriched = [new_quality[item_id] for item_id in sorted(new_selected_ids)]

    promoted = []
    demoted_quality = []
    demoted_constraint = []
    affected = []
    all_ids = sorted(set(old_quality) | set(new_quality))
    relevant_rules = {
        "scatter_identifier_axis", "invalid_scatter_axes", "identifier_as_measure",
        "missing_required_dimension", "missing_aggregate_condition", "time_grain_source_conflict",
        "source_conflict", "pie_not_part_to_whole",
    }
    for item_id in all_ids:
        before = old_quality.get(item_id)
        after = new_quality.get(item_id)
        before_tier = before.get("quality_tier") if before else None
        after_tier = after.get("quality_tier") if after else None
        before_failed = set((before or {}).get("failed_rules") or [])
        after_failed = set((after or {}).get("failed_rules") or [])
        old_constraints = _constraints(before)
        new_constraints = _constraints(after)
        parser_changed = any(
            old_constraints.get(key) != new_constraints.get(key)
            for key in ("sort", "limit", "having", "time_grain", "visual_grouping")
        )
        changed = before_tier != after_tier or before_failed != after_failed or parser_changed
        if not changed:
            continue
        summary = {
            "item_id": item_id,
            "before_tier": before_tier,
            "after_tier": after_tier,
            "before_score": (before or {}).get("quality_score"),
            "after_score": (after or {}).get("quality_score"),
            "before_failed_rules": sorted(before_failed),
            "after_failed_rules": sorted(after_failed),
            "selected_before": item_id in old_selected_ids,
            "selected_after": item_id in new_selected_ids,
            "parser_constraints_before": {key: old_constraints.get(key) for key in ("sort", "limit", "having", "time_grain")},
            "parser_constraints_after": {key: new_constraints.get(key) for key in ("sort", "limit", "having", "time_grain")},
        }
        affected.append(summary)
        if before_tier != "A" and after_tier == "A":
            promoted.append(summary)
        if before_tier == "A" and after_tier != "A" and after_failed & {
            "scatter_identifier_axis", "invalid_scatter_axes", "identifier_as_measure", "pie_not_part_to_whole",
        }:
            demoted_quality.append(summary)
        if before_tier == "A" and after_tier != "A" and after_failed & {
            "missing_required_dimension", "missing_aggregate_condition", "time_grain_source_conflict", "source_conflict",
        }:
            demoted_constraint.append(summary)

    removed_selected = sorted(old_selected_ids - new_selected_ids)
    added_selected = sorted(new_selected_ids - old_selected_ids)
    replacements = []
    for index, item_id in enumerate(added_selected):
        row = new_quality[item_id]
        replacements.append({
            "item_id": item_id,
            "replacement_for": removed_selected[index] if index < len(removed_selected) else None,
            "chart_type": row.get("chart_type"),
            "db_id": row.get("db_id"),
            "quality_tier": row.get("quality_tier"),
            "quality_score": row.get("quality_score"),
            "failed_rules": row.get("failed_rules") or [],
            "source_group_id": row.get("source_group_id"),
            "record": row.get("record"),
        })

    distribution = collections.Counter(_chart_of_flat(record) for record in new_selected)
    groups = collections.Counter(_group_of_flat(record) for record in new_selected)
    split_counts = collections.Counter(record.get("split") for record in new_selected)
    exact_duplicates = len(new_selected) - len({json.dumps(record.get("brief") or {}, sort_keys=True) for record in new_selected})
    near_duplicates = _near_duplicate_pairs(new_selected, threshold=0.8)
    cross_split: dict[str, set[str]] = collections.defaultdict(set)
    for record in new_selected:
        cross_split[_group_of_flat(record)].add(str(record.get("split")))
    cross_split_groups = sorted(group for group, splits in cross_split.items() if len(splits) > 1)

    scatter_report = {
        "rule_version": new_summary.get("rule_version"),
        "fixtures": [_fixture_state(item_id, new_quality, new_selected_ids) for item_id in SCATTER_FIXTURES],
        "valid_two_measure_fixture": _fixture_state(VALID_SCATTER_FIXTURE, new_quality, new_selected_ids),
        "demoted_identifier_scatter_count": len([
            row for row in new_quality.values()
            if "scatter_identifier_axis" in (row.get("failed_rules") or [])
        ]),
        "tier_a_invalid_identifier_scatter_ids": sorted(
            row["item_id"] for row in new_quality.values()
            if row.get("quality_tier") == "A"
            and set(row.get("failed_rules") or []) & {"scatter_identifier_axis", "invalid_scatter_axes"}
        ),
        "chart_rewrites": 0,
    }

    corrupted_sorts = []
    limit_records = []
    vql_bin_tier_a = []
    for row in new_quality.values():
        constraints = _constraints(row)
        sort = constraints.get("sort") or {}
        if "limit" in str(sort.get("field") or "").lower():
            corrupted_sorts.append(row["item_id"])
        if constraints.get("limit") is not None:
            limit_records.append(row["item_id"])
        if (_grouping(row).get("grouping_origin") == "vql_bin" and row.get("quality_tier") == "A"):
            vql_bin_tier_a.append(row["item_id"])
    constraint_report = {
        "bin_regression_fixtures": [_fixture_state(item_id, new_quality, new_selected_ids) for item_id in BIN_FIXTURES],
        "limit_regression_fixtures": [_fixture_state(item_id, new_quality, new_selected_ids) for item_id in LIMIT_FIXTURES],
        "true_dimension_conflict_fixtures": [
            _fixture_state(item_id, new_quality, new_selected_ids) for item_id in TRUE_DIMENSION_CONFLICT_FIXTURES
        ],
        "time_grain_conflict_fixture": _fixture_state(TIME_CONFLICT_FIXTURE, new_quality, new_selected_ids),
        "pie_regression_fixture": _fixture_state(PIE_FIXTURE, new_quality, new_selected_ids),
        "false_ai_rejections_missing_sql_grouping": BIN_FIXTURES,
        "promoted_after_parser_fix_count": len(promoted),
        "corrected_limit_record_count": len(limit_records),
        "corrupted_sort_fields_containing_limit": sorted(corrupted_sorts),
        "tier_a_vql_bin_record_count": len(vql_bin_tier_a),
        "tier_a_time_grain_conflicts": sorted(
            row["item_id"] for row in new_quality.values()
            if row.get("quality_tier") == "A" and "time_grain_source_conflict" in (row.get("failed_rules") or [])
        ),
        "tier_a_missing_required_dimensions": sorted(
            row["item_id"] for row in new_quality.values()
            if row.get("quality_tier") == "A" and "missing_required_dimension" in (row.get("failed_rules") or [])
        ),
    }

    deterministic = _independent_rebuild()
    count_status = "PASS_CORRECTED_CORPUS_GE_1800" if len(new_selected) >= 1800 else "FAIL_CORRECTED_CORPUS_BELOW_1800"
    before_after = {
        "before": {
            "quality_pool": {
                "total": old_summary.get("total_candidates"),
                "tier_a": old_summary.get("tier_a_count"),
                "tier_b": old_summary.get("tier_b_count"),
                "tier_c": old_summary.get("tier_c_count"),
            },
            "selected_total": len(old_selected),
            "chart_distribution": dict(sorted(collections.Counter(_chart_of_flat(record) for record in old_selected).items())),
        },
        "after": {
            "quality_pool": {
                "total": new_summary.get("total_candidates"),
                "tier_a": new_summary.get("tier_a_count"),
                "tier_b": new_summary.get("tier_b_count"),
                "tier_c": new_summary.get("tier_c_count"),
            },
            "selected_total": len(new_selected),
            "split_counts": dict(sorted(split_counts.items())),
            "chart_distribution": dict(sorted(distribution.items())),
            "source_group_count": len(groups),
        },
    }
    summary = {
        "status": count_status if deterministic["passed"] else "FAIL_NONDETERMINISTIC_REBUILD",
        "rule_version": new_summary.get("rule_version"),
        "false_rejection_missing_sql_grouping_count": len(BIN_FIXTURES),
        "promoted_after_parser_fix_count": len(promoted),
        "demoted_after_scatter_or_pie_quality_fix_count": len(demoted_quality),
        "demoted_for_true_constraint_conflict_count": len(demoted_constraint),
        "demoted_for_time_grain_conflict_count": sum(
            "time_grain_source_conflict" in row["after_failed_rules"] for row in demoted_constraint
        ),
        "replacement_record_count": len(replacements),
        "removed_selected_count": len(removed_selected),
        "final_selected_total": len(new_selected),
        "final_split_counts": dict(sorted(split_counts.items())),
        "final_chart_distribution": dict(sorted(distribution.items())),
        "final_source_group_count": len(groups),
        "quality_tier_a_only": all(new_quality[item_id].get("quality_tier") == "A" for item_id in new_selected_ids),
        "minimum_quality_score": min(new_quality[item_id].get("quality_score", 0) for item_id in new_selected_ids),
        "mandatory_failure_count": sum(bool(new_quality[item_id].get("failed_rules")) for item_id in new_selected_ids),
        "exact_duplicate_count": exact_duplicates,
        "near_duplicate_count_at_or_above_0_8": len(near_duplicates),
        "evaluation_leakage_count": sum(
            1 for line in (SELECTED_DIR / "reports/leakage_report.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ),
        "cross_split_source_group_leakage_count": len(cross_split_groups),
        "deterministic_comparison": deterministic,
        "manual_spotcheck_path": "data/staging/dashboard_v3/nvbench_large_v1/reports/manual_spotcheck_template_30_v2.csv",
        "phase_3_started": False,
    }

    _write_jsonl(REPAIR_DIR / "affected_records.jsonl", affected)
    _write_jsonl(REPAIR_DIR / "promoted_after_parser_fix.jsonl", promoted)
    _write_jsonl(REPAIR_DIR / "demoted_after_quality_fix.jsonl", demoted_quality + demoted_constraint)
    _write_jsonl(REPAIR_DIR / "replacement_records.jsonl", replacements)
    _write_json(REPAIR_DIR / "scatter_validation_report.json", scatter_report)
    _write_json(REPAIR_DIR / "constraint_validation_report.json", constraint_report)
    _write_json(REPAIR_DIR / "before_after_counts.json", before_after)
    _write_json(REPAIR_DIR / "rule_change_summary.json", summary)

    lines = [
        "# nvBench Rule Repair — Final Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Rule version: `{summary['rule_version']}`",
        f"- Parser promotions: {summary['promoted_after_parser_fix_count']}",
        f"- Scatter/Pie quality demotions: {summary['demoted_after_scatter_or_pie_quality_fix_count']}",
        f"- True constraint-conflict demotions: {summary['demoted_for_true_constraint_conflict_count']}",
        f"- Replacements: {summary['replacement_record_count']}",
        f"- Selected: {summary['final_selected_total']}",
        f"- Splits: {summary['final_split_counts']}",
        f"- Charts: {summary['final_chart_distribution']}",
        f"- Source groups: {summary['final_source_group_count']}",
        f"- Exact duplicates: {summary['exact_duplicate_count']}",
        f"- Near duplicates at or above 0.8: {summary['near_duplicate_count_at_or_above_0_8']}",
        f"- Evaluation leakage: {summary['evaluation_leakage_count']}",
        f"- Cross-split group leakage: {summary['cross_split_source_group_leakage_count']}",
        f"- Deterministic rebuild: {'PASS' if deterministic['passed'] else 'FAIL'}",
        "- Phase 3 started: no",
        "",
        "VQL BIN is treated as source-backed implicit visual grouping. SQL grouping is not fabricated. "
        "Identifier Scatter axes and unresolved source conflicts are demoted rather than rewritten.",
    ]
    (REPAIR_DIR / "rule_change_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report_files = sorted(
        path for path in REPAIR_DIR.iterdir()
        if path.is_file() and path.name != "hashes.json"
    )
    _write_json(REPAIR_DIR / "hashes.json", {
        "algorithm": "sha256",
        "files": {path.name: _sha256(path) for path in report_files},
    })
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["status"] == "PASS_CORRECTED_CORPUS_GE_1800" else 1)


if __name__ == "__main__":
    if "--correct-incomplete-baseline-reports" in sys.argv[1:]:
        _correct_incomplete_baseline_reports()
    else:
        main()
