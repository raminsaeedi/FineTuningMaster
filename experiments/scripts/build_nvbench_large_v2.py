"""Rebuild and independently verify nvbench quality-pool v2 and large v2.

This orchestrator never writes to ``nvbench_large_v1``. It snapshots every v1
file, runs canonical and isolated-scratch builds, compares deterministic
artifacts, verifies the full v1 tree byte-for-byte, finalizes v2 hashes/status,
and removes only its validated scratch directory.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.frozen_validation import sha256_of_file  # noqa: E402
from src.data_pipeline.nvbench_large_v2 import snapshot_tree  # noqa: E402
from src.utils.io import write_json  # noqa: E402

QUALITY_OUT = _PROJECT_ROOT / "data/staging/dashboard_v3/nvbench_quality_pool_final_v2"
LARGE_OUT = _PROJECT_ROOT / "data/staging/dashboard_v3/nvbench_large_v2"
V1_DIR = _PROJECT_ROOT / "data/staging/dashboard_v3/nvbench_large_v1"
PRE_VERIFICATION = LARGE_OUT / "reports/pre_repair_verification.md"
FINAL_STATUS = "PASS_LARGE_V2_READY_FOR_HUMAN_R1"
FAIL_STATUS = "FAIL_RULE_REPAIR_OR_VALIDATION"

QUALITY_DETERMINISTIC = (
    "tier_a_candidates.jsonl",
    "tier_b_diagnostics.jsonl",
    "tier_c_rejected.jsonl",
    "quality_pool_summary.json",
    "quality_pool_summary.md",
    "quality_rule_failures.csv",
)
LARGE_DETERMINISTIC = (
    "all_selected.jsonl",
    "train.jsonl",
    "val.jsonl",
    "test.jsonl",
    "distribution_report.json",
    "reports/distribution_report.csv",
    "reports/duplicate_report.jsonl",
    "reports/leakage_report.jsonl",
    "reports/replacement_records.jsonl",
    "reports/rule_change_summary.md",
    "reports/manual_spotcheck_template_30_r1.csv",
    "reports/manual_spotcheck_protocol_r1.md",
)
REQUIRED_OUTPUTS = (
    "all_selected.jsonl",
    "train.jsonl",
    "val.jsonl",
    "test.jsonl",
    "distribution_report.json",
    "independent_evaluation_reference.json",
    "quality_pool_reference.json",
    "reports/pre_repair_verification.md",
    "reports/validation_report.json",
    "reports/validation_report.md",
    "reports/distribution_report.csv",
    "reports/duplicate_report.jsonl",
    "reports/leakage_report.jsonl",
    "reports/selection_attrition.json",
    "reports/selection_attrition.md",
    "reports/replacement_records.jsonl",
    "reports/rule_change_summary.json",
    "reports/rule_change_summary.md",
    "reports/manual_spotcheck_template_30_r1.csv",
    "reports/manual_spotcheck_protocol_r1.md",
    "reports/multi_record_source_groups.csv",
    "reports/multi_record_source_groups.md",
    "reports/human_eval_test_items_40.jsonl",
    "reports/human_eval_test_items_40.csv",
    "reports/quality_report.json",
    "reports/quality_report.md",
    "reports/warnings.jsonl",
)


def _run(command: list[str]) -> None:
    print(f"[run] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=_PROJECT_ROOT, check=True)


def _tree_digest(snapshot: dict) -> str:
    payload = "".join(
        f"{path}\0{meta['size']}\0{meta['sha256']}\n"
        for path, meta in sorted(snapshot.items())
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compare_files(canonical: Path, scratch: Path, relative_paths: tuple[str, ...]) -> dict:
    comparisons = {}
    for relative in relative_paths:
        canonical_path = canonical / relative
        scratch_path = scratch / relative
        comparisons[relative] = {
            "canonical_exists": canonical_path.is_file(),
            "scratch_exists": scratch_path.is_file(),
            "canonical_sha256": sha256_of_file(canonical_path) if canonical_path.is_file() else None,
            "scratch_sha256": sha256_of_file(scratch_path) if scratch_path.is_file() else None,
        }
        comparisons[relative]["match"] = (
            comparisons[relative]["canonical_exists"]
            and comparisons[relative]["scratch_exists"]
            and comparisons[relative]["canonical_sha256"] == comparisons[relative]["scratch_sha256"]
        )
    return comparisons


def _build_quality(out_dir: Path, profile_cache: Path) -> None:
    _run([
        sys.executable,
        "experiments/scripts/rebuild_nvbench_quality_pool_final.py",
        "--out", str(out_dir),
        "--profile-cache", str(profile_cache),
    ])


def _build_large(out_dir: Path, quality_dir: Path) -> None:
    _run([
        sys.executable,
        "experiments/scripts/run_nvbench_large_v1.py",
        "--dataset-version", "v2",
        "--out", str(out_dir),
        "--quality-pool-dir", str(quality_dir),
        "--previous-quality-pool-dir",
        str(_PROJECT_ROOT / "data/staging/dashboard_v3/nvbench_quality_pool_final"),
        "--baseline-selected", str(V1_DIR / "all_selected.jsonl"),
        "--preferred-target", "1819",
        "--minimum-acceptable", "1800",
        "--seed", "42",
        "--max-per-group", "2",
        "--val-fraction", "0.15",
        "--test-fraction", "0.15",
    ])


def _finalize(
    quality_comparison: dict,
    large_comparison: dict,
    v1_before: dict,
    v1_after: dict,
) -> None:
    deterministic = all(
        result["match"] for result in [*quality_comparison.values(), *large_comparison.values()]
    )
    v1_unchanged = v1_before == v1_after
    completed_absent = not (LARGE_OUT / "reports/manual_spotcheck_completed_r1.csv").exists()
    required_present = all((LARGE_OUT / relative).is_file() for relative in REQUIRED_OUTPUTS)
    post_build_passed = deterministic and v1_unchanged and completed_absent and required_present

    validation_path = LARGE_OUT / "reports/validation_report.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["post_build_gates"] = {
        "independent_scratch_rebuild_deterministic": deterministic,
        "quality_artifact_comparison": quality_comparison,
        "large_artifact_comparison": large_comparison,
        "v1_full_tree_unchanged": v1_unchanged,
        "v1_file_count": len(v1_after),
        "v1_tree_digest_before": _tree_digest(v1_before),
        "v1_tree_digest_after": _tree_digest(v1_after),
        "required_v2_outputs_present": required_present,
        "completed_human_r1_file_absent": completed_absent,
    }
    validation["passed"] = bool(validation.get("passed")) and post_build_passed
    validation["status"] = FINAL_STATUS if validation["passed"] else FAIL_STATUS
    write_json(validation, validation_path)

    validation_md = LARGE_OUT / "reports/validation_report.md"
    with validation_md.open("a", encoding="utf-8") as stream:
        stream.write(
            "\n## Post-build reproducibility gates\n"
            f"- [{'PASS' if deterministic else 'FAIL'}] independent scratch rebuild deterministic\n"
            f"- [{'PASS' if v1_unchanged else 'FAIL'}] full historical v1 tree unchanged "
            f"({len(v1_after)} files; digest `{_tree_digest(v1_after)}`)\n"
            f"- [{'PASS' if required_present else 'FAIL'}] all required v2 outputs present\n"
            f"- [{'PASS' if completed_absent else 'FAIL'}] completed R1 file not fabricated\n"
        )

    output_hashes = {
        relative: sha256_of_file(LARGE_OUT / relative)
        for relative in sorted(REQUIRED_OUTPUTS)
    }
    manifest_path = LARGE_OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "passed": validation["passed"],
        "status": validation["status"],
        "human_gate_status": "WAITING_FOR_HUMAN_R1" if validation["passed"] else "NOT_READY",
        "deterministic_scratch_rebuild_passed": deterministic,
        "v1_full_tree_unchanged": v1_unchanged,
        "v1_file_count": len(v1_after),
        "v1_tree_digest": _tree_digest(v1_after),
        "required_v2_outputs_present": required_present,
        "output_hashes": output_hashes,
    })
    write_json(manifest, manifest_path)
    hashes = {
        "algorithm": "sha256",
        "files": output_hashes,
        "manifest": sha256_of_file(manifest_path),
    }
    write_json(hashes, LARGE_OUT / "hashes.json")
    if not validation["passed"]:
        raise RuntimeError(f"{FAIL_STATUS}: {validation['post_build_gates']}")


def main() -> None:
    if not PRE_VERIFICATION.is_file():
        raise SystemExit(f"required independent verification report missing: {PRE_VERIFICATION}")
    if not V1_DIR.is_dir():
        raise SystemExit(f"historical v1 directory missing: {V1_DIR}")
    v1_before = snapshot_tree(V1_DIR)

    scratch_parent = _PROJECT_ROOT / "data/staging/dashboard_v3/.scratch"
    scratch_parent.mkdir(parents=True, exist_ok=True)
    scratch_root = Path(tempfile.mkdtemp(prefix="nvbench_large_v2_", dir=scratch_parent)).resolve()
    expected_parent = scratch_parent.resolve()
    if expected_parent not in scratch_root.parents:
        raise SystemExit(f"unsafe scratch path: {scratch_root}")

    try:
        canonical_profile = QUALITY_OUT / "field_profiles_v2.json"
        _build_quality(QUALITY_OUT, canonical_profile)
        _build_large(LARGE_OUT, QUALITY_OUT)

        scratch_quality = scratch_root / "quality_pool"
        scratch_large = scratch_root / "large"
        scratch_verification = scratch_large / "reports/pre_repair_verification.md"
        scratch_verification.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PRE_VERIFICATION, scratch_verification)
        _build_quality(scratch_quality, scratch_root / "field_profiles_scratch.json")
        _build_large(scratch_large, scratch_quality)

        quality_comparison = _compare_files(QUALITY_OUT, scratch_quality, QUALITY_DETERMINISTIC)
        large_comparison = _compare_files(LARGE_OUT, scratch_large, LARGE_DETERMINISTIC)
        v1_after = snapshot_tree(V1_DIR)
        _finalize(quality_comparison, large_comparison, v1_before, v1_after)
    finally:
        # Exact generated directory, already verified beneath the dedicated
        # scratch parent. Historical/canonical paths are never deletion targets.
        if scratch_root.is_dir() and expected_parent in scratch_root.parents:
            shutil.rmtree(scratch_root)

    print(json.dumps({
        "status": FINAL_STATUS,
        "human_gate_status": "WAITING_FOR_HUMAN_R1",
        "quality_pool": str(QUALITY_OUT),
        "large_dataset": str(LARGE_OUT),
        "v1_tree_digest": _tree_digest(snapshot_tree(V1_DIR)),
    }, indent=2))


if __name__ == "__main__":
    main()
