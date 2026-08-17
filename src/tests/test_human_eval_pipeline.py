"""Focused tests for the final Professor human-evaluation workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.evaluation.human.assignment import build_assignment
from src.evaluation.human.pipeline import (
    HumanEvaluationError,
    IncompleteStudyError,
    build_study,
    load_study,
    run_analysis,
    validate_ratings,
)
from src.evaluation.human.storage import append_rating, load_done_units


METHODS = ("A", "B", "C", "D")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_fixture(
    tmp_path: Path,
    *,
    n_items: int = 40,
    wrong: dict[str, object] | None = None,
    dataset: str = "dashboard_v4",
):
    model = "qwen3_8b"
    seed = 42
    frozen = tmp_path / "data" / "frozen" / dataset
    frozen.mkdir(parents=True)
    item_ids = [f"item_{i:03d}" for i in range(n_items)]
    test_path = frozen / "test.jsonl"
    with test_path.open("w", encoding="utf-8") as handle:
        for item_id in item_ids:
            handle.write(json.dumps({
                "item_id": item_id,
                "brief": {
                    "item_id": item_id,
                    "users": "Analyst",
                    "goals": [f"Goal {item_id}"],
                    "kpis": ["COUNT(value)"],
                    "columns": [{"name": "value", "dtype": "number"}],
                    "constraints": None,
                    "extra": {"reference": "must not reach rater"},
                },
                "recommendation": {"gold": "must not reach rater"},
            }) + "\n")
    hashes = {"test": _sha256(test_path)}
    (frozen / "hashes.json").write_text(json.dumps(hashes), encoding="utf-8")
    item_list = frozen / "human_eval_test_items_40.csv"
    with item_list.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id"])
        writer.writeheader()
        writer.writerows({"item_id": item_id} for item_id in item_ids)

    outputs_root = tmp_path / "experiments" / "outputs" / "final"
    for method in METHODS:
        run_dir = outputs_root / dataset / model / method / f"seed_{seed}"
        run_dir.mkdir(parents=True)
        rows = []
        for item_id in item_ids:
            rows.append({
                "item_id": item_id,
                "method_name": method,
                "raw_text": f"recommendation {method} {item_id}",
                "parsed": {"layout": {"method": method}},
            })
        (run_dir / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        metadata_method = method
        metadata_model = model
        metadata_dataset = dataset
        metadata_seed = seed
        if wrong and method in wrong:
            key, value = next(iter(wrong.items())) if len(wrong) == 1 else (None, None)
            # Tests pass a method-specific mapping such as {"B": ("model", "qwen3_14b")}.
            if isinstance(value, tuple):
                key, value = value
            if key == "model":
                metadata_model = str(value)
            elif key == "seed":
                metadata_seed = int(value)
            elif key == "dataset":
                metadata_dataset = str(value)
            elif key == "method":
                metadata_method = str(value)
        (run_dir / "manifest.json").write_text(json.dumps({
            "run_id": f"{dataset}_{model}_{metadata_method}_{seed}",
            "config_hash": f"config-{method}",
            "dataset_version": metadata_dataset,
            "dataset_hashes": hashes,
            "model_key": metadata_model,
            "model_hf_id": metadata_model,
            "method_key": metadata_method,
            "seed": metadata_seed,
        }), encoding="utf-8")
        (run_dir / "dataset_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")
        (run_dir / "config_snapshot.yaml").write_text(
            "model:\n  key: %s\nmethod:\n  name: %s\ndata:\n  dataset_version: %s\n  test_file: %s\nseed: %s\n"
            % (metadata_model, metadata_method, metadata_dataset, str(test_path), metadata_seed),
            encoding="utf-8",
        )
        (run_dir / "config_hash.txt").write_text(f"config-{method}\n", encoding="utf-8")
        if method in ("B", "D"):
            (run_dir / "kb_hashes.json").write_text(json.dumps({"chunks_sha256": "kb-1"}), encoding="utf-8")
    return tmp_path, outputs_root, item_list, dataset, model, seed, item_ids


def test_professor_layout_builds_fixed_v4_study_and_balanced_assignment(tmp_path):
    project_root, outputs_root, item_list, dataset, model, seed, item_ids = _make_fixture(tmp_path)
    result = build_study(
        project_root=project_root,
        dataset=dataset,
        model=model,
        seed=seed,
        outputs_root=outputs_root,
        n_items=40,
        n_raters=6,
        ratings_per_output=3,
        assignment_seed=17,
        item_list=item_list,
    )
    manifest = result["manifest"]
    assert manifest["study_type"] == "final"
    assert manifest["n_items"] == 40
    assert manifest["total_expected_outputs"] == 160
    assert manifest["total_expected_ratings"] == 480
    assert manifest["item_ids"] == item_ids
    assert set(manifest["methods"]) == set(METHODS)
    assert set(result["assignment"]["load"].values()) == {80}
    first = json.loads((result["study_dir"] / "items.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "extra" not in first["brief"]
    assert "recommendation" not in first


@pytest.mark.parametrize("wrong", [
    {"B": ("model", "qwen3_14b")},
    {"B": ("seed", 43)},
    {"B": ("dataset", "dashboard_v3")},
    {"B": ("method", "A")},
])
def test_compatibility_rejects_mixed_professor_runs(tmp_path, wrong):
    project_root, outputs_root, item_list, dataset, model, seed, _ = _make_fixture(tmp_path, wrong=wrong)
    with pytest.raises(HumanEvaluationError, match="compatibility validation failed"):
        build_study(
            project_root=project_root,
            dataset=dataset,
            model=model,
            seed=seed,
            outputs_root=outputs_root,
            item_list=item_list,
        )


def test_missing_prediction_is_hard_failure(tmp_path):
    project_root, outputs_root, item_list, dataset, model, seed, _ = _make_fixture(tmp_path)
    (outputs_root / dataset / model / "D" / f"seed_{seed}" / "predictions.jsonl").unlink()
    with pytest.raises(HumanEvaluationError, match="Missing predictions"):
        build_study(
            project_root=project_root,
            dataset=dataset,
            model=model,
            seed=seed,
            outputs_root=outputs_root,
            item_list=item_list,
        )


def _populate_ratings(study_dir: Path, assignment: dict, *, only_first: bool = False):
    count = 0
    for rater_id, tasks in assignment["raters"].items():
        for task in tasks:
            if only_first and count:
                return
            append_rating(
                study_dir / "ratings",
                rater_id=rater_id,
                unit_id=task["unit_id"],
                item_id=task["item_id"],
                method=task["method"],
                scores={
                    "chart_appropriateness": 5,
                    "layout_quality": 4,
                    "styling_accessibility": 3,
                    "interaction_design": 2,
                    "rationale_quality": 4,
                    "overall_usefulness": 5,
                },
            )
            count += 1


def test_resume_and_rating_validation_detect_duplicate_and_incomplete(tmp_path):
    project_root, outputs_root, item_list, dataset, model, seed, _ = _make_fixture(tmp_path)
    result = build_study(
        project_root=project_root,
        dataset=dataset,
        model=model,
        seed=seed,
        outputs_root=outputs_root,
        item_list=item_list,
    )
    _populate_ratings(result["study_dir"], result["assignment"], only_first=True)
    rater_id = next(iter(result["assignment"]["raters"]))
    first_task = result["assignment"]["raters"][rater_id][0]
    assert first_task["unit_id"] in load_done_units(result["study_dir"] / "ratings", rater_id)
    rows = []
    for path in sorted((result["study_dir"] / "ratings").glob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    rows.append(dict(rows[0]))
    manifest, _, assignment = load_study(result["study_dir"])
    validation = validate_ratings(manifest, assignment, rows)
    assert validation["completion"]["duplicate_ratings"]
    assert not validation["completion"]["complete"]
    with pytest.raises(IncompleteStudyError):
        run_analysis(study_dir=result["study_dir"], project_root=project_root, bootstrap_resamples=20)


def test_full_mocked_analysis_writes_all_artifacts_and_outcomes(tmp_path):
    project_root, outputs_root, item_list, dataset, model, seed, _ = _make_fixture(tmp_path)
    result = build_study(
        project_root=project_root,
        dataset=dataset,
        model=model,
        seed=seed,
        outputs_root=outputs_root,
        item_list=item_list,
    )
    _populate_ratings(result["study_dir"], result["assignment"])
    analyzed = run_analysis(
        study_dir=result["study_dir"],
        project_root=project_root,
        bootstrap_resamples=50,
    )
    analysis_dir = result["study_dir"] / "analysis"
    required = {
        "rating_completion.json",
        "irr_alphas.json",
        "irr_alphas.csv",
        "system_means.json",
        "system_means.csv",
        "per_item_scores.csv",
        "human_stats.json",
        "human_chart_acceptability.json",
        "human_eval_summary.md",
    }
    assert required <= {path.name for path in analysis_dir.iterdir()}
    assert analyzed["completion"]["received_ratings"] == 480
    assert set(analyzed["human_stats"]["outcomes"]) == {
        "chart_appropriateness", "layout_quality", "styling_accessibility",
        "interaction_design", "rationale_quality", "overall_usefulness", "composite_score",
    }
    assert analyzed["human_chart_acceptability"]["label"] == "human chart acceptability"
    assert analyzed["human_chart_acceptability"]["n_items"] == 40


def test_cli_end_to_end_with_professor_layout_mock_predictions(tmp_path):
    project_root, outputs_root, item_list, dataset, model, seed, _ = _make_fixture(
        tmp_path, dataset="mock_v4"
    )
    repository = Path(__file__).resolve().parents[2]
    study_dir = tmp_path / "study"
    build_cmd = [
        sys.executable,
        str(repository / "experiments" / "scripts" / "build_human_eval.py"),
        "--dataset", dataset,
        "--model", model,
        "--seed", str(seed),
        "--outputs-root", str(outputs_root),
        "--item-list", str(item_list),
        "--test-file", str(project_root / "data" / "frozen" / dataset / "test.jsonl"),
        "--out-dir", str(study_dir),
    ]
    built = subprocess.run(build_cmd, cwd=repository, capture_output=True, text=True)
    assert built.returncode == 0, built.stderr + built.stdout
    manifest, _, assignment = load_study(study_dir)
    assert manifest["total_expected_ratings"] == 480
    _populate_ratings(study_dir, assignment)
    analysis_cmd = [
        sys.executable,
        str(repository / "experiments" / "scripts" / "compute_irr.py"),
        "--study-dir", str(study_dir),
        "--bootstrap-resamples", "50",
    ]
    analyzed = subprocess.run(analysis_cmd, cwd=repository, capture_output=True, text=True)
    assert analyzed.returncode == 0, analyzed.stderr + analyzed.stdout
    assert (study_dir / "analysis" / "human_stats.json").exists()
