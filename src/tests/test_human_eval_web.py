"""Tests for exporting a study as a static site and importing web ratings."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.evaluation.human.pipeline import build_study, run_analysis, validate_ratings, load_study
from src.evaluation.human.rubric import RUBRIC_KEYS
from src.evaluation.human.web import (
    WebExportError,
    build_web_bundle,
    import_web_ratings,
)

METHODS = ("A", "B", "C", "D")
N_ITEMS = 4
N_RATERS = 3
RATINGS_PER_OUTPUT = 2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_project(tmp_path: Path, *, hostile_text: bool = False) -> dict:
    """Create a minimal project tree with four comparable runs."""
    dataset, model, seed = "dashboard_v4", "qwen3_8b", 42
    item_ids = [f"item_{index:02d}" for index in range(N_ITEMS)]

    frozen = tmp_path / "data" / "frozen" / dataset
    frozen.mkdir(parents=True)
    test_path = frozen / "test.jsonl"
    with test_path.open("w", encoding="utf-8") as handle:
        for item_id in item_ids:
            handle.write(json.dumps({
                "item_id": item_id,
                "brief": {
                    "users": "Analysts <in> logistics",
                    "goals": ["reduce lead time"],
                    "kpis": ["Order Volume"],
                    "columns": [{"name": "date", "dtype": "datetime", "role": "dimension"}],
                    "constraints": "WCAG AA",
                    "extra": {"reference": "must not be published"},
                },
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
            raw = f"recommendation for {item_id}"
            parsed = {
                "context_summary": {"goal": "reduce lead time"},
                "kpi_chart_mapping": [
                    {"kpi": "Order Volume", "task_type": "trend", "chart_type": "line", "alternatives": ["area"]}
                ],
                "layout": {"grid": "2x2"},
                "rationales": [{"claim": "trend over time", "principle": "line encodes change"}],
            }
            if hostile_text and method == "A":
                parsed = None
                raw = "<script>alert('x')</script> & <b>bold</b>"
            rows.append({"item_id": item_id, "method_name": method, "raw_text": raw, "parsed": parsed})
        (run_dir / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (run_dir / "manifest.json").write_text(json.dumps({
            "run_id": f"{dataset}_{model}_{method}_{seed}",
            "config_hash": f"config-{method}",
            "dataset_version": dataset,
            "dataset_hashes": hashes,
            "model_key": model,
            "method_key": method,
            "seed": seed,
        }), encoding="utf-8")
        (run_dir / "dataset_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")
        (run_dir / "config_snapshot.yaml").write_text(
            "model:\n  key: %s\nmethod:\n  name: %s\ndata:\n  dataset_version: %s\n  test_file: %s\nseed: %s\n"
            % (model, method, dataset, str(test_path), seed),
            encoding="utf-8",
        )
        if method in ("B", "D"):
            (run_dir / "kb_hashes.json").write_text(json.dumps({"chunks_sha256": "kb-1"}), encoding="utf-8")

    result = build_study(
        project_root=tmp_path,
        dataset=dataset,
        model=model,
        seed=seed,
        outputs_root=outputs_root,
        n_items=N_ITEMS,
        n_raters=N_RATERS,
        ratings_per_output=RATINGS_PER_OUTPUT,
        item_list=item_list,
    )
    return {"project_root": tmp_path, "study_dir": Path(result["study_dir"]), "item_ids": item_ids}


def _export(project: dict, **kwargs) -> dict:
    return build_web_bundle(
        study_dir=project["study_dir"],
        project_root=project["project_root"],
        base_url=kwargs.pop("base_url", "https://example.org/eval"),
        salt=kwargs.pop("salt", "test-salt"),
        **kwargs,
    )


def test_export_publishes_only_blinded_content(tmp_path):
    project = _make_project(tmp_path)
    bundle = _export(
        project,
        contact_email="researcher@example.org",
        institution="Example University",
        retention_months=12,
        ethics_status="approved for use in this thesis",
    )
    site = Path(bundle["site_dir"])
    for name in ("index.html", "app.js", "styles.css", "study-data.js", "config.js"):
        assert (site / name).exists(), name

    published = (site / "study-data.js").read_text(encoding="utf-8")
    for method in METHODS:
        assert f'"method": "{method}"' not in published
    for needle in ("prompt_only", "QLoRA", "method_name", "must not be published"):
        assert needle not in published
    # Chapter 6.8 also hides model identity and seed from raters.
    for needle in ("qwen3_8b", "seed_42", '"seed"'):
        assert needle not in published
    assert json.loads(published[len("window.HEVAL_STUDY = "):].rstrip().rstrip(";"))["study_id"].startswith("study_")
    for item_id in project["item_ids"]:
        assert item_id not in published

    payload = json.loads(published[len("window.HEVAL_STUDY = "):].rstrip().rstrip(";"))
    assert len(payload["units"]) == N_ITEMS * len(METHODS)
    assert len(payload["items"]) == N_ITEMS
    assert sorted(payload["raters"]) == ["rater_01", "rater_02", "rater_03"]
    loads = [len(rater["units"]) for rater in payload["raters"].values()]
    total_assignments = N_ITEMS * len(METHODS) * RATINGS_PER_OUTPUT
    assert sum(loads) == total_assignments
    assert max(loads) - min(loads) <= 1
    for rater in payload["raters"].values():
        assert len(set(rater["units"])) == len(rater["units"])

    key = json.loads(Path(bundle["key_path"]).read_text(encoding="utf-8"))
    assert set(key["units"]) == set(payload["units"])
    assert {entry["method"] for entry in key["units"].values()} == set(METHODS)
    links = Path(bundle["links_path"]).read_text(encoding="utf-8")
    assert "https://example.org/eval/?r=rater_01&t=" in links
    config = (site / "config.js").read_text(encoding="utf-8")
    assert "researcher@example.org" in config
    assert "Example University" in config
    assert '"retention_months": 12' in config
    index = (site / "index.html").read_text(encoding="utf-8")
    app = (site / "app.js").read_text(encoding="utf-8")
    assert 'id="consent"' in index
    assert 'id="intro-time"' in index
    assert 'id="task-meta"' in index
    assert "Save rating and continue" in index
    assert '"time_budget"' in published
    assert "consent_at" in app
    assert "rater-picker" not in index


def test_export_is_deterministic_for_a_fixed_salt(tmp_path):
    project = _make_project(tmp_path)
    first = _export(project)
    first_data = (Path(first["site_dir"]) / "study-data.js").read_text(encoding="utf-8")
    second = _export(project)
    second_data = (Path(second["site_dir"]) / "study-data.js").read_text(encoding="utf-8")
    assert first["export_id"] == second["export_id"]
    assert first_data == second_data


def test_presentation_order_avoids_consecutive_outputs_for_one_brief(tmp_path):
    project = _make_project(tmp_path)
    bundle = _export(project)
    payload = json.loads(
        (Path(bundle["site_dir"]) / "study-data.js").read_text(encoding="utf-8")
        [len("window.HEVAL_STUDY = "):].rstrip().rstrip(";")
    )
    for rater in payload["raters"].values():
        items = [payload["units"][token]["item"] for token in rater["units"]]
        assert all(left != right for left, right in zip(items, items[1:]))


def test_model_text_is_escaped_before_publication(tmp_path):
    project = _make_project(tmp_path, hostile_text=True)
    bundle = _export(project)
    published = (Path(bundle["site_dir"]) / "study-data.js").read_text(encoding="utf-8")
    assert "<script>" not in published
    assert "&lt;script&gt;" in published


def _rating_rows(bundle: dict, rater: str, *, score: int = 4, limit: int | None = None) -> list[dict]:
    payload = json.loads(
        (Path(bundle["site_dir"]) / "study-data.js").read_text(encoding="utf-8")
        [len("window.HEVAL_STUDY = "):].rstrip().rstrip(";")
    )
    tokens = payload["raters"][rater]["units"]
    if limit is not None:
        tokens = tokens[:limit]
    rows = []
    for index, token in enumerate(tokens):
        rows.append({
            "received_at": f"2026-01-01T10:{index:02d}:00Z",
            "study_id": payload["study_id"],
            "export_id": payload["export_id"],
            "rater_id": rater,
            "rating_id": f"{rater}-{index}",
            "unit_token": token,
            "position": index + 1,
            "comment": "",
            "scores_json": json.dumps({dimension: score for dimension in RUBRIC_KEYS}),
            "client_timestamp": f"2026-01-01T10:{index:02d}:00Z",
            "duration_ms": 12000,
            "app_version": "human-eval-web-v1",
        })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> Path:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_import_round_trip_produces_analysable_ratings(tmp_path):
    project = _make_project(tmp_path)
    bundle = _export(project)
    rows = []
    for rater in ("rater_01", "rater_02", "rater_03"):
        rows.extend(_rating_rows(bundle, rater))
    csv_path = _write_csv(tmp_path / "sheet.csv", rows)

    report = import_web_ratings(study_dir=project["study_dir"], inputs=[csv_path])
    assert report["accepted"] == len(rows)
    assert report["rejected"] == 0
    assert report["completion_percentage"] == 100.0

    manifest, _items, assignment = load_study(project["study_dir"])
    stored = []
    for path in sorted((project["study_dir"] / "ratings").glob("*.jsonl")):
        stored.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    validation = validate_ratings(manifest, assignment, stored)
    assert validation["completion"]["complete"] is True

    analysis = run_analysis(
        study_dir=project["study_dir"],
        project_root=project["project_root"],
        bootstrap_resamples=50,
    )
    assert analysis["completion"]["received_ratings"] == len(rows)


def test_import_deduplicates_resends_and_accepts_json_backups(tmp_path):
    project = _make_project(tmp_path)
    bundle = _export(project)
    rows = _rating_rows(bundle, "rater_01", score=3)
    duplicated = rows + [dict(row, rating_id=row["rating_id"] + "-resend") for row in rows]
    csv_path = _write_csv(tmp_path / "sheet.csv", duplicated)

    backup_rows = _rating_rows(bundle, "rater_02", score=5)
    backup = {
        "study_id": "ignored",
        "rater_id": "rater_02",
        "ratings": [
            {
                "rating_id": row["rating_id"],
                "unit_token": row["unit_token"],
                "scores": json.loads(row["scores_json"]),
                "comment": "from backup",
                "client_timestamp": row["client_timestamp"],
            }
            for row in backup_rows
        ],
    }
    backup_path = tmp_path / "ratings_rater_02.json"
    backup_path.write_text(json.dumps(backup), encoding="utf-8")

    report = import_web_ratings(study_dir=project["study_dir"], inputs=[csv_path, backup_path])
    assert report["duplicate_submissions"] == len(rows)
    written = {
        path.stem: len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
        for path in sorted((project["study_dir"] / "ratings").glob("*.jsonl"))
    }
    assert written == {"rater_01": len(rows), "rater_02": len(backup_rows)}


def test_import_rejects_foreign_tokens_and_unassigned_units(tmp_path):
    project = _make_project(tmp_path)
    bundle = _export(project)
    payload = json.loads(
        (Path(bundle["site_dir"]) / "study-data.js").read_text(encoding="utf-8")
        [len("window.HEVAL_STUDY = "):].rstrip().rstrip(";")
    )
    foreign = set(payload["raters"]["rater_01"]["units"]) - set(payload["raters"]["rater_03"]["units"])
    rows = _rating_rows(bundle, "rater_01", limit=2)
    rows[0]["unit_token"] = "u_notarealtoken"
    rows[1]["unit_token"] = sorted(foreign)[0]
    rows[1]["rater_id"] = "rater_03"  # a unit that belongs to rater_01 only
    invalid_scores = _rating_rows(bundle, "rater_02", limit=1)
    invalid_scores[0]["scores_json"] = json.dumps({dimension: 9 for dimension in RUBRIC_KEYS})
    csv_path = _write_csv(tmp_path / "sheet.csv", rows + invalid_scores)

    report = import_web_ratings(study_dir=project["study_dir"], inputs=[csv_path], dry_run=True)
    assert report["accepted"] == 0
    assert report["rejected"] == 3
    problems = " ".join(entry["problem"] for entry in report["rejected_rows"])
    assert "does not belong to this export" in problems
    assert "was not assigned" in problems
    assert "outside 1-5" in problems
    assert not list((project["study_dir"] / "ratings").glob("*.jsonl"))


def test_import_requires_the_unblinding_key(tmp_path):
    project = _make_project(tmp_path)
    bundle = _export(project)
    Path(bundle["key_path"]).unlink()
    with pytest.raises(WebExportError, match="Unblinding key not found"):
        import_web_ratings(study_dir=project["study_dir"], inputs=[tmp_path / "sheet.csv"])
