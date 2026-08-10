"""Tests for the deterministic dashboard_v3 robustness perturbation builder.

The frozen package under ``data/frozen/dashboard_v3`` is immutable: the builder
may only read it, and every artifact must land in ``data/eval/robustness_v3``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.scripts.build_perturbations_v3 import (
    DEFAULT_OUT_DIR,
    DEFAULT_SEED,
    DEFAULT_SOURCE,
    MANIFEST_NAME,
    VARIANTS,
    _n_briefs_changed,
    _perturb_rows,
    build_perturbation_sets,
)
from src.data_pipeline.frozen_validation import sha256_of_file
from src.data_pipeline.perturbations import paraphrase_brief
from src.utils.io import read_jsonl, write_jsonl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FROZEN_TEST = _PROJECT_ROOT / DEFAULT_SOURCE
_BUILT_DIR = _PROJECT_ROOT / DEFAULT_OUT_DIR


def _row(index: int) -> dict:
    return {
        "item_id": f"nvbench:{1000 + index}:query:0",
        "split": "test",
        "brief": {
            "item_id": f"nvbench:{1000 + index}:query:0",
            "users": "Analyst who wants to track key metrics",
            "goals": [f"Show the revenue across region {index}"],
            "kpis": [f"SUM(revenue_{index})", f"COUNT(order_{index})"],
            "columns": [
                {"name": "region", "dtype": "categorical", "role": "dimension"},
                {"name": f"revenue_{index}", "dtype": "number", "role": "measure"},
            ],
            "constraints": "Compare the last two quarters",
            "extra": {"source": "nvbench", "provenance": {"source_group_id": f"nvbench:{1000 + index}"}},
        },
        "recommendation": {"layout": {"type": "single"}, "kpi_chart_mapping": [{"kpi": f"SUM(revenue_{index})"}]},
    }


@pytest.fixture()
def source_file(tmp_path: Path) -> Path:
    path = tmp_path / "source" / "test.jsonl"
    # Deliberately unsorted input: output order must not depend on read order.
    write_jsonl([_row(i) for i in (3, 1, 2, 0)], path)
    return path


# ------------------------------------------------------------- row transform


def test_perturb_rows_preserves_item_id_linkage():
    rows = [_row(i) for i in range(3)]
    out = _perturb_rows(rows, paraphrase_brief, "paraphrased")
    assert [r["item_id"] for r in out] == sorted(r["item_id"] for r in rows)
    for row in out:
        assert row["brief"]["item_id"] == row["item_id"]
        assert row["brief"]["extra"]["base_item_id"] == row["item_id"]
        assert row["brief"]["extra"]["variant"] == "paraphrased"


def test_perturb_rows_keeps_gold_recommendation_and_provenance():
    rows = [_row(0)]
    out = _perturb_rows(rows, paraphrase_brief, "paraphrased")
    assert out[0]["recommendation"] == rows[0]["recommendation"]
    assert out[0]["split"] == "test"
    assert out[0]["brief"]["extra"]["provenance"] == rows[0]["brief"]["extra"]["provenance"]
    assert rows[0]["brief"]["users"] == "Analyst who wants to track key metrics"  # source untouched


def test_perturb_rows_sorted_output_is_read_order_independent():
    rows = [_row(i) for i in range(4)]
    forward = _perturb_rows(rows, paraphrase_brief, "paraphrased")
    reverse = _perturb_rows(list(reversed(rows)), paraphrase_brief, "paraphrased")
    assert forward == reverse


def test_n_briefs_changed_counts_only_altered_briefs():
    rows = [_row(0)]
    assert _n_briefs_changed(rows, paraphrase_brief) == 1
    assert _n_briefs_changed(rows, lambda b: b) == 0


# ------------------------------------------------------------------ builder


def test_build_writes_both_variants_with_full_record_count(source_file: Path, tmp_path: Path):
    out_dir = tmp_path / "robustness_v3"
    manifest = build_perturbation_sets(source_file, out_dir)
    for fname, _transform, variant in VARIANTS:
        records = read_jsonl(out_dir / fname)
        assert len(records) == 4
        assert manifest["outputs"][fname]["n_records"] == 4
        assert {r["brief"]["extra"]["variant"] for r in records} == {variant}


def test_build_preserves_item_id_pairing_with_the_source(source_file: Path, tmp_path: Path):
    out_dir = tmp_path / "robustness_v3"
    build_perturbation_sets(source_file, out_dir)
    source_ids = {r["item_id"] for r in read_jsonl(source_file)}
    for fname, _transform, _variant in VARIANTS:
        assert {r["item_id"] for r in read_jsonl(out_dir / fname)} == source_ids


def test_build_is_deterministic_for_the_same_seed(source_file: Path, tmp_path: Path):
    first, second = tmp_path / "run_a", tmp_path / "run_b"
    m1 = build_perturbation_sets(source_file, first, seed=DEFAULT_SEED)
    m2 = build_perturbation_sets(source_file, second, seed=DEFAULT_SEED)
    for fname, _transform, _variant in VARIANTS:
        assert (first / fname).read_bytes() == (second / fname).read_bytes()
        assert m1["outputs"][fname]["sha256"] == m2["outputs"][fname]["sha256"]
        assert m1["outputs"][fname]["sha256"] == sha256_of_file(first / fname)


def test_rebuilding_in_place_reproduces_identical_bytes(source_file: Path, tmp_path: Path):
    out_dir = tmp_path / "robustness_v3"
    build_perturbation_sets(source_file, out_dir)
    before = {f: (out_dir / f).read_bytes() for f, _t, _v in VARIANTS}
    build_perturbation_sets(source_file, out_dir)
    assert {f: (out_dir / f).read_bytes() for f, _t, _v in VARIANTS} == before


def test_build_leaves_the_source_file_untouched(source_file: Path, tmp_path: Path):
    before = sha256_of_file(source_file)
    build_perturbation_sets(source_file, tmp_path / "robustness_v3")
    assert sha256_of_file(source_file) == before


def test_build_refuses_to_write_into_the_frozen_package(source_file: Path):
    with pytest.raises(SystemExit):
        build_perturbation_sets(source_file, "data/frozen/dashboard_v3")


def test_build_fails_loudly_on_a_missing_source(tmp_path: Path):
    with pytest.raises(SystemExit):
        build_perturbation_sets(tmp_path / "does_not_exist.jsonl", tmp_path / "out")


def test_manifest_carries_provenance_fields(source_file: Path, tmp_path: Path):
    out_dir = tmp_path / "robustness_v3"
    manifest = build_perturbation_sets(source_file, out_dir, seed=7)
    on_disk = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert on_disk == manifest
    for key in ("generator", "generator_version", "description", "created_utc", "seed",
                "perturbation_module", "pairing_key", "source", "outputs"):
        assert key in manifest
    assert manifest["seed"] == 7
    assert manifest["source"]["sha256"] == sha256_of_file(source_file)
    assert manifest["source"]["n_records"] == 4
    assert set(manifest["outputs"]) == {f for f, _t, _v in VARIANTS}
    for info in manifest["outputs"].values():
        assert set(info) == {"variant", "n_records", "n_briefs_changed", "sha256"}


# --------------------------------------------------------- the built artifact


@pytest.mark.skipif(not (_BUILT_DIR / "test_paraphrased.jsonl").exists(),
                    reason="robustness_v3 perturbations not built")
def test_built_sets_match_the_frozen_test_split():
    frozen_ids = [r["item_id"] for r in read_jsonl(_FROZEN_TEST)]
    for fname, _transform, _variant in VARIANTS:
        records = read_jsonl(_BUILT_DIR / fname)
        assert len(records) == len(frozen_ids)
        assert {r["item_id"] for r in records} == set(frozen_ids)


@pytest.mark.skipif(not (_BUILT_DIR / MANIFEST_NAME).exists(),
                    reason="robustness_v3 perturbations not built")
def test_built_manifest_hashes_match_the_files_on_disk():
    manifest = json.loads((_BUILT_DIR / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["source"]["path"] == DEFAULT_SOURCE
    assert manifest["source"]["sha256"] == sha256_of_file(_FROZEN_TEST)
    for fname, info in manifest["outputs"].items():
        assert info["sha256"] == sha256_of_file(_BUILT_DIR / fname)
