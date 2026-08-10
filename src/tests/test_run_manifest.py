"""Run manifest provenance: dataset, KB, adapter, dirty tree and completion.

A manifest that omits which knowledge base or adapter a run consumed cannot be
audited afterwards, and a commit hash recorded from a dirty tree is misleading.
These tests pin the fields that make a run reconstructible.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from src.utils import git as git_utils
from src.utils.artifacts import (
    _adapter_provenance,
    _kb_provenance,
    finalize_manifest,
    write_manifest,
)

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _cfg(tmp_path: Path, **overrides) -> dict:
    cfg = {
        "output_root": str(tmp_path / "outputs"),
        "experiment_id": "E03_qwen0_5b_ft_42",
        "experiment_name": "E03_qwen0_5b_ft",
        "seed": 42,
        "model": {"name": "qwen2_5_0_5b", "hf_id": BASE_MODEL},
        "data": {"dataset_version": "dashboard_v3"},
        "method": {"name": "ft", "type": "fine_tuned",
                   "adapter_path": None, "adapter_source_experiment": None},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return cfg


@pytest.fixture(autouse=True)
def _no_real_git(monkeypatch):
    """Keep manifests deterministic and off the real repository state."""
    monkeypatch.setattr("src.utils.artifacts.get_git_hash", lambda: "deadbeef")
    monkeypatch.setattr("src.utils.artifacts.is_git_dirty", lambda: False)


def _make_kb(kb_dir: Path, *, manifest: bool = True) -> Path:
    kb_dir.mkdir(parents=True, exist_ok=True)
    chunks = kb_dir / "chunks.jsonl"
    chunks.write_text('{"id": "a_0", "text": "use a line chart for trends"}\n',
                      encoding="utf-8")
    if manifest:
        (kb_dir / "kb_manifest.json").write_text(
            json.dumps({"kb_version": "kb_2f8c1a", "n_chunks": 1}), encoding="utf-8")
    return chunks


# --- write_manifest --------------------------------------------------------

def test_manifest_records_release_provenance_keys(tmp_path):
    manifest = write_manifest(tmp_path, _cfg(tmp_path))

    for key in ("dataset_version", "git_dirty", "knowledge_base", "adapter"):
        assert key in manifest
    assert manifest["dataset_version"] == "dashboard_v3"
    assert manifest["git_dirty"] is False
    assert set(manifest["knowledge_base"]) == {
        "kb_version", "chunks_path", "chunks_sha256"}
    assert set(manifest["adapter"]) == {"adapter_path", "adapter_training_metadata"}


def test_manifest_is_written_to_disk_and_reloadable(tmp_path):
    written = write_manifest(tmp_path, _cfg(tmp_path))
    on_disk = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == written
    assert on_disk["experiment_id"] == "E03_qwen0_5b_ft_42"
    assert on_disk["seed"] == 42
    assert len(on_disk["config_hash"]) == 12


def test_manifest_marks_a_dirty_working_tree(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.artifacts.is_git_dirty", lambda: True)
    assert write_manifest(tmp_path, _cfg(tmp_path))["git_dirty"] is True


def test_manifest_dataset_version_is_null_when_the_config_omits_it(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["data"] = {}
    assert write_manifest(tmp_path, cfg)["dataset_version"] is None


# --- adapter provenance ----------------------------------------------------

@pytest.mark.parametrize("method_type", ["prompt_only", "rag"])
def test_adapter_provenance_is_null_for_methods_without_an_adapter(tmp_path, method_type):
    provenance = _adapter_provenance(_cfg(tmp_path, method={"type": method_type}))
    assert provenance == {"adapter_path": None, "adapter_training_metadata": None}


@pytest.mark.parametrize("method_type", ["fine_tuned", "fine_tuned_rag"])
def test_adapter_provenance_records_the_resolved_path(tmp_path, method_type):
    cfg = _cfg(tmp_path, method={
        "type": method_type, "adapter_source_experiment": "E03_qwen0_5b_ft"})
    provenance = _adapter_provenance(cfg)

    expected = tmp_path / "outputs" / "E03_qwen0_5b_ft_42" / "adapter"
    assert Path(provenance["adapter_path"]) == expected
    # No adapter on disk yet -> nothing to report about its training.
    assert provenance["adapter_training_metadata"] is None


def test_adapter_provenance_includes_the_recorded_training_metadata(tmp_path):
    adapter = tmp_path / "outputs" / "E03_qwen0_5b_ft_43" / "adapter"
    adapter.mkdir(parents=True)
    metadata = {"base_model": BASE_MODEL, "seed": 43, "dataset_version": "dashboard_v3"}
    (adapter / "training_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    cfg = _cfg(tmp_path, seed=43, method={
        "type": "fine_tuned_rag", "adapter_source_experiment": "E03_qwen0_5b_ft"})
    provenance = _adapter_provenance(cfg)

    # This is what proves afterwards that D seed 43 consumed the C seed-43 adapter.
    assert provenance["adapter_training_metadata"] == metadata
    assert provenance["adapter_path"].endswith(str(Path("E03_qwen0_5b_ft_43") / "adapter"))


def test_manifest_carries_the_adapter_provenance(tmp_path):
    cfg = _cfg(tmp_path, method={
        "type": "fine_tuned_rag", "adapter_source_experiment": "E03_qwen0_5b_ft"})
    assert write_manifest(tmp_path, cfg)["adapter"]["adapter_path"] is not None


# --- knowledge base provenance --------------------------------------------

def test_kb_provenance_is_null_without_a_retriever_config(tmp_path):
    assert _kb_provenance(_cfg(tmp_path)) == {
        "kb_version": None, "chunks_path": None, "chunks_sha256": None}


def test_kb_provenance_records_version_and_chunk_hash(tmp_path, monkeypatch):
    chunks = _make_kb(tmp_path / "data" / "knowledge_base")
    monkeypatch.chdir(tmp_path)

    cfg = _cfg(tmp_path, method={
        "type": "rag",
        "retriever": {"name": "tfidf", "top_k": 3,
                      "chunks_path": "data/knowledge_base/chunks.jsonl"}})
    provenance = _kb_provenance(cfg)

    assert provenance["kb_version"] == "kb_2f8c1a"
    assert provenance["chunks_path"] == "data/knowledge_base/chunks.jsonl"
    assert provenance["chunks_sha256"] == hashlib.sha256(chunks.read_bytes()).hexdigest()


def test_kb_provenance_hash_changes_with_the_chunks_file(tmp_path, monkeypatch):
    chunks = _make_kb(tmp_path / "kb")
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(tmp_path, method={
        "type": "rag", "retriever": {"chunks_path": "kb/chunks.jsonl"}})

    before = _kb_provenance(cfg)["chunks_sha256"]
    with chunks.open("a", encoding="utf-8") as f:
        f.write('{"id": "a_1", "text": "bar charts compare categories"}\n')
    assert _kb_provenance(cfg)["chunks_sha256"] != before


def test_kb_provenance_survives_a_missing_manifest_and_missing_chunks(tmp_path, monkeypatch):
    _make_kb(tmp_path / "kb", manifest=False)
    monkeypatch.chdir(tmp_path)

    cfg = _cfg(tmp_path, method={
        "type": "rag", "retriever": {"chunks_path": "kb/chunks.jsonl"}})
    provenance = _kb_provenance(cfg)
    assert provenance["kb_version"] is None
    assert provenance["chunks_sha256"] is not None

    absent = _cfg(tmp_path, method={
        "type": "rag", "retriever": {"chunks_path": "kb/absent.jsonl"}})
    assert _kb_provenance(absent)["chunks_sha256"] is None


def test_manifest_carries_the_kb_provenance(tmp_path, monkeypatch):
    _make_kb(tmp_path / "kb")
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(tmp_path, method={
        "type": "rag", "retriever": {"chunks_path": "kb/chunks.jsonl"}})

    manifest = write_manifest(tmp_path, cfg)
    assert manifest["knowledge_base"]["kb_version"] == "kb_2f8c1a"


# --- finalize_manifest -----------------------------------------------------

def test_finalize_stamps_completion_and_duration(tmp_path):
    write_manifest(tmp_path, _cfg(tmp_path))
    finalized = finalize_manifest(tmp_path)

    assert finalized["status"] == "completed"
    assert finalized["finished_utc"]
    assert isinstance(finalized["duration_seconds"], (int, float))
    assert finalized["duration_seconds"] >= 0

    on_disk = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "completed"
    assert on_disk["finished_utc"] == finalized["finished_utc"]


def test_finalize_keeps_the_original_fields(tmp_path):
    original = write_manifest(tmp_path, _cfg(tmp_path))
    finalized = finalize_manifest(tmp_path, status="failed")

    assert finalized["status"] == "failed"
    for key, value in original.items():
        assert finalized[key] == value


def test_finalize_is_a_no_op_without_a_manifest(tmp_path):
    assert finalize_manifest(tmp_path) is None
    assert not (tmp_path / "manifest.json").exists()


def test_finalize_tolerates_a_corrupt_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    assert finalize_manifest(tmp_path) is None


def test_finalize_without_a_start_time_reports_no_duration(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"experiment_id": "x"}),
                                            encoding="utf-8")
    finalized = finalize_manifest(tmp_path)
    assert finalized["status"] == "completed"
    assert "duration_seconds" not in finalized


# --- dirty-tree detection --------------------------------------------------

def test_is_git_dirty_reads_the_porcelain_status(monkeypatch):
    monkeypatch.setattr(git_utils.subprocess, "check_output", lambda *a, **k: b"")
    assert git_utils.is_git_dirty() is False

    monkeypatch.setattr(git_utils.subprocess, "check_output",
                        lambda *a, **k: b" M src/utils/git.py\n")
    assert git_utils.is_git_dirty() is True


def test_is_git_dirty_is_none_when_git_is_unavailable(monkeypatch):
    def _boom(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(git_utils.subprocess, "check_output", _boom)
    # None means "unknown", which must stay distinguishable from "clean".
    assert git_utils.is_git_dirty() is None
