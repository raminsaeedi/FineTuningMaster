"""Guardrails for the supervisor result package.

The package must stay small and safe: no adapter weights, no trainer
checkpoints, no credentials — and every included file must be listed with a
verifiable hash so the receiving side can check integrity.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from experiments.scripts.package_results import (
    build_manifest,
    collect_result_files,
    collect_run_files,
    exclusion_reason,
    is_wanted_run_file,
    main,
    write_package,
)


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _make_run(root: Path, run_id: str = "E03_qwen0_5b_ft_42") -> Path:
    """Build a realistic run folder: wanted artifacts + everything forbidden."""
    run = root / run_id
    # wanted
    _write(run / "predictions.jsonl", '{"item_id": "a"}\n')
    _write(run / "predictions_paraphrased.jsonl", '{"item_id": "a"}\n')
    _write(run / "errors.jsonl", '{"item_id": "b"}\n')
    _write(run / "metrics_auto.json", '{"n_predictions": 1}')
    _write(run / "metrics.json", "{}")
    _write(run / "eval_per_item.jsonl", '{"item_id": "a"}\n')
    _write(run / "manifest.json", '{"seed": 42}')
    _write(run / "config_snapshot.yaml", "seed: 42\n")
    _write(run / "config_hash.txt", "abc123\n")
    _write(run / "git_hash.txt", "deadbeef\n")
    _write(run / "env.txt", "torch==2.6.0\n")
    _write(run / "logs" / "run.log", "started\n")
    _write(run / "adapter" / "training_metadata.json", '{"steps": 100}')
    # forbidden
    _write(run / "adapter" / "adapter_model.safetensors", "WEIGHTS")
    _write(run / "adapter" / "adapter_model.bin", "WEIGHTS")
    _write(run / "adapter" / "tokenizer.json", "{}")
    _write(run / "checkpoints" / "checkpoint-100" / "optimizer.pt", "STATE")
    _write(run / "checkpoints" / "checkpoint-100" / "trainer_state.json", "{}")
    _write(run / ".cache" / "models--Qwen--Qwen2.5-0.5B" / "model.safetensors", "BASE")
    _write(run / ".env", "OPENAI_API_KEY=sk-secret\n")
    _write(run / "logs" / "api_key.txt", "sk-secret")
    _write(run / "__pycache__" / "helper.cpython-311.pyc", "bytecode")
    # not part of the contract
    _write(run / "README.md", "notes")
    return run


def _arcnames(entries) -> set[str]:
    return {arcname for _, arcname in entries}


# --- exclusion rules -------------------------------------------------------

@pytest.mark.parametrize("rel", [
    "E03_42/adapter/adapter_model.safetensors",
    "E03_42/adapter/adapter_model.bin",
    "E03_42/adapter/model.safetensors",
    "E03_42/pytorch_model.bin",
    "E03_42/optimizer.pt",
    "E03_42/weights.pth",
])
def test_model_weights_are_excluded(rel):
    assert exclusion_reason(rel) is not None


@pytest.mark.parametrize("rel", [
    "E03_42/checkpoints/checkpoint-100/trainer_state.json",
    "E03_42/checkpoints/checkpoint-500/config.json",
])
def test_checkpoints_are_excluded(rel):
    assert exclusion_reason(rel) is not None


@pytest.mark.parametrize("rel", [
    "E03_42/.env",
    "E03_42/.env.local",
    "E03_42/logs/api_key.txt",
    "E03_42/HF_TOKEN.json",
    "E03_42/client_secret.yaml",
    "E03_42/secrets/creds.json",
])
def test_secrets_are_excluded(rel):
    assert exclusion_reason(rel) is not None


@pytest.mark.parametrize("rel", [
    "E03_42/.cache/models--Qwen--Qwen2.5-0.5B/model.safetensors",
    "E03_42/hf_cache/blobs/abc",
    "E03_42/huggingface/hub/config.json",
])
def test_base_model_cache_is_excluded(rel):
    assert exclusion_reason(rel) is not None


@pytest.mark.parametrize("rel", ["E03_42/__pycache__/x.cpython-311.pyc", "E03_42/x.pyc"])
def test_bytecode_is_excluded(rel):
    assert exclusion_reason(rel) is not None


@pytest.mark.parametrize("rel", [
    "E03_42/predictions.jsonl",
    "E03_42/metrics_auto.json",
    "E03_42/logs/run.log",
    "E03_42/adapter/training_metadata.json",
])
def test_wanted_artifacts_are_not_excluded(rel):
    assert exclusion_reason(rel) is None
    assert is_wanted_run_file(rel)


def test_adapter_directory_ships_only_the_training_summary():
    assert is_wanted_run_file("E03_42/adapter/training_metadata.json")
    assert not is_wanted_run_file("E03_42/adapter/tokenizer.json")
    assert not is_wanted_run_file("E03_42/adapter/adapter_config.json")


# --- collection ------------------------------------------------------------

def test_collect_run_files_keeps_contract_and_drops_the_rest(tmp_path):
    outputs_root = tmp_path / "outputs"
    _make_run(outputs_root)
    names = _arcnames(collect_run_files(outputs_root))

    run = "outputs/E03_qwen0_5b_ft_42"
    assert names == {
        f"{run}/predictions.jsonl",
        f"{run}/predictions_paraphrased.jsonl",
        f"{run}/errors.jsonl",
        f"{run}/metrics_auto.json",
        f"{run}/metrics.json",
        f"{run}/eval_per_item.jsonl",
        f"{run}/manifest.json",
        f"{run}/config_snapshot.yaml",
        f"{run}/config_hash.txt",
        f"{run}/git_hash.txt",
        f"{run}/env.txt",
        f"{run}/logs/run.log",
        f"{run}/adapter/training_metadata.json",
    }
    assert not any("safetensors" in n or "checkpoints" in n for n in names)


def test_collect_result_files_takes_everything_safe(tmp_path):
    results = tmp_path / "results"
    _write(results / "comparison_table.csv", "a,b\n")
    _write(results / "final_report.md", "# report")
    _write(results / "stats" / "mcnemar.csv", "a,b\n")
    _write(results / "__pycache__" / "x.pyc", "bytecode")

    names = _arcnames(collect_result_files(results))
    assert names == {
        "results/comparison_table.csv",
        "results/final_report.md",
        "results/stats/mcnemar.csv",
    }


def test_missing_outputs_root_fails_with_guidance(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main([
            "--outputs-root", str(tmp_path / "nope"),
            "--results-dir", str(tmp_path / "results"),
            "--out", str(tmp_path / "professor_results.zip"),
        ])
    assert "Run the experiments first" in str(exc.value)


# --- manifest + packaging --------------------------------------------------

def test_manifest_lists_included_files_with_hashes(tmp_path):
    outputs_root = tmp_path / "outputs"
    run = _make_run(outputs_root)
    entries = collect_run_files(outputs_root)
    manifest = build_manifest(entries)

    assert manifest["counts"]["total_files"] == len(entries)
    assert manifest["counts"]["run_artifact_files"] == len(entries)
    assert manifest["created_utc"]
    assert manifest["total_bytes"] == sum(f["size_bytes"] for f in manifest["files"])

    by_path = {f["path"]: f for f in manifest["files"]}
    expected = hashlib.sha256(
        (run / "predictions.jsonl").read_bytes()).hexdigest()
    record = by_path["outputs/E03_qwen0_5b_ft_42/predictions.jsonl"]
    assert record["sha256"] == expected
    assert record["size_bytes"] == (run / "predictions.jsonl").stat().st_size
    assert all(len(f["sha256"]) == 64 for f in manifest["files"])


def test_zip_contains_manifest_and_only_selected_members(tmp_path):
    outputs_root = tmp_path / "outputs"
    _make_run(outputs_root)
    results = tmp_path / "results"
    _write(results / "final_report.md", "# report")
    out = tmp_path / "professor_results.zip"

    main([
        "--outputs-root", str(outputs_root),
        "--results-dir", str(results),
        "--out", str(out),
    ])

    assert out.exists()
    with zipfile.ZipFile(out) as zf:
        members = set(zf.namelist())
        manifest = json.loads(zf.read("PACKAGE_MANIFEST.json"))

    assert "PACKAGE_MANIFEST.json" in members
    assert "results/final_report.md" in members
    assert "outputs/E03_qwen0_5b_ft_42/adapter/training_metadata.json" in members
    assert not any(".safetensors" in m or "/checkpoints/" in m or ".env" in m for m in members)
    assert {f["path"] for f in manifest["files"]} == members - {"PACKAGE_MANIFEST.json"}


def test_dry_run_writes_nothing(tmp_path, capsys):
    outputs_root = tmp_path / "outputs"
    _make_run(outputs_root)
    results = tmp_path / "results"
    _write(results / "final_report.md", "# report")
    out = tmp_path / "professor_results.zip"

    main([
        "--outputs-root", str(outputs_root),
        "--results-dir", str(results),
        "--out", str(out),
        "--dry-run",
    ])

    assert not out.exists()
    stdout = capsys.readouterr().out
    assert "Dry run" in stdout
    assert "predictions.jsonl" in stdout


def test_write_package_can_be_reproduced_from_manifest(tmp_path):
    outputs_root = tmp_path / "outputs"
    _make_run(outputs_root)
    entries = collect_run_files(outputs_root)
    manifest = build_manifest(entries)
    out = write_package(entries, manifest, tmp_path / "pkg.zip")

    with zipfile.ZipFile(out) as zf:
        for record in manifest["files"]:
            data = zf.read(record["path"])
            assert hashlib.sha256(data).hexdigest() == record["sha256"]
