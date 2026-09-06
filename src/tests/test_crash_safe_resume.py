"""Crash-safe resume for inference and training.

The property under test is narrow and absolute: an interruption may cost the
remaining work, never the finished work. Every test here runs in a temporary
directory with a stub method — no model is loaded and no repository result
folder is touched.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.core.schemas import DashboardBrief, GenerationResult
from src.inference.runner import InferenceRunner
from src.inference.status import InferenceStatusRecorder, inference_identity
from src.utils.resume import (
    CHECKPOINT_COMPLETE_FILENAME,
    STATUS_COMPLETED,
    STATUS_RUNNING,
    IncompatibleResumeError,
    RunStatus,
    atomic_write_json,
    quarantine_files,
    repair_trailing_partial_line,
    source_code_hash,
)

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "train_script_resume", _ROOT / "experiments" / "scripts" / "train.py"
)
train_script = importlib.util.module_from_spec(_SPEC)
sys.modules["train_script_resume"] = train_script
_SPEC.loader.exec_module(train_script)

CONFIG_HASH = "cfg0123456789"


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class StubMethod:
    """Deterministic method that can be told to die after N items."""

    name = "stub"

    def __init__(self, interrupt_after: int | None = None, fail_ids: set[str] | None = None) -> None:
        self.config_hash = CONFIG_HASH
        self.interrupt_after = interrupt_after
        self.fail_ids = fail_ids or set()
        self.generated: list[str] = []
        self.setup_calls = 0
        self.teardown_calls = 0

    def setup(self) -> None:
        self.setup_calls += 1

    def teardown(self) -> None:
        self.teardown_calls += 1

    def generate(self, brief: DashboardBrief) -> GenerationResult:
        if self.interrupt_after is not None and len(self.generated) >= self.interrupt_after:
            # A killed process, not a recoverable item error.
            raise KeyboardInterrupt("simulated interruption")
        self.generated.append(brief.item_id)
        if brief.item_id in self.fail_ids:
            raise ValueError(f"item failed: {brief.item_id}")
        return GenerationResult(
            item_id=brief.item_id,
            method_name=self.name,
            model_name="stub-model",
            config_hash=self.config_hash,
            raw_text=f'{{"item": "{brief.item_id}"}}',
            latency_ms=1.0,
        )


def _briefs(n: int) -> list[DashboardBrief]:
    return [
        DashboardBrief(item_id=f"item_{index:03d}", users="analysts",
                       goals=["track revenue"], kpis=["revenue"])
        for index in range(n)
    ]


def _records(path: Path) -> list[dict]:
    """Strict read: every line must be valid JSON, unlike the tolerant reader."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _ids(path: Path) -> list[str]:
    return [record["item_id"] for record in _records(path)]


# ---------------------------------------------------------------------------
# 1. Interrupt inference after item 3 and resume
# ---------------------------------------------------------------------------

def test_interrupted_inference_resumes_without_losing_or_duplicating_work(tmp_path):
    out = tmp_path / "predictions.jsonl"
    briefs = _briefs(10)

    crashed = StubMethod(interrupt_after=3)
    with pytest.raises(KeyboardInterrupt):
        InferenceRunner(crashed, out).run(briefs)

    before = _records(out)
    assert [record["item_id"] for record in before] == ["item_000", "item_001", "item_002"]
    assert crashed.teardown_calls == 1  # the model was released on the way out

    resumed = StubMethod()
    results = InferenceRunner(resumed, out).run(briefs)

    after = _records(out)
    # 1. items 1-3 are unchanged, byte for byte.
    assert after[:3] == before
    # 2. no duplicate ids.
    item_ids = [record["item_id"] for record in after]
    assert len(item_ids) == len(set(item_ids))
    # 3. only the missing items were generated.
    assert resumed.generated == [f"item_{index:03d}" for index in range(3, 10)]
    # 4. coverage is complete and in dataset order.
    assert item_ids == [brief.item_id for brief in briefs]
    assert len(results) == 10


def test_resume_never_appends_an_item_that_is_already_present(tmp_path):
    """The write path refuses a duplicate even if a caller hands one over."""
    out = tmp_path / "predictions.jsonl"
    briefs = _briefs(3)
    InferenceRunner(StubMethod(), out).run(briefs)

    class ReplayMethod(StubMethod):
        def generate(self, brief):  # noqa: D401 - always returns item_000
            return GenerationResult(
                item_id="item_000",
                method_name=self.name,
                model_name="stub-model",
                config_hash=CONFIG_HASH,
                raw_text="{}",
            )

    runner = InferenceRunner(ReplayMethod(), out)
    runner._written_ids = runner._load_done()
    with out.open("a", encoding="utf-8") as handle:
        runner._write(handle, ReplayMethod().generate(briefs[0]), "original", briefs[0], 1, 1)

    assert _ids(out) == ["item_000", "item_001", "item_002"]


def test_failed_items_are_retried_on_resume_and_recorded(tmp_path):
    out = tmp_path / "predictions.jsonl"
    briefs = _briefs(4)
    recorder = InferenceStatusRecorder(tmp_path, {"config_hash": CONFIG_HASH})
    recorder.start()
    InferenceRunner(StubMethod(fail_ids={"item_001"}), out, recorder=recorder).run(briefs)

    assert _ids(out) == ["item_000", "item_002", "item_003"]

    second = StubMethod()
    InferenceRunner(second, out, recorder=recorder).run(briefs)
    recorder.finish()

    assert second.generated == ["item_001"]
    assert _ids(out) == ["item_000", "item_002", "item_003", "item_001"]
    status = json.loads((tmp_path / "inference_status.json").read_text(encoding="utf-8"))
    retries = [entry for entry in status["retries"] if entry["item_id"] == "item_001"]
    assert retries and retries[0]["reason"] == "failed in an earlier attempt"
    # A resampled item is never claimed to reproduce the lost one.
    assert retries[0]["bitwise_equality_verified"] is False


# ---------------------------------------------------------------------------
# 2. Incomplete trailing JSONL line
# ---------------------------------------------------------------------------

def test_incomplete_trailing_line_is_removed_and_earlier_records_survive(tmp_path):
    out = tmp_path / "predictions.jsonl"
    briefs = _briefs(6)
    InferenceRunner(StubMethod(interrupt_after=3), out).run  # not executed
    InferenceRunner(StubMethod(), out).run(briefs[:3])
    intact = _records(out)

    # A process killed mid-append leaves a fragment with no trailing newline.
    with out.open("a", encoding="utf-8") as handle:
        handle.write('{"item_id": "item_003", "method_name": "stu')

    repair = repair_trailing_partial_line(out)

    assert repair is not None
    assert repair.recovered_item_id == "item_003"
    assert repair.kept_records == 3
    assert _records(out) == intact           # nothing valid was discarded
    assert repair.backup_path.exists()       # the fragment is preserved for audit


def test_resume_after_a_torn_write_produces_a_clean_complete_file(tmp_path):
    out = tmp_path / "predictions.jsonl"
    briefs = _briefs(6)
    InferenceRunner(StubMethod(), out).run(briefs[:3])
    with out.open("a", encoding="utf-8") as handle:
        handle.write('{"item_id": "item_003", "raw_text": "half')

    recorder = InferenceStatusRecorder(tmp_path, {"config_hash": CONFIG_HASH})
    recorder.start()
    method = StubMethod()
    InferenceRunner(method, out, recorder=recorder).run(briefs, variant="original")
    recorder.finish()

    # Strict parsing: without the repair the next append would have fused onto
    # the fragment and destroyed item_003 as well.
    assert _ids(out) == [brief.item_id for brief in briefs]
    assert method.generated == ["item_003", "item_004", "item_005"]
    status = json.loads((tmp_path / "inference_status.json").read_text(encoding="utf-8"))
    assert status["repairs"][0]["recovered_item_id"] == "item_003"
    assert status["retries"][0]["item_id"] == "item_003"
    assert status["status"] == STATUS_COMPLETED
    assert status["variants"]["original"]["n_completed"] == 6
    assert status["variants"]["original"]["has_duplicate_ids"] is False


def test_a_complete_file_is_never_touched(tmp_path):
    out = tmp_path / "predictions.jsonl"
    InferenceRunner(StubMethod(), out).run(_briefs(3))
    before = out.read_bytes()

    assert repair_trailing_partial_line(out) is None
    assert out.read_bytes() == before


def test_a_trailing_invalid_line_with_a_newline_is_also_removed(tmp_path):
    out = tmp_path / "predictions.jsonl"
    InferenceRunner(StubMethod(), out).run(_briefs(2))
    with out.open("a", encoding="utf-8") as handle:
        handle.write('{"item_id": "item_002", "raw\n')

    repair = repair_trailing_partial_line(out)

    assert repair is not None
    assert _ids(out) == ["item_000", "item_001"]


# ---------------------------------------------------------------------------
# 3 & 4. Training checkpoints
# ---------------------------------------------------------------------------

def _write_checkpoint(root: Path, step: int, *, complete: bool = True) -> Path:
    path = root / f"checkpoint-{step}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "epoch": 1.0}), encoding="utf-8"
    )
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "optimizer.pt").write_bytes(b"optimizer")
    (path / "scheduler.pt").write_bytes(b"scheduler")
    (path / "rng_state.pth").write_bytes(b"rng")
    if complete:
        (path / CHECKPOINT_COMPLETE_FILENAME).write_text(
            json.dumps({"global_step": step}), encoding="utf-8"
        )
    return path


def test_resume_starts_from_the_saved_global_step_not_zero(tmp_path):
    root = tmp_path / "checkpoints"
    _write_checkpoint(root, 4)
    _write_checkpoint(root, 12)

    selected = train_script.find_latest_checkpoint(root)

    assert selected == root / "checkpoint-12"
    assert train_script.checkpoint_global_step(selected) == 12
    assert train_script.checkpoint_global_step(selected) != 0


@pytest.mark.parametrize(
    "missing",
    ["optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"],
)
def test_a_checkpoint_missing_resume_state_is_invalid(tmp_path, missing):
    """Weights alone are not a resumption: the schedule and RNG must come back."""
    checkpoint = _write_checkpoint(tmp_path / "checkpoints", 10)
    (checkpoint / missing).unlink()

    problems = train_script.checkpoint_problems(checkpoint)

    assert problems and not train_script.is_valid_checkpoint(checkpoint)


def test_a_corrupt_newest_checkpoint_falls_back_to_the_previous_valid_one(tmp_path):
    root = tmp_path / "checkpoints"
    _write_checkpoint(root, 10)
    newest = _write_checkpoint(root, 20)
    # Interrupted mid-write: the state file exists but never finished.
    (newest / "trainer_state.json").write_text('{"global_step": 2', encoding="utf-8")

    selected = train_script.find_latest_checkpoint(root)

    assert selected == root / "checkpoint-10"
    assert train_script.checkpoint_global_step(selected) == 10


def test_a_partially_written_checkpoint_is_never_selected(tmp_path):
    root = tmp_path / "checkpoints"
    _write_checkpoint(root, 10)
    partial = root / "checkpoint-20"
    partial.mkdir(parents=True)
    (partial / "adapter_model.safetensors").write_bytes(b"weights")

    assert train_script.find_latest_checkpoint(root) == root / "checkpoint-10"


def test_completion_marker_is_written_atomically(tmp_path):
    from src.utils.checkpoints import mark_checkpoint_complete

    checkpoint = _write_checkpoint(tmp_path / "checkpoints", 30, complete=False)
    marker = mark_checkpoint_complete(checkpoint, global_step=30)

    assert marker is not None and marker.exists()
    assert json.loads(marker.read_text(encoding="utf-8"))["global_step"] == 30
    # No temporary file is left behind next to it.
    assert not list(checkpoint.glob("*.tmp"))


def test_oom_report_preserves_the_checkpoint_and_prints_the_resume_command(tmp_path, capsys):
    import argparse

    root = tmp_path / "checkpoints"
    checkpoint = _write_checkpoint(root, 15)
    status = RunStatus(path=tmp_path / "training_status.json", kind="training", identity={})
    args = argparse.Namespace(
        experiment="E03_qwen0_5b_ft",
        override=["model=qwen2_5_0_5b", "seed=42"],
    )

    train_script._report_interrupted_training(
        status, tmp_path, args, "cuda_out_of_memory", RuntimeError("CUDA out of memory")
    )

    out = capsys.readouterr().out
    assert "CUDA OUT OF MEMORY" in out
    assert "--experiment E03_qwen0_5b_ft --resume" in out
    assert "model=qwen2_5_0_5b" in out
    # Nothing was reconfigured or removed.
    assert checkpoint.exists() and (checkpoint / "optimizer.pt").exists()
    record = json.loads((tmp_path / "training_status.json").read_text(encoding="utf-8"))
    assert record["failure_reason"] == "cuda_out_of_memory"
    assert record["last_valid_global_step"] == 15
    assert record["last_valid_checkpoint"].endswith("checkpoint-15")


def test_cuda_oom_is_recognised():
    assert train_script._is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert not train_script._is_cuda_oom(RuntimeError("dataset not found"))


# ---------------------------------------------------------------------------
# 5. Identity gating
# ---------------------------------------------------------------------------

def _identity(**overrides) -> dict:
    identity = {
        "config_hash": CONFIG_HASH,
        "code_hash": "code00000001",
        "cache_identity": {
            "dataset_version": "dashboard_v4",
            "dataset_hashes": {"test": "abc"},
            "model_hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "model_revision": None,
            "method": "A",
            "seed": 42,
            "kb_hash": "kb123",
        },
    }
    identity.update(overrides)
    return identity


def _nested_identity(**cache_overrides) -> dict:
    identity = _identity()
    identity["cache_identity"] = {**identity["cache_identity"], **cache_overrides}
    return identity


@pytest.mark.parametrize(
    ("label", "identity"),
    [
        ("dataset", _nested_identity(dataset_version="dashboard_v3")),
        ("dataset hash", _nested_identity(dataset_hashes={"test": "different"})),
        ("model", _nested_identity(model_hf_id="meta-llama/Llama-3.1-8B")),
        ("revision", _nested_identity(model_revision="deadbeef")),
        ("seed", _nested_identity(seed=43)),
        ("knowledge base", _nested_identity(kb_hash="kb999")),
        ("method", _nested_identity(method="B")),
        ("config hash", _identity(config_hash="different0001")),
        ("code hash", _identity(code_hash="code99999999")),
    ],
)
def test_resume_refuses_incompatible_artifacts(tmp_path, label, identity):
    InferenceStatusRecorder(tmp_path, _identity()).start()

    with pytest.raises(IncompatibleResumeError) as exc:
        InferenceStatusRecorder(tmp_path, identity).start()

    assert "--no-resume" in str(exc.value)


def test_matching_identity_resumes_and_keeps_the_original_start_time(tmp_path):
    first = InferenceStatusRecorder(tmp_path, _identity())
    first.start()
    started = json.loads((tmp_path / "inference_status.json").read_text(encoding="utf-8"))["started_utc"]

    second = InferenceStatusRecorder(tmp_path, _identity())
    second.start()
    record = json.loads((tmp_path / "inference_status.json").read_text(encoding="utf-8"))

    assert record["started_utc"] == started
    assert record["attempts"] == 2
    assert record["status"] == STATUS_RUNNING


def test_no_resume_skips_the_identity_gate(tmp_path):
    InferenceStatusRecorder(tmp_path, _identity()).start()

    fresh = InferenceStatusRecorder(tmp_path, _identity(config_hash="other"), resume=False)
    fresh.start()  # must not raise

    record = json.loads((tmp_path / "inference_status.json").read_text(encoding="utf-8"))
    assert record["identity"]["config_hash"] == "other"
    assert record["attempts"] == 1


def test_inference_identity_carries_the_gated_fields():
    identity = inference_identity(
        {"config_hash": CONFIG_HASH},
        _ROOT,
        {"model_hf_id": "x", "seed": 42, "kb_hash": "k", "method": "A"},
    )

    assert identity["config_hash"] == CONFIG_HASH
    assert identity["code_hash"] and len(identity["code_hash"]) == 12
    assert identity["cache_identity"]["seed"] == 42


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def test_source_code_hash_is_stable_and_tracks_source_changes(tmp_path):
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    module = tmp_path / "src" / "pkg" / "thing.py"
    module.write_text("x = 1\n", encoding="utf-8")

    first = source_code_hash(tmp_path, use_cache=False)
    assert first == source_code_hash(tmp_path, use_cache=False)

    module.write_text("x = 2\n", encoding="utf-8")
    assert source_code_hash(tmp_path, use_cache=False) != first


def test_source_code_hash_ignores_test_files(tmp_path):
    (tmp_path / "src" / "tests").mkdir(parents=True)
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "thing.py").write_text("x = 1\n", encoding="utf-8")
    before = source_code_hash(tmp_path, use_cache=False)

    (tmp_path / "src" / "tests" / "test_thing.py").write_text("assert True\n", encoding="utf-8")

    assert source_code_hash(tmp_path, use_cache=False) == before


def test_atomic_write_json_leaves_no_temporary_file(tmp_path):
    target = tmp_path / "status.json"
    atomic_write_json(target, {"a": 1})
    atomic_write_json(target, {"a": 2})

    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 2}
    assert [path.name for path in tmp_path.iterdir()] == ["status.json"]


def test_quarantine_moves_files_instead_of_deleting_them(tmp_path):
    source = tmp_path / "predictions.jsonl"
    source.write_text('{"item_id": "a"}\n', encoding="utf-8")

    moved = quarantine_files([source], tmp_path / "_stale_cache" / "fresh")

    assert not source.exists()
    assert moved[0].read_text(encoding="utf-8") == '{"item_id": "a"}\n'


def test_status_write_is_readable_after_every_transition(tmp_path):
    status = RunStatus(path=tmp_path / "s.json", kind="inference", identity={"a": "b"})
    status.write(STATUS_RUNNING, note="one")
    status.write(STATUS_COMPLETED, note="two")

    record = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert record["status"] == STATUS_COMPLETED
    assert record["note"] == "two"
    assert record["identity"] == {"a": "b"}
