"""The stale-config guard on the prediction cache.

Predictions are cached per ``item_id`` only. An override that changes generation
settings or the adapter — without changing the experiment name or seed, which is
what names the directory — would otherwise be served from the old file and mix
two configurations inside one result set. The runner must refuse instead.

No model is loaded here: the method is a stub implementing the BaseMethod surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.schemas import DashboardBrief, GenerationResult
from src.inference.runner import InferenceRunner

CURRENT_HASH = "a1b2c3d4e5f6"
OTHER_HASH = "999888777666"


class FakeMethod:
    """Minimal BaseMethod stand-in: no model, no tokenizer, no inference."""

    def __init__(self, config_hash: str = CURRENT_HASH) -> None:
        self.name = "fake"
        self.config_hash = config_hash
        self.setup_calls = 0
        self.teardown_calls = 0
        self.generated: list[str] = []

    def setup(self) -> None:
        self.setup_calls += 1

    def generate(self, brief: DashboardBrief) -> GenerationResult:
        self.generated.append(brief.item_id)
        return GenerationResult(
            item_id=brief.item_id,
            method_name=self.name,
            model_name="fake-model",
            config_hash=self.config_hash,
            raw_text="{}",
            latency_ms=1.0,
        )

    def teardown(self) -> None:
        self.teardown_calls += 1


def _briefs(n: int = 2) -> list[DashboardBrief]:
    return [DashboardBrief(item_id=f"item_{i}", users="analysts",
                           goals=["track revenue"], kpis=["revenue"])
            for i in range(n)]


def _write_predictions(path: Path, item_ids, config_hash: str | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item_id in item_ids:
            record = {"item_id": item_id, "method_name": "fake",
                      "model_name": "fake-model", "raw_text": "{}"}
            if config_hash is not None:
                record["config_hash"] = config_hash
            f.write(json.dumps(record) + "\n")
    return path


# --- cache hit -------------------------------------------------------------

def test_matching_config_hash_is_served_from_cache(tmp_path):
    out = _write_predictions(tmp_path / "predictions.jsonl",
                             ["item_0", "item_1"], CURRENT_HASH)
    method = FakeMethod()

    results = InferenceRunner(method, out).run(_briefs(2))

    assert [r.item_id for r in results] == ["item_0", "item_1"]
    assert method.setup_calls == 0     # nothing was loaded
    assert method.generated == []      # nothing was generated


def test_partial_cache_resumes_without_regenerating_done_items(tmp_path):
    out = _write_predictions(tmp_path / "predictions.jsonl", ["item_0"], CURRENT_HASH)
    method = FakeMethod()

    results = InferenceRunner(method, out).run(_briefs(3))

    assert method.generated == ["item_1", "item_2"]
    assert {r.item_id for r in results} == {"item_0", "item_1", "item_2"}
    assert method.teardown_calls == 1


# --- the guard -------------------------------------------------------------

def test_a_different_config_hash_aborts_the_run(tmp_path):
    out = _write_predictions(tmp_path / "predictions.jsonl", ["item_0"], OTHER_HASH)
    method = FakeMethod()

    with pytest.raises(RuntimeError) as exc:
        InferenceRunner(method, out).run(_briefs(2))

    message = str(exc.value)
    assert "config_hash" in message
    assert OTHER_HASH in message and CURRENT_HASH in message
    assert str(out) in message
    assert method.setup_calls == 0     # aborted before any model would load


def test_the_guard_fires_even_when_the_cache_is_complete(tmp_path):
    """A full cache under different settings is the dangerous case, not the safe one."""
    out = _write_predictions(tmp_path / "predictions.jsonl",
                             ["item_0", "item_1"], OTHER_HASH)

    with pytest.raises(RuntimeError):
        InferenceRunner(FakeMethod(), out).run(_briefs(2))


def test_mixed_hashes_in_one_file_are_all_reported(tmp_path):
    out = tmp_path / "predictions.jsonl"
    _write_predictions(out, ["item_0"], CURRENT_HASH)
    with out.open("a", encoding="utf-8") as f:
        for item_id, digest in (("item_1", OTHER_HASH), ("item_2", "111222333444")):
            f.write(json.dumps({"item_id": item_id, "method_name": "fake",
                                "model_name": "fake-model",
                                "config_hash": digest}) + "\n")

    with pytest.raises(RuntimeError) as exc:
        InferenceRunner(FakeMethod(), out).run(_briefs(3))
    assert OTHER_HASH in str(exc.value) and "111222333444" in str(exc.value)


# --- cases the guard must ignore ------------------------------------------

def test_absent_predictions_file_does_not_trigger_the_guard(tmp_path):
    out = tmp_path / "predictions.jsonl"
    method = FakeMethod()

    results = InferenceRunner(method, out).run(_briefs(2))

    assert method.generated == ["item_0", "item_1"]
    assert len(results) == 2
    assert out.exists()


def test_empty_predictions_file_does_not_trigger_the_guard(tmp_path):
    out = tmp_path / "predictions.jsonl"
    out.write_text("", encoding="utf-8")
    method = FakeMethod()

    results = InferenceRunner(method, out).run(_briefs(1))

    assert method.generated == ["item_0"]
    assert len(results) == 1


def test_records_without_a_config_hash_are_tolerated(tmp_path):
    """Predictions written before the field existed must stay usable."""
    out = _write_predictions(tmp_path / "predictions.jsonl", ["item_0"], None)
    method = FakeMethod()

    results = InferenceRunner(method, out).run(_briefs(1))

    assert method.setup_calls == 0            # served from cache, not regenerated
    assert [r.item_id for r in results] == ["item_0"]


def test_blank_config_hash_is_tolerated(tmp_path):
    out = _write_predictions(tmp_path / "predictions.jsonl", ["item_0"], "")
    method = FakeMethod()

    assert len(InferenceRunner(method, out).run(_briefs(1))) == 1


def test_no_guard_when_the_method_reports_no_config_hash(tmp_path):
    """Without an expected hash there is nothing to compare against."""
    out = _write_predictions(tmp_path / "predictions.jsonl", ["item_0"], OTHER_HASH)
    method = FakeMethod(config_hash="")

    assert len(InferenceRunner(method, out).run(_briefs(1))) == 1


# --- variant files ---------------------------------------------------------

def test_the_guard_applies_to_variant_prediction_files(tmp_path):
    out = _write_predictions(tmp_path / "predictions_paraphrased.jsonl",
                             ["item_0"], OTHER_HASH)

    with pytest.raises(RuntimeError):
        InferenceRunner(FakeMethod(), out).run(_briefs(1), variant="paraphrased")
