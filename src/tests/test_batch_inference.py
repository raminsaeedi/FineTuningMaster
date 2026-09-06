"""Optional batched inference: safety gate, plumbing, and equality.

Three layers are covered here.

1. **The gate** (:mod:`src.inference.batching`) — pure config logic. Batching is
   off unless it is both configured *and* acknowledged, and the decision is
   recorded in the run metadata.
2. **The plumbing** — left padding, attention masks, output slicing, item order,
   resume, error logging. These must be identical to the sequential path.
3. **Equality** — batch 1 vs batch N over a fixed 20-item fixture, compared
   per item.

Note on what layer 3 proves: the fake model here is a deterministic function of
each row's *unpadded* tokens, so a mismatch means the batching plumbing is
wrong. It says nothing about a real LLM, where batching changes both the
sampling stream and the numerics — which is exactly why the gate exists and why
``experiments/scripts/benchmark_batch_inference.py`` measures a real model.
"""

from __future__ import annotations

import json

import pytest
import torch

from src.core.schemas import DashboardBrief, GenerationResult
from src.inference.batching import (
    EQUIVALENCE_EXACT,
    EQUIVALENCE_NUMERIC,
    EQUIVALENCE_SAMPLING,
    UnsafeBatchingError,
    batching_provenance,
    resolve_batch_plan,
)
from src.inference.runner import InferenceRunner
from src.methods.base import HFMethod
from src.models.hf_causal import HFCausalModel, _real_token_ids, left_pad_batch

FIXTURE_SIZE = 20


# ---------------------------------------------------------------------------
# 1. The safety gate
# ---------------------------------------------------------------------------

def test_no_configuration_means_sequential():
    plan = resolve_batch_plan({"generate": {"do_sample": True}})

    assert plan.batch_size == 1
    assert plan.enabled is False
    assert plan.equivalence == EQUIVALENCE_EXACT


def test_explicit_batch_size_one_is_still_sequential():
    plan = resolve_batch_plan({"generate": {}, "inference": {"batch_size": 1}})

    assert plan.batch_size == 1
    assert plan.equivalence == EQUIVALENCE_EXACT


def test_sampling_batch_without_acknowledgement_is_refused():
    with pytest.raises(UnsafeBatchingError) as exc:
        resolve_batch_plan(
            {"generate": {"do_sample": True}, "inference": {"batch_size": 4}}
        )

    message = str(exc.value)
    assert "do_sample=true" in message
    assert "method.inference.allow_nonequivalent_batching" in message
    assert "benchmark_batch_inference.py" in message


def test_greedy_batch_without_acknowledgement_is_also_refused():
    """Padding and batch-shaped kernels are enough to lose bitwise identity."""
    with pytest.raises(UnsafeBatchingError) as exc:
        resolve_batch_plan(
            {"generate": {"do_sample": False}, "inference": {"batch_size": 2}}
        )

    assert "numerics" in str(exc.value)


def test_acknowledged_sampling_batch_is_recorded_as_non_equivalent():
    plan = resolve_batch_plan({
        "generate": {"do_sample": True},
        "inference": {"batch_size": 8, "allow_nonequivalent_batching": True},
    })

    assert plan.batch_size == 8
    assert plan.enabled is True
    assert plan.equivalence == EQUIVALENCE_SAMPLING
    assert plan.latency_semantics == "amortized-per-batch"


def test_acknowledged_greedy_batch_is_recorded_as_numerically_unproven():
    plan = resolve_batch_plan({
        "generate": {"do_sample": False},
        "inference": {"batch_size": 2, "allow_nonequivalent_batching": True},
    })

    assert plan.equivalence == EQUIVALENCE_NUMERIC


@pytest.mark.parametrize("value", [0, -1, "four"])
def test_invalid_batch_sizes_are_rejected(value):
    with pytest.raises(ValueError):
        resolve_batch_plan({"generate": {}, "inference": {"batch_size": value}})


def test_constrained_decoding_cannot_be_batched():
    with pytest.raises(UnsafeBatchingError, match="Constrained"):
        resolve_batch_plan({
            "generate": {"constrained": True},
            "inference": {"batch_size": 4, "allow_nonequivalent_batching": True},
        })


def test_provenance_records_the_sequential_default():
    record = batching_provenance({"method": {"generate": {"do_sample": True}}})

    assert record["batch_size"] == 1
    assert record["equivalence"] == EQUIVALENCE_EXACT


def test_provenance_records_a_rejected_configuration_without_raising():
    record = batching_provenance(
        {"method": {"generate": {"do_sample": True}, "inference": {"batch_size": 4}}}
    )

    assert record["equivalence"] == "rejected"
    assert record["requested_batch_size"] == 4
    assert "allow_nonequivalent_batching" in record["note"]


def test_manifest_records_the_inference_regime(tmp_path):
    from src.utils.artifacts import write_manifest

    manifest = write_manifest(tmp_path, {
        "experiment_id": "x", "seed": 42,
        "model": {"name": "fake"}, "method": {"name": "prompt_only", "type": "prompt_only",
                                              "generate": {"do_sample": True}},
        "data": {"dataset_version": "dashboard_v4"},
    })

    assert manifest["inference_batching"]["batch_size"] == 1
    assert manifest["inference_batching"]["equivalence"] == EQUIVALENCE_EXACT
    on_disk = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["inference_batching"]["batch_size"] == 1


# ---------------------------------------------------------------------------
# 2. Config-hash stability: the feature must be invisible until it is used
# ---------------------------------------------------------------------------

EXPERIMENTS = (
    "E01_qwen0_5b_prompt",
    "E02_qwen0_5b_rag",
    "E03_qwen0_5b_ft",
    "E04_qwen0_5b_ft_rag",
)


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_batching_keys_are_absent_from_the_default_composition(experiment):
    """A default key would change config_hash and invalidate existing caches."""
    from src.utils.config import load_cfg

    cfg = load_cfg(experiment=experiment)
    inference_cfg = cfg.method.get("inference", {}) or {}

    assert "batch_size" not in inference_cfg
    assert "allow_nonequivalent_batching" not in inference_cfg
    assert resolve_batch_plan(cfg.method).batch_size == 1


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_enabling_batching_changes_the_config_hash(experiment):
    """Batched and sequential predictions must never share one result file."""
    from src.utils.config import load_cfg

    sequential = load_cfg(experiment=experiment)
    batched = load_cfg(experiment=experiment, overrides=[
        "+method.inference.batch_size=4",
        "+method.inference.allow_nonequivalent_batching=true",
    ])

    assert sequential.config_hash != batched.config_hash
    assert resolve_batch_plan(batched.method).batch_size == 4


# ---------------------------------------------------------------------------
# 3. Padding and attention masks
# ---------------------------------------------------------------------------

def test_left_pad_batch_pads_on_the_left_and_masks_the_padding():
    input_ids, attention_mask = left_pad_batch([[5, 6, 7], [8], [9, 10]], pad_id=0)

    assert input_ids.tolist() == [[5, 6, 7], [0, 0, 8], [0, 9, 10]]
    assert attention_mask.tolist() == [[1, 1, 1], [0, 0, 1], [0, 1, 1]]
    # Every row's last position is a real token: the model continues from there.
    assert [row[-1] for row in attention_mask.tolist()] == [1, 1, 1]


def test_left_pad_batch_leaves_a_single_row_untouched():
    input_ids, attention_mask = left_pad_batch([[1, 2, 3]], pad_id=0)

    assert input_ids.tolist() == [[1, 2, 3]]
    assert attention_mask.tolist() == [[1, 1, 1]]


def test_real_token_ids_honours_an_incoming_mask():
    inputs = {
        "input_ids": torch.tensor([[0, 0, 4, 5]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1]]),
    }

    assert _real_token_ids(inputs) == [4, 5]


# ---------------------------------------------------------------------------
# Fake model: deterministic in each row's unpadded tokens
# ---------------------------------------------------------------------------

class _CharTokenizer:
    """One token per character. eos/pad share id 0, which no prompt contains."""

    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return "|".join(message["content"] for message in messages)

    def __call__(self, text, return_tensors=None, **kwargs):
        del kwargs
        ids = [ord(character) for character in text]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            }
        return {"input_ids": ids}

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        del kwargs
        ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        if skip_special_tokens:
            ids = [token for token in ids if token != self.eos_token_id]
        return "".join(chr(token) for token in ids)


class _DeterministicModel(torch.nn.Module):
    """Continuation depends only on the row's *unpadded* tokens.

    So sequential and batched output may only differ if padding, masking,
    slicing or ordering is wrong — which is precisely what is under test.
    """

    NEW_TOKENS = 16

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.batch_sizes: list[int] = []

    def generate(self, input_ids, attention_mask=None, **kwargs):
        del kwargs
        self.batch_sizes.append(int(input_ids.shape[0]))
        rows = []
        for index in range(int(input_ids.shape[0])):
            ids = input_ids[index].tolist()
            if attention_mask is not None:
                real = [
                    token
                    for token, flag in zip(ids, attention_mask[index].tolist())
                    if flag == 1
                ]
            else:
                real = ids
            text = f'{{"n": {len(real)}, "c": {sum(real) % 9973}}}'
            new = [ord(character) for character in text][: self.NEW_TOKENS]
            new += [0] * (self.NEW_TOKENS - len(new))
            rows.append(ids + new)
        return torch.tensor(rows, dtype=torch.long)


def _brief(index: int, users: str | None = None) -> DashboardBrief:
    return DashboardBrief(
        item_id=f"item_{index:03d}",
        users=users if users is not None else f"analyst cohort {index}",
        goals=[f"goal {index}", "track revenue"],
        kpis=[f"kpi_{index}", "revenue"],
        columns=[{"name": f"col_{index}", "dtype": "float"}],
    )


def _fixture(n: int = FIXTURE_SIZE) -> list[DashboardBrief]:
    return [_brief(index) for index in range(n)]


class _FakeHFMethod(HFMethod):
    """Method A wiring with the model injected instead of downloaded."""

    name = "prompt_only"

    def setup(self) -> None:  # the fake model is already attached
        return None

    def teardown(self) -> None:
        return None


def _method(batch_size: int, *, max_seq_length: int = 100_000, max_new_tokens: int = 16):
    inference: dict = {}
    if batch_size > 1:
        inference = {
            "batch_size": batch_size,
            "allow_nonequivalent_batching": True,
        }
    cfg = {
        "model": {"name": "fake-model", "max_seq_length": max_seq_length},
        "method": {
            "name": "prompt_only",
            "generate": {"max_new_tokens": max_new_tokens, "do_sample": False},
            "inference": inference,
        },
        "seed": 42,
        "config_hash": "batchtest0001",
    }
    method = _FakeHFMethod(cfg)
    model = HFCausalModel({"max_seq_length": max_seq_length})
    model.tokenizer = _CharTokenizer()
    model.model = _DeterministicModel()
    model.max_seq_length = max_seq_length
    method.model = model
    return method


# ---------------------------------------------------------------------------
# 4. Method-level behaviour
# ---------------------------------------------------------------------------

def test_batch_size_one_keeps_the_sequential_path():
    method = _method(batch_size=1)

    assert method.inference_batch_size == 1
    outcomes = method.generate_batch(_fixture(3))

    # The default BaseMethod loop runs one item per generate call.
    assert method.model.model.batch_sizes == [1, 1, 1]
    assert [result.item_id for result in outcomes] == ["item_000", "item_001", "item_002"]


def test_batched_generation_matches_sequential_item_by_item():
    """The 20-item fixture: exact per-item equality, in order."""
    sequential = _method(batch_size=1)
    batched = _method(batch_size=4)
    briefs = _fixture()

    expected = [sequential.generate(brief) for brief in briefs]
    actual = batched.generate_batch(briefs)

    assert [result.item_id for result in actual] == [result.item_id for result in expected]
    for produced, reference in zip(actual, expected):
        assert produced.raw_text == reference.raw_text
        assert produced.prompt_input_tokens == reference.prompt_input_tokens
        assert produced.prompt_input_budget == reference.prompt_input_budget
        assert produced.parse_error == reference.parse_error
        assert produced.parsed == reference.parsed


def test_batched_generation_uses_one_call_per_chunk():
    method = _method(batch_size=5)

    method.generate_batch(_fixture(12)[:5])

    assert method.model.model.batch_sizes == [5]


def test_prompt_budget_error_fails_only_its_own_item():
    """An over-long prompt still raises for that item; the batch survives."""
    from src.core.prompts import SYSTEM_PROMPT, build_user_message

    probe = _method(batch_size=1)
    normal_tokens = probe.model.prompt_token_count(
        SYSTEM_PROMPT, build_user_message(_brief(0))
    )
    # Room for a normal item plus its output budget, nothing like 4000 extra.
    method = _method(batch_size=4, max_seq_length=normal_tokens + 100, max_new_tokens=16)
    briefs = [_brief(0), _brief(1, users="x" * 4000), _brief(2)]

    outcomes = method.generate_batch(briefs)

    assert isinstance(outcomes[1], ValueError)
    assert "Prompt exceeds" in str(outcomes[1])
    assert isinstance(outcomes[0], GenerationResult)
    assert isinstance(outcomes[2], GenerationResult)
    assert [outcomes[0].item_id, outcomes[2].item_id] == ["item_000", "item_002"]


def test_a_failing_batch_attributes_the_error_to_every_item_in_it():
    method = _method(batch_size=3)

    def explode(*_args, **_kwargs):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    method.model.generate_prepared_batch = explode
    outcomes = method.generate_batch(_fixture(3))

    assert all(isinstance(outcome, RuntimeError) for outcome in outcomes)


# ---------------------------------------------------------------------------
# 5. Runner-level behaviour
# ---------------------------------------------------------------------------

class _CountingMethod:
    """BaseMethod stand-in that records how the runner called it."""

    name = "counting"
    config_hash = "counthash0001"

    def __init__(self, batch_size: int = 1, failing_ids: set[str] | None = None) -> None:
        self.inference_batch_size = batch_size
        self.failing_ids = failing_ids or set()
        self.single_calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def _result(self, brief: DashboardBrief) -> GenerationResult:
        return GenerationResult(
            item_id=brief.item_id,
            method_name=self.name,
            model_name="fake-model",
            config_hash=self.config_hash,
            raw_text=f"out::{brief.item_id}",
            latency_ms=1.0,
        )

    def generate(self, brief: DashboardBrief) -> GenerationResult:
        self.single_calls.append(brief.item_id)
        if brief.item_id in self.failing_ids:
            raise ValueError(f"boom for {brief.item_id}")
        return self._result(brief)

    def generate_batch(self, briefs):
        self.batch_calls.append([brief.item_id for brief in briefs])
        outcomes = []
        for brief in briefs:
            if brief.item_id in self.failing_ids:
                outcomes.append(ValueError(f"boom for {brief.item_id}"))
            else:
                outcomes.append(self._result(brief))
        return outcomes


def _written_ids(path):
    return [json.loads(line)["item_id"] for line in path.read_text(encoding="utf-8").splitlines()]


def test_runner_default_never_calls_the_batch_path(tmp_path):
    method = _CountingMethod(batch_size=1)
    out = tmp_path / "predictions.jsonl"

    InferenceRunner(method, out).run(_fixture(5))

    assert method.batch_calls == []
    assert method.single_calls == [f"item_{i:03d}" for i in range(5)]
    assert _written_ids(out) == method.single_calls


def test_runner_batches_in_order_and_writes_ids_in_order(tmp_path):
    method = _CountingMethod(batch_size=6)
    out = tmp_path / "predictions.jsonl"

    InferenceRunner(method, out).run(_fixture())

    assert method.single_calls == []
    assert method.batch_calls == [
        [f"item_{i:03d}" for i in range(0, 6)],
        [f"item_{i:03d}" for i in range(6, 12)],
        [f"item_{i:03d}" for i in range(12, 18)],
        [f"item_{i:03d}" for i in range(18, 20)],
    ]
    assert _written_ids(out) == [f"item_{i:03d}" for i in range(FIXTURE_SIZE)]


def test_runner_resume_in_batch_mode_regenerates_only_missing_items(tmp_path):
    out = tmp_path / "predictions.jsonl"
    done = _CountingMethod(batch_size=4)
    InferenceRunner(done, out).run(_fixture(8))

    resumed = _CountingMethod(batch_size=4)
    InferenceRunner(resumed, out).run(_fixture(12))

    assert resumed.batch_calls == [[f"item_{i:03d}" for i in range(8, 12)]]
    assert _written_ids(out) == [f"item_{i:03d}" for i in range(12)]


def test_runner_logs_failed_items_in_batch_mode(tmp_path):
    method = _CountingMethod(batch_size=4, failing_ids={"item_001", "item_005"})
    out = tmp_path / "predictions.jsonl"
    runner = InferenceRunner(method, out)

    runner.run(_fixture(8))

    assert _written_ids(out) == [
        f"item_{i:03d}" for i in range(8) if i not in (1, 5)
    ]
    errors = [json.loads(line) for line in runner.errors_path.read_text().splitlines()]
    assert [error["item_id"] for error in errors] == ["item_001", "item_005"]
    assert all(error["error_type"] == "ValueError" for error in errors)
    # Formatted from the exception object, not from a live except block.
    assert all("ValueError" in error["traceback"] for error in errors)


def test_runner_stops_on_a_fatal_cuda_error_in_batch_mode(tmp_path):
    class _FatalMethod(_CountingMethod):
        def generate_batch(self, briefs):
            self.batch_calls.append([brief.item_id for brief in briefs])
            return [
                RuntimeError("CUDA error: device-side assert triggered")
                for _ in briefs
            ]

    method = _FatalMethod(batch_size=2)
    runner = InferenceRunner(method, tmp_path / "predictions.jsonl")

    with pytest.raises(RuntimeError, match="device-side assert triggered"):
        runner.run(_fixture(6))

    # Only the first chunk ran; the run aborted instead of burning the rest.
    assert method.batch_calls == [["item_000", "item_001"]]


def test_runner_rejects_a_mismatched_batch_result_length(tmp_path):
    class _ShortMethod(_CountingMethod):
        def generate_batch(self, briefs):
            return [self._result(briefs[0])]

    runner = InferenceRunner(_ShortMethod(batch_size=3), tmp_path / "predictions.jsonl")

    with pytest.raises(RuntimeError, match="item identity"):
        runner.run(_fixture(3))


# ---------------------------------------------------------------------------
# 6. The matrix runner's CLI gate
# ---------------------------------------------------------------------------

def _matrix_runner():
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "run_final_matrix", root / "experiments" / "scripts" / "run_final_matrix.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_final_matrix"] = module
    spec.loader.exec_module(module)
    return module


def test_matrix_runner_emits_nothing_for_the_default():
    """A sequential run must compose exactly the config it always composed."""
    assert _matrix_runner()._batching_overrides(1, False) == []
    assert _matrix_runner()._batching_overrides(1, True) == []


def test_matrix_runner_refuses_batching_without_acknowledgement():
    with pytest.raises(SystemExit) as exc:
        _matrix_runner()._batching_overrides(4, False)

    assert "--allow-nonequivalent-batching" in str(exc.value)


def test_matrix_runner_emits_both_keys_when_acknowledged():
    assert _matrix_runner()._batching_overrides(4, True) == [
        "+method.inference.batch_size=4",
        "+method.inference.allow_nonequivalent_batching=true",
    ]


def test_matrix_runner_rejects_a_zero_batch_size():
    with pytest.raises(SystemExit):
        _matrix_runner()._batching_overrides(0, True)


# ---------------------------------------------------------------------------
# 7. End to end: batch 1 vs batch N over the fixed 20-item fixture
# ---------------------------------------------------------------------------

_VOLATILE_FIELDS = {"latency_ms"}


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_batch_one_and_batch_n_produce_identical_prediction_files(tmp_path):
    briefs = _fixture()
    sequential_path = tmp_path / "seq" / "predictions.jsonl"
    batched_path = tmp_path / "batched" / "predictions.jsonl"

    InferenceRunner(_method(batch_size=1), sequential_path).run(briefs)
    InferenceRunner(_method(batch_size=4), batched_path).run(briefs)

    sequential = _records(sequential_path)
    batched = _records(batched_path)

    assert len(sequential) == FIXTURE_SIZE
    assert [row["item_id"] for row in sequential] == [row["item_id"] for row in batched]
    for left, right in zip(sequential, batched):
        assert {k: v for k, v in left.items() if k not in _VOLATILE_FIELDS} == {
            k: v for k, v in right.items() if k not in _VOLATILE_FIELDS
        }
    # Latency is still populated for every item, amortized in batch mode.
    assert all(row["latency_ms"] >= 0.0 for row in batched)
