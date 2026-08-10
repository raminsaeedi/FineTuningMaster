"""Focused tests for the Phase-3 full-run orchestration (offline, no API call)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data_pipeline.enrichment import ENRICHABLE_FIELDS
from src.data_pipeline.enrichment_full import (
    EXPECTED_COUNTS,
    LINEAGE_LLM,
    MAX_RETRIES,
    annotate_lineage,
    assert_allowed_input,
    backoff_seconds,
    classify_error,
    config_fingerprint,
    is_reusable_response,
    evaluate_quality_gates,
    load_raw_cache,
    load_split,
    pending_records,
    summarize_split,
    validate_and_merge,
    verify_human_r1,
    write_json_atomic,
    write_jsonl_atomic,
)


def _record(item_id: str, split: str = "train", chart: str = "bar", task: str = "comparison") -> dict:
    return {
        "item_id": item_id,
        "split": split,
        "brief": {
            "item_id": item_id, "users": "source persona",
            "goals": ["Bar chart of product counts."], "kpis": ["COUNT(product_name)"],
            "columns": [{"name": "product_name", "dtype": "categorical", "role": "dimension"}],
            "constraints": None,
            "extra": {
                "source": "nvbench", "usage_tier": "train_aug",
                "provenance": {"db_id": "db", "source_group_id": f"grp:{item_id}",
                               "source_record_id": item_id},
                "task_inference": {"task_type": task},
                "lineage": {"chart_type": "source-provided", "kpi": "source-provided",
                            "task_type": "rule-derived", "layout": "template-derived"},
            },
        },
        "recommendation": {
            "context_summary": {"db_id": "db", "n_kpis": 1},
            "kpi_chart_mapping": [{"kpi": "COUNT(product_name)", "task_type": task,
                                   "chart_type": chart, "alternatives": [],
                                   "encoding": {"x": "product_name", "y": "COUNT(product_name)"}}],
            "layout": {"type": "single"}, "styling": {"theme": "minimal"},
            "interactions": [{"type": "tooltip", "fields": ["product_name"]}],
            "rationales": [{"claim": "source", "principle": "source"}],
        },
    }


def _reply(chart: str = "bar") -> dict:
    return {
        "users": "Product analyst comparing product counts.",
        "context_summary": {"db_id": "db", "n_kpis": 1, "data_scope": "products by name"},
        "layout": {"type": "single", "blocks": [{"kpi": "COUNT(product_name)",
                                                 "chart": "the selected chart"}]},
        "styling": {"theme": "minimal", "emphasis": "readable ordered categories"},
        "interactions": [{"type": "tooltip", "fields": ["product_name", "COUNT(product_name)"]}],
        "rationales": [{"claim": "The selected chart supports comparison of COUNT(product_name) "
                                 "across product_name.",
                        "principle": "position on a common scale"}],
    }


# ------------------------------------------------------------ input scope


def test_expected_train_and_val_counts_are_the_contract():
    assert EXPECTED_COUNTS == {"train": 1281, "val": 264}
    assert sum(EXPECTED_COUNTS.values()) == 1545


@pytest.mark.parametrize("name", ["test.jsonl", "human_eval_test_items_40.csv",
                                  "reports/human_eval_test_items_40.jsonl"])
def test_held_out_inputs_are_refused(name):
    with pytest.raises(ValueError):
        assert_allowed_input(name)


def test_load_split_rejects_unknown_split_and_mislabelled_records(tmp_path):
    (tmp_path / "train.jsonl").write_text(
        json.dumps(_record("a", split="train")) + "\n" + json.dumps(_record("b", split="test")) + "\n",
        encoding="utf-8")
    with pytest.raises(ValueError):
        load_split(tmp_path, "train")
    with pytest.raises(ValueError):
        load_split(tmp_path, "test")


def test_load_split_keeps_file_order_and_rejects_duplicates(tmp_path):
    ids = ["c", "a", "b"]
    (tmp_path / "val.jsonl").write_text(
        "".join(json.dumps(_record(i, split="val")) + "\n" for i in ids), encoding="utf-8")
    assert [r["item_id"] for r in load_split(tmp_path, "val")] == ids

    (tmp_path / "val.jsonl").write_text(
        "".join(json.dumps(_record(i, split="val")) + "\n" for i in ["a", "a"]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_split(tmp_path, "val")


@pytest.mark.skipif(
    not Path("data/staging/dashboard_v3/nvbench_large_v2/train.jsonl").exists(),
    reason="pre-freeze staging artifacts are gitignored and absent in a fresh clone",
)
def test_real_repository_splits_match_the_expected_counts():
    source = "data/staging/dashboard_v3/nvbench_large_v2"
    train, val = load_split(source, "train"), load_split(source, "val")
    assert (len(train), len(val)) == (1281, 264)
    assert {r["split"] for r in train} == {"train"}
    assert {r["split"] for r in val} == {"val"}
    assert not ({r["item_id"] for r in train} & {r["item_id"] for r in val})


# ------------------------------------------------------------ cache/resume


def test_cache_reuses_only_matching_configuration(tmp_path):
    path = tmp_path / "raw_responses.jsonl"
    write_jsonl_atomic([
        {"item_id": "a", "config_hash": "h1", "response_text": "{}"},
        {"item_id": "b", "config_hash": "h2", "response_text": "{}"},   # other config
        {"item_id": "c", "config_hash": "h1", "response_text": ""},      # failed, not reusable
    ], path)
    cache, preserved = load_raw_cache(path, "h1")
    assert set(cache) == {"a"}
    assert len(preserved) == 3  # nothing lost


def test_missing_cache_file_is_not_an_error(tmp_path):
    assert load_raw_cache(tmp_path / "absent.jsonl", "h1") == ({}, [])


def test_resume_only_calls_the_records_without_a_cached_reply():
    records = [_record("a"), _record("b"), _record("c")]
    cache = {"a": {"response_text": "{}"}}
    todo = pending_records(records, cache)
    assert [r["item_id"] for r in todo] == ["b", "c"]  # order preserved


def test_config_fingerprint_changes_with_every_generation_parameter():
    base = config_fingerprint("m", "https://u/v1", 0.0, "xhigh")
    assert base == config_fingerprint("m", "https://u/v1", 0.0, "xhigh")
    assert base != config_fingerprint("m2", "https://u/v1", 0.0, "xhigh")
    assert base != config_fingerprint("m", "https://u/v1", 0.1, "xhigh")
    assert base != config_fingerprint("m", "https://u2/v1", 0.0, "xhigh")
    assert base != config_fingerprint("m", "https://u/v1", 0.0, None)


def test_token_budget_is_not_part_of_the_fingerprint():
    """A cap only truncates; a completed reply is identical under a larger cap."""
    base = config_fingerprint("m", "https://u/v1", 0.0, "xhigh")
    legacy_3000 = config_fingerprint("m", "https://u/v1", 0.0, "xhigh", max_tokens=3000)
    legacy_4096 = config_fingerprint("m", "https://u/v1", 0.0, "xhigh", max_tokens=4096)
    assert base != legacy_3000            # legacy form stays distinguishable
    assert legacy_3000 != legacy_4096     # ... and budget-specific
    assert base == config_fingerprint("m", "https://u/v1", 0.0, "xhigh", max_tokens=None)


def test_legacy_cache_rows_are_still_reused(tmp_path):
    """Raising the budget must not re-call the already completed records."""
    legacy = config_fingerprint("m", "https://u/v1", 0.0, "xhigh", max_tokens=3000)
    current = config_fingerprint("m", "https://u/v1", 0.0, "xhigh")
    path = tmp_path / "raw_responses.jsonl"
    write_jsonl_atomic([
        {"item_id": "a", "config_hash": legacy, "response_text": "{}", "finish_reason": "stop",
         "max_tokens": 3000},
        {"item_id": "b", "config_hash": current, "response_text": "{}", "finish_reason": "stop",
         "max_tokens": 4096},
    ], path)
    cache, preserved = load_raw_cache(path, current, legacy_hashes=[legacy])
    assert set(cache) == {"a", "b"}
    assert len(preserved) == 2


def test_truncated_replies_are_never_reused():
    """finish_reason=length is an incomplete artifact, not a valid generation."""
    assert is_reusable_response({"response_text": "{...", "finish_reason": "length"}) is False
    assert is_reusable_response({"response_text": "{}", "finish_reason": "stop"}) is True
    assert is_reusable_response({"response_text": "", "finish_reason": "stop"}) is False


def test_truncated_row_is_recalled_while_completed_rows_stay_cached(tmp_path):
    current = config_fingerprint("m", "https://u/v1", 0.0, "xhigh")
    path = tmp_path / "raw_responses.jsonl"
    write_jsonl_atomic([
        {"item_id": "a", "config_hash": current, "response_text": "{}", "finish_reason": "stop"},
        {"item_id": "b", "config_hash": current, "response_text": '{"users": "cut',
         "finish_reason": "length"},
    ], path)
    cache, _ = load_raw_cache(path, current)
    todo = pending_records([_record("a"), _record("b")], cache)
    assert [r["item_id"] for r in todo] == ["b"]


# --------------------------------------------------------- atomic writing


def test_atomic_writers_leave_no_temp_file_and_write_full_content(tmp_path):
    jsonl = tmp_path / "out.jsonl"
    write_jsonl_atomic([{"a": 1}, {"a": 2}], jsonl)
    assert [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines()] == [{"a": 1}, {"a": 2}]

    js = tmp_path / "out.json"
    write_json_atomic({"k": "v"}, js)
    assert json.loads(js.read_text(encoding="utf-8")) == {"k": "v"}
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_rewrite_replaces_content_without_partial_state(tmp_path):
    path = tmp_path / "out.jsonl"
    write_jsonl_atomic([{"a": 1}], path)
    write_jsonl_atomic([{"a": 1}, {"a": 2}], path)
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


# ------------------------------------------------------------ retry policy


def test_bounded_retries_and_growing_backoff():
    assert MAX_RETRIES == 2
    assert backoff_seconds(1) < backoff_seconds(2) < backoff_seconds(3)


class _Err(Exception):
    def __init__(self, message="", status=None):
        super().__init__(message)
        if status is not None:
            self.status_code = status


@pytest.mark.parametrize("exc,kind", [
    (_Err(status=401), "fatal"),
    (_Err(status=404), "fatal"),
    (_Err(status=429), "transient"),
    (_Err(status=503), "transient"),
    (_Err("Connection reset by peer"), "transient"),
    (_Err("Request timed out"), "transient"),
    (_Err(status=400), "permanent"),
    (_Err("unparsable nonsense"), "permanent"),
])
def test_error_classification(exc, kind):
    assert classify_error(exc) == kind


# ------------------------------------------------- validation accounting


def test_every_input_record_gets_a_validation_result_accepted_or_not():
    records = [_record("a"), _record("b")]
    results = []
    accepted = []
    for record, reply in zip(records, [_reply(), None]):
        merged, result = validate_and_merge(record, reply, {"model": "m"})
        results.append(result)
        if merged is not None:
            accepted.append(merged)
    summary = summarize_split(records, results)
    assert summary["accepted"] + summary["rejected"] == summary["input_records"] == 2
    assert len(accepted) == 1
    assert summary["reason_codes"] == {"response_not_json": 1}


def test_immutable_violation_is_reported_and_blocks_acceptance():
    record = _record("a")
    reply = _reply()
    reply["context_summary"] = {"db_id": "other_db", "n_kpis": 1}  # contradicts source
    merged, result = validate_and_merge(record, reply, {"model": "m"})
    assert merged is None
    assert "context_summary_conflicts_source" in result["reason_codes"]
    assert result["immutable_fingerprint_before"] == result["immutable_fingerprint_after"]


def test_accepted_record_keeps_the_immutable_fingerprint_and_gains_lineage():
    record = _record("a")
    merged, result = validate_and_merge(record, _reply(), {"model": "m"})
    assert merged is not None and result["accepted"]
    assert result["immutable_fingerprint_before"] == result["immutable_fingerprint_after"]
    enrichment = merged["brief"]["extra"]["enrichment"]
    assert enrichment["field_lineage"] == {f: LINEAGE_LLM for f in ENRICHABLE_FIELDS}
    assert enrichment["lineage_classes"][LINEAGE_LLM] == list(ENRICHABLE_FIELDS)
    assert "chart_type" in enrichment["lineage_classes"]["source_backed"]
    assert "task_type" in enrichment["lineage_classes"]["deterministically_derived"]


def test_lineage_annotation_never_claims_gold_status():
    merged, _ = validate_and_merge(_record("a"), _reply(), {"model": "m"})
    blob = json.dumps(merged).lower()
    assert "gold" not in blob
    assert merged["brief"]["extra"]["enrichment"]["annotation_kind"] == "llm_generated_design_annotation"


# ------------------------------------------------------------ quality gates


def _summary(n=100, accepted=100, schema=None, immutable=0, charts=None, reasons=None):
    schema = n if schema is None else schema
    return {
        "input_records": n, "results": n, "accepted": accepted, "rejected": n - accepted,
        "schema_valid": schema, "schema_valid_rate": schema / n, "accept_rate": accepted / n,
        "immutable_violations": immutable, "reason_codes": reasons or {},
        "per_chart_type": charts or {"bar": {"n": n, "accepted": accepted,
                                             "accept_rate": accepted / n}},
    }


def test_gate_passes_on_a_clean_full_run():
    summaries = {"train": _summary(1281, 1281), "val": _summary(264, 264)}
    status, failures = evaluate_quality_gates(summaries, cache_verified=True)
    assert failures == []
    assert status == "PASS_FULL_ENRICHMENT_READY_FOR_HYBRID_DATASET"


def test_gate_fails_on_wrong_input_count():
    summaries = {"train": _summary(1280, 1280), "val": _summary(264, 264)}
    status, failures = evaluate_quality_gates(summaries, cache_verified=True)
    assert status == "FULL_ENRICHMENT_QUALITY_GATE_FAIL"
    assert any("train input count" in f for f in failures)


def test_gate_fails_on_low_accept_rate_and_on_immutable_violation():
    summaries = {"train": _summary(1281, 1100), "val": _summary(264, 264)}
    status, failures = evaluate_quality_gates(summaries, cache_verified=True)
    assert status == "FULL_ENRICHMENT_QUALITY_GATE_FAIL"
    assert any("accept rate" in f for f in failures)

    summaries = {"train": _summary(1281, 1281, immutable=1), "val": _summary(264, 264)}
    _, failures = evaluate_quality_gates(summaries, cache_verified=True)
    assert any("immutable-field violations" in f for f in failures)


def test_gate_fails_when_test_or_human_eval_records_were_processed():
    summaries = {"train": _summary(1281, 1281), "val": _summary(264, 264)}
    _, failures = evaluate_quality_gates(summaries, test_records_processed=3, cache_verified=True)
    assert any("test records processed" in f for f in failures)
    _, failures = evaluate_quality_gates(summaries, human_eval_items_processed=40, cache_verified=True)
    assert any("human-evaluation items processed" in f for f in failures)


def test_gate_fails_on_duplicate_processing_and_flags_systematic_chart_failure():
    summaries = {"train": _summary(1281, 1281), "val": _summary(264, 264)}
    _, failures = evaluate_quality_gates(summaries, duplicate_items=2, cache_verified=True)
    assert any("processed more than once" in f for f in failures)

    charts = {"bar": {"n": 1200, "accepted": 1200, "accept_rate": 1.0},
              "scatter": {"n": 81, "accepted": 10, "accept_rate": 0.1235}}
    summaries = {"train": _summary(1281, 1210, charts=charts), "val": _summary(264, 264)}
    _, failures = evaluate_quality_gates(summaries, cache_verified=True)
    assert any("systematic failure for chart type scatter" in f for f in failures)


def test_gate_reports_incomplete_when_results_are_missing():
    summaries = {"train": {**_summary(1281, 1281), "results": 1000},
                 "val": _summary(264, 264)}
    status, failures = evaluate_quality_gates(summaries, cache_verified=True)
    assert status == "FULL_ENRICHMENT_INCOMPLETE"
    assert any("accounted for" in f for f in failures)


# --------------------------------------------------------------- human gate


@pytest.mark.skipif(
    not Path(
        "data/staging/enrichment/pilot_30/manual_enrichment_audit_template_30_R1.csv"
    ).exists(),
    reason="pre-freeze staging artifacts are gitignored and absent in a fresh clone",
)
def test_human_r1_gate_reads_the_real_file_without_modifying_it():
    path = "data/staging/enrichment/pilot_30/manual_enrichment_audit_template_30_R1.csv"
    before = open(path, "rb").read()
    gate = verify_human_r1(path)
    assert open(path, "rb").read() == before
    assert gate["rows"] == 30
    assert gate["accepted"] >= 27
    assert gate["gate_passed"] is True
    assert gate["rows_flagging_immutable_violation"] == []
    # Reviewer identity and any placeholder cell are recorded, never assumed.
    assert gate["reviewer_ids"] == ["R1"]
    assert gate["review_kind"] == "human_review"
    assert isinstance(gate["rows_with_placeholder_values"], list)


def test_human_r1_gate_fails_on_too_few_accepted(tmp_path):
    path = tmp_path / "r1.csv"
    header = "item_id,reviewer_id,overall_accept,immutable_violation_found\n"
    rows = "".join(f"id{i},R1,{1 if i < 20 else 0},none found\n" for i in range(30))
    path.write_text(header + rows, encoding="utf-8")
    gate = verify_human_r1(path)
    assert gate["accepted"] == 20
    assert gate["gate_passed"] is False


def test_human_r1_gate_fails_on_incomplete_reviewer_fields(tmp_path):
    path = tmp_path / "r1.csv"
    header = "item_id,reviewer_id,overall_accept,immutable_violation_found\n"
    rows = "".join(f"id{i},{'R1' if i else ''},1,none found\n" for i in range(30))
    path.write_text(header + rows, encoding="utf-8")
    assert verify_human_r1(path)["gate_passed"] is False
