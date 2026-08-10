"""Regression tests for the Phase-3 human-audit template (offline, no API call).

The pilot bug: the template received only the automatically accepted records, so
the CSV had 29 of 30 rows and the automatically rejected candidate never reached
human review.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "run_enrichment_sample", _ROOT / "experiments" / "scripts" / "run_enrichment_sample.py")
runner = importlib.util.module_from_spec(_SPEC)
sys.modules["run_enrichment_sample"] = runner
_SPEC.loader.exec_module(runner)


def _record(item_id: str, split: str = "train", chart: str = "bar") -> dict:
    return {
        "item_id": item_id,
        "split": split,
        "brief": {"item_id": item_id, "users": "template persona", "kpis": ["COUNT(x)"],
                  "columns": [{"name": "x", "dtype": "categorical"}]},
        "recommendation": {
            "context_summary": {"n_kpis": 1},
            "kpi_chart_mapping": [{"kpi": "COUNT(x)", "task_type": "comparison", "chart_type": chart,
                                   "encoding": {"x": "x", "y": "COUNT(x)"}}],
            "layout": {"type": "single"}, "styling": {"theme": "minimal"},
            "interactions": [{"type": "tooltip", "fields": ["x"]}],
            "rationales": [{"claim": "source", "principle": "source"}],
        },
    }


def _enriched(record: dict) -> dict:
    merged = {**record, "brief": {**record["brief"], "users": "enriched persona"},
              "recommendation": {**record["recommendation"], "styling": {"theme": "enriched"}}}
    return merged


def _fixture():
    selected = [_record("id:1"), _record("id:2", split="val"), _record("id:3")]
    accepted = [_enriched(selected[0]), _enriched(selected[2])]
    rejected = [{"item_id": "id:2", "split": "val",
                 "reason_codes": ["rationale_disagrees_with_task_chart_or_encoding"],
                 "details": ["no rationale mentions an encoded field"]}]
    raw = [{"item_id": "id:2", "response_text":
            '{"users": "rejected persona", "context_summary": {"n_kpis": 1}, '
            '"layout": {"type": "single"}, "styling": {"theme": "draft"}, '
            '"interactions": [{"type": "tooltip", "fields": ["x"]}], '
            '"rationales": [{"claim": "c", "principle": "p"}]}'}]
    return selected, accepted, rejected, raw


def test_rejected_records_are_included_in_the_human_audit_template():
    selected, accepted, rejected, raw = _fixture()
    rows = runner.build_audit_rows(selected, accepted, rejected, raw)
    assert len(rows) == len(selected) == 3
    assert [r["item_id"] for r in rows] == ["id:1", "id:2", "id:3"]  # selection order, not status
    statuses = {r["item_id"]: r["automatic_validation_status"] for r in rows}
    assert statuses == {"id:1": "accepted", "id:2": "rejected", "id:3": "accepted"}


def test_validation_columns_carry_existing_reason_codes_only():
    selected, accepted, rejected, raw = _fixture()
    rows = {r["item_id"]: r for r in runner.build_audit_rows(selected, accepted, rejected, raw)}
    assert rows["id:2"]["automatic_reason_codes"] == "rationale_disagrees_with_task_chart_or_encoding"
    assert rows["id:2"]["automatic_validation_details"] == "no rationale mentions an encoded field"
    assert rows["id:1"]["automatic_reason_codes"] == ""
    assert rows["id:1"]["automatic_validation_details"] == ""


def test_rejected_row_shows_the_unmerged_model_content():
    selected, accepted, rejected, raw = _fixture()
    rows = {r["item_id"]: r for r in runner.build_audit_rows(selected, accepted, rejected, raw)}
    assert rows["id:2"]["users"] == "rejected persona"
    assert "draft" in rows["id:2"]["styling"]
    # Accepted rows show the merged enrichment.
    assert rows["id:1"]["users"] == "enriched persona"
    assert "enriched" in rows["id:1"]["styling"]


def test_source_facts_come_from_the_immutable_record():
    selected, accepted, rejected, raw = _fixture()
    rows = {r["item_id"]: r for r in runner.build_audit_rows(selected, accepted, rejected, raw)}
    for item_id in ("id:1", "id:2", "id:3"):
        assert rows[item_id]["chart_type"] == "bar"
        assert rows[item_id]["kpi"] == "COUNT(x)"
        assert rows[item_id]["encoding_x"] == "x"


def test_all_human_columns_stay_blank_in_the_written_csv(tmp_path):
    selected, accepted, rejected, raw = _fixture()
    path = tmp_path / "audit.csv"
    runner._write_audit_template(path, runner.build_audit_rows(selected, accepted, rejected, raw))
    with path.open(encoding="utf-8", newline="") as f:
        written = list(csv.DictReader(f))
    assert len(written) == 3
    assert list(written[0]) == runner.AUDIT_HEADER
    for row in written:
        for column in runner.AUDIT_HUMAN_COLUMNS:
            assert row[column] == ""


def test_missing_raw_reply_does_not_break_the_template():
    selected, accepted, rejected, _ = _fixture()
    rows = {r["item_id"]: r for r in runner.build_audit_rows(selected, accepted, rejected, [])}
    assert rows["id:2"]["automatic_validation_status"] == "rejected"
    assert rows["id:2"]["users"] is None
