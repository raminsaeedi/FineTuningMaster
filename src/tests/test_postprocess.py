"""Tests for lenient post-processing of model JSON output, in particular the
``context_summary`` field which the 0.5B model sometimes returns as a plain
string instead of the object the schema (``Dict[str, Any]``) expects.
"""

import json

from src.inference.postprocess import parse_json_safe


def test_context_summary_string_is_coerced_to_object():
    raw = json.dumps({
        "context_summary": "Branch Managers want to monitor loan performance.",
        "kpi_chart_mapping": [],
        "layout": {},
        "styling": {},
        "interactions": [],
        "rationales": [],
    })
    parsed, err = parse_json_safe(raw)
    assert err is None, err
    assert isinstance(parsed.context_summary, dict)
    assert parsed.context_summary["summary"] == (
        "Branch Managers want to monitor loan performance."
    )


def test_context_summary_json_string_is_parsed():
    raw = json.dumps({
        "context_summary": json.dumps({"objective": "monitor risk"}),
        "kpi_chart_mapping": [],
        "layout": {},
        "styling": {},
        "interactions": [],
        "rationales": [],
    })
    parsed, err = parse_json_safe(raw)
    assert err is None, err
    assert parsed.context_summary == {"objective": "monitor risk"}


def test_context_summary_object_passes_through_unchanged():
    raw = json.dumps({
        "context_summary": {"objective": "monitor risk", "n_kpis": 3},
        "kpi_chart_mapping": [],
        "layout": {},
        "styling": {},
        "interactions": [],
        "rationales": [],
    })
    parsed, err = parse_json_safe(raw)
    assert err is None, err
    assert parsed.context_summary == {"objective": "monitor risk", "n_kpis": 3}


def test_object_fields_and_interaction_string_are_coerced():
    raw = json.dumps({
        "context_summary": {},
        "kpi_chart_mapping": [],
        "layout": "single-page executive layout",
        "styling": "light and compact",
        "interactions": "hover tooltips",
        "rationales": [],
    })
    parsed, err = parse_json_safe(raw)
    assert err is None, err
    assert parsed.layout == {"description": "single-page executive layout"}
    assert parsed.styling == {"description": "light and compact"}
    assert parsed.interactions == ["hover tooltips"]
