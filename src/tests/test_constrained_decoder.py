"""Public contract tests for strict constrained JSON decoding."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError


def _strict_payload() -> dict:
    return {
        "context_summary": {},
        "kpi_chart_mapping": [
            {
                "kpi": "Revenue",
                "task_type": "trend",
                "chart_type": "line",
                "alternatives": ["bar"],
                "encoding": {"x": "date", "y": "revenue", "aggregate": None},
            }
        ],
        "layout": {},
        "styling": {},
        "interactions": [],
        "rationales": [],
    }


def test_constrained_schema_requires_all_design_output_sections() -> None:
    from src.core.constants import REQUIRED_KEYS
    from src.inference.decoders import design_output_json_schema

    schema = design_output_json_schema()

    assert set(schema["required"]) == set(REQUIRED_KEYS)


def test_strict_response_requires_at_least_one_kpi_chart_mapping() -> None:
    from src.core.strict_response_schema import StrictDesignOutput

    payload = _strict_payload()
    payload["kpi_chart_mapping"] = []

    with pytest.raises(ValidationError):
        StrictDesignOutput.model_validate(payload)


def test_strict_response_rejects_string_encoding() -> None:
    from src.core.strict_response_schema import StrictDesignOutput

    payload = _strict_payload()
    payload["kpi_chart_mapping"][0]["encoding"] = "x=date, y=revenue"

    with pytest.raises(ValidationError):
        StrictDesignOutput.model_validate(payload)


@pytest.mark.parametrize("missing_field", ["x", "y"])
def test_strict_response_requires_encoding_axes(missing_field: str) -> None:
    from src.core.strict_response_schema import StrictDesignOutput

    payload = _strict_payload()
    del payload["kpi_chart_mapping"][0]["encoding"][missing_field]

    with pytest.raises(ValidationError):
        StrictDesignOutput.model_validate(payload)


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("empty_value", ["", "   "])
def test_strict_response_requires_non_empty_encoding_axes(
    axis: str, empty_value: str
) -> None:
    from src.core.strict_response_schema import StrictDesignOutput

    payload = _strict_payload()
    payload["kpi_chart_mapping"][0]["encoding"][axis] = empty_value

    with pytest.raises(ValidationError):
        StrictDesignOutput.model_validate(payload)


def test_strict_response_accepts_null_aggregate() -> None:
    from src.core.strict_response_schema import StrictDesignOutput

    output = StrictDesignOutput.model_validate(_strict_payload())

    assert output.kpi_chart_mapping[0].encoding.aggregate is None


@pytest.mark.parametrize("aggregate", ["SUM", "AVG", "COUNT", "MIN", "MAX"])
def test_strict_response_accepts_v4_aggregate_allowlist(aggregate: str) -> None:
    from src.core.strict_response_schema import StrictDesignOutput

    payload = _strict_payload()
    payload["kpi_chart_mapping"][0]["encoding"]["aggregate"] = aggregate

    output = StrictDesignOutput.model_validate(payload)

    assert output.kpi_chart_mapping[0].encoding.aggregate == aggregate


@pytest.mark.parametrize("invalid_aggregate", ["MEDIAN", "sum", ""])
def test_strict_response_rejects_aggregate_outside_v4_allowlist(
    invalid_aggregate: str,
) -> None:
    from src.core.strict_response_schema import StrictDesignOutput

    payload = _strict_payload()
    payload["kpi_chart_mapping"][0]["encoding"]["aggregate"] = invalid_aggregate

    with pytest.raises(ValidationError):
        StrictDesignOutput.model_validate(payload)


def test_strict_response_rejects_non_chart_alternative() -> None:
    from src.core.strict_response_schema import StrictDesignOutput

    payload = _strict_payload()
    payload["kpi_chart_mapping"][0]["alternatives"] = ["Revenue"]

    with pytest.raises(ValidationError):
        StrictDesignOutput.model_validate(payload)


@pytest.mark.parametrize("empty_kpi", ["", "   "])
def test_strict_response_requires_non_empty_kpi(empty_kpi: str) -> None:
    from src.core.strict_response_schema import StrictDesignOutput

    payload = _strict_payload()
    payload["kpi_chart_mapping"][0]["kpi"] = empty_kpi

    with pytest.raises(ValidationError):
        StrictDesignOutput.model_validate(payload)


@pytest.mark.parametrize("level", ["output", "mapping", "encoding"])
def test_strict_response_forbids_extra_fields(level: str) -> None:
    from src.core.strict_response_schema import StrictDesignOutput

    payload = _strict_payload()
    if level == "output":
        target = payload
    elif level == "mapping":
        target = payload["kpi_chart_mapping"][0]
    else:
        target = payload["kpi_chart_mapping"][0]["encoding"]
    target["unexpected"] = True

    with pytest.raises(ValidationError):
        StrictDesignOutput.model_validate(payload)


def test_strict_response_forbids_extra_rationale_fields() -> None:
    from src.core.strict_response_schema import StrictDesignOutput

    payload = _strict_payload()
    payload["rationales"] = [
        {
            "claim": "A line chart shows change.",
            "principle": "temporal order",
            "note": "extra",
        }
    ]

    with pytest.raises(ValidationError):
        StrictDesignOutput.model_validate(payload)


def test_strict_response_keeps_frozen_v4_object_interactions_compatible() -> None:
    from src.core.strict_response_schema import StrictDesignOutput

    payload = _strict_payload()
    payload["interactions"] = [
        {"type": "tooltip", "fields": ["date", "revenue"]}
    ]

    output = StrictDesignOutput.model_validate(payload)

    assert output.interactions == payload["interactions"]


def test_decoder_uses_outlines_13_api_and_max_new_tokens(monkeypatch) -> None:
    from src.core.strict_response_schema import StrictDesignOutput
    from src.inference.decoders import ConstrainedDecoder

    model = object()
    tokenizer = object()
    wrapped_model = object()

    def from_transformers(received_model, received_tokenizer):
        assert received_model is model
        assert received_tokenizer is tokenizer
        return wrapped_model

    class FakeGenerator:
        def __init__(self, received_model, output_type):
            assert received_model is wrapped_model
            assert output_type is StrictDesignOutput

        def __call__(self, prompt, *, max_new_tokens):
            assert prompt == "dashboard prompt"
            assert max_new_tokens == 77
            return StrictDesignOutput.model_validate(_strict_payload())

    fake_outlines = SimpleNamespace(
        from_transformers=from_transformers,
        Generator=FakeGenerator,
    )
    monkeypatch.setitem(sys.modules, "outlines", fake_outlines)

    decoder = ConstrainedDecoder(max_new_tokens=77)
    decoder.setup(model, tokenizer)

    assert json.loads(decoder.generate("dashboard prompt")) == _strict_payload()
