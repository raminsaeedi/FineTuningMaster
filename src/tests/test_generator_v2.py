"""The v2 source-conditioned generator: sample mode, determinism, provenance."""

import pytest

from src.core.schemas import GoldItem
from src.data_pipeline.synth_generator import TASK_CHART
from src.data_pipeline.synth_generator_v2 import (
    GENERATOR_VERSION,
    generate_candidates,
)


def test_sample_mode_produces_valid_gold_items():
    for it in generate_candidates(n=24, seed=42, mode="sample"):
        # Constructs without error → enums, columns, rationales all valid.
        GoldItem(item_id=it["item_id"], brief=it["brief"], recommendation=it["recommendation"])


def test_labels_match_principled_mapping():
    for it in generate_candidates(n=24, seed=42):
        for m in it["recommendation"]["kpi_chart_mapping"]:
            assert m["task_type"] in TASK_CHART
            assert m["chart_type"] == TASK_CHART[m["task_type"]][0]


def test_generation_is_deterministic():
    assert generate_candidates(n=20, seed=42) == generate_candidates(n=20, seed=42)


def test_growing_set_keeps_earlier_items_stable():
    small = generate_candidates(n=10, seed=42)
    large = generate_candidates(n=40, seed=42)
    assert large[:10] == small


def test_provenance_and_id_prefix():
    for it in generate_candidates(n=10, seed=7):
        assert it["item_id"].startswith("v2_")
        extra = it["brief"]["extra"]
        assert extra["source_id"].startswith("tmpl_")
        assert extra["source_ref"]
        assert extra["generator_version"] == GENERATOR_VERSION
        assert extra["domain"]


def test_required_brief_fields_non_empty():
    it = generate_candidates(n=1, seed=1)[0]
    for field in ("users", "goals", "kpis", "columns"):
        assert it["brief"][field]
    assert len(it["brief"]["kpis"]) >= 3


def test_api_mode_not_implemented():
    with pytest.raises(NotImplementedError):
        generate_candidates(n=2, mode="api")


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        generate_candidates(n=2, mode="nonsense")
