"""The synthetic generator must produce correct, deterministic task->chart labels."""

from src.core.schemas import GoldItem
from src.data_pipeline.synth_generator import TASK_CHART, generate_dataset


def test_labels_match_principled_ground_truth():
    items = generate_dataset(n=120, base_seed=42)
    for it in items:
        for m in it["recommendation"]["kpi_chart_mapping"]:
            task = m["task_type"]
            assert task in TASK_CHART, f"unknown task {task}"
            assert m["chart_type"] == TASK_CHART[task][0], (
                f"label mismatch: {task} -> {m['chart_type']}, expected {TASK_CHART[task][0]}"
            )


def test_generated_items_validate_against_schema():
    for it in generate_dataset(n=30, base_seed=7):
        # Should construct without error (enums, columns, rationales all valid).
        GoldItem(item_id="x", brief=it["brief"], recommendation=it["recommendation"])


def test_generation_is_deterministic():
    assert generate_dataset(n=20, base_seed=42) == generate_dataset(n=20, base_seed=42)


def test_growing_set_keeps_earlier_items_stable():
    small = generate_dataset(n=10, base_seed=42)
    large = generate_dataset(n=50, base_seed=42)
    assert large[:10] == small


def test_brief_has_masterplan_fields():
    item = generate_dataset(n=1, base_seed=1)[0]
    for field in ("users", "goals", "kpis", "columns", "constraints"):
        assert field in item["brief"]
    assert len(item["brief"]["kpis"]) >= 3
    assert all("name" in c and "dtype" in c for c in item["brief"]["columns"])
