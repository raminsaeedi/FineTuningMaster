"""Tests for the deterministic input perturbations (robustness study)."""

from __future__ import annotations

from src.data.perturbations import drop_info_brief, paraphrase_brief, paraphrase_text

_BRIEF = {
    "item_id": "item_abc123",
    "users": "Show key metrics to managers",
    "goals": ["Track revenue over time", "Compare regions"],
    "kpis": ["revenue", "growth", "churn"],
    "columns": [{"name": "date", "dtype": "datetime"}, {"name": "amount", "dtype": "float"}],
    "constraints": "Display on mobile",
}


def test_paraphrase_is_deterministic():
    assert paraphrase_text("Show the key metrics") == paraphrase_text("Show the key metrics")


def test_paraphrase_changes_text_but_preserves_structure():
    out = paraphrase_brief(_BRIEF)
    assert out["users"] != _BRIEF["users"]          # reworded
    assert "display" in out["users"].lower()         # show -> display
    assert len(out["goals"]) == len(_BRIEF["goals"])  # structure intact
    assert len(out["kpis"]) == len(_BRIEF["kpis"])
    assert _BRIEF["users"] == "Show key metrics to managers"  # original untouched (deepcopy)


def test_drop_info_reduces_information():
    out = drop_info_brief(_BRIEF)
    assert out["constraints"] is None
    assert len(out["kpis"]) == len(_BRIEF["kpis"]) - 1
    assert len(out["columns"]) == len(_BRIEF["columns"]) - 1


def test_drop_info_keeps_at_least_one_kpi():
    out = drop_info_brief({"kpis": ["only"], "columns": [{"name": "a", "dtype": "int"}]})
    assert out["kpis"] == ["only"]
    assert len(out["columns"]) == 1
