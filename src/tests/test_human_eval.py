"""Tests for the human-evaluation system (assignment, IRR, eval-set build)."""

from src.evaluation.human.assignment import build_assignment, build_eval_items
from src.evaluation.human.irr import krippendorff_alpha


def test_assignment_is_balanced_and_blind_safe():
    items = [f"item_{i}" for i in range(10)]
    methods = ["prompt_only", "rag", "ft", "ft_rag"]
    raters = [f"rater_{i:02d}" for i in range(1, 7)]
    k = 3
    a = build_assignment(items, methods, raters, ratings_per_output=k, seed=1)

    # Every unit rated by exactly k distinct raters.
    unit_raters = {}
    for rater, tasks in a["raters"].items():
        for t in tasks:
            unit_raters.setdefault(t["unit_id"], set()).add(rater)
    assert len(unit_raters) == len(items) * len(methods)
    assert all(len(rs) == k for rs in unit_raters.values())

    # Load is balanced: spread of per-rater counts is small.
    loads = [len(v) for v in a["raters"].values()]
    assert max(loads) - min(loads) <= len(methods)


def test_assignment_rejects_too_many_ratings():
    import pytest

    with pytest.raises(ValueError):
        build_assignment(["i1"], ["m1"], ["r1", "r2"], ratings_per_output=3)


def test_krippendorff_perfect_and_disagreement():
    assert krippendorff_alpha([[3, 3], [5, 5], [2, 2]]) == 1.0
    # Systematic disagreement -> alpha well below 0.
    bad = krippendorff_alpha([[1, 5], [1, 5], [1, 5], [1, 5]])
    assert bad is not None and bad < 0.0
    # No ratable units.
    assert krippendorff_alpha([[3]]) is None


def test_krippendorff_partial_agreement_in_range():
    units = [[4, 4, 5], [3, 3, 2], [5, 5, 5], [2, 3, 2]]
    a = krippendorff_alpha(units, level="ordinal")
    assert a is not None and 0.0 < a <= 1.0


def test_build_eval_items_intersection_and_limit():
    preds = {
        "prompt_only": [{"item_id": "a", "raw_text": "x", "parsed": None},
                        {"item_id": "b", "raw_text": "y", "parsed": {"k": 1}}],
        "ft": [{"item_id": "a", "raw_text": "z", "parsed": {"k": 2}},
               {"item_id": "c", "raw_text": "w", "parsed": None}],
    }
    briefs = {"a": {"users": "U"}, "b": {"users": "V"}, "c": {"users": "W"}}
    items = build_eval_items(preds, briefs, n_items=10, seed=0)
    # Only 'a' is common to both methods.
    assert len(items) == 1 and items[0]["item_id"] == "a"
    assert set(items[0]["outputs"]) == {"prompt_only", "ft"}
