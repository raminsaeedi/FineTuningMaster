"""Tests for pure identifier-likelihood detection (src/data_pipeline/nvbench_identifier.py).

Policy questions ("is SUM(identifier) allowed") belong to nvbench_quality's
kpi_suitability/chart checkers and are tested there; this file only tests the
identifier decision itself against constructed profile fixtures.
"""

from src.data_pipeline.nvbench_identifier import detect_identifier

_CFG = {
    "identifier": {
        "strong_unique_ratio": 0.98,
        "strong_min_distinct": 20,
        "ambiguous_unique_ratio": 0.5,
        "name_patterns": ["(^|_)id$", "^id(_|$)", "identifier", "(^|_)key$", "(^|_)code$"],
    }
}


def _profile(**overrides):
    base = {
        "stats_available": True, "is_primary_key": False, "is_foreign_key": False,
        "is_unique_index": False,
        "distinct_count": 4, "unique_ratio": 1.0, "resolution": "unique_table_match",
    }
    base.update(overrides)
    return base


def test_primary_key_is_identifier_strong():
    p = _profile(is_primary_key=True, distinct_count=4, unique_ratio=1.0)
    r = detect_identifier(p, "Employee_ID", _CFG)
    assert r["is_identifier"] is True
    assert r["confidence"] == "strong"
    assert "primary_key" in r["evidence"]


def test_high_uniqueness_no_name_match_is_identifier_strong():
    p = _profile(distinct_count=1000, unique_ratio=0.995)
    r = detect_identifier(p, "shop_reference", _CFG)  # name does NOT match id/_id/key/code patterns
    assert r["is_identifier"] is True
    assert r["confidence"] == "strong"


def test_generic_numeric_word_name_not_identifier():
    p = _profile(distinct_count=50, unique_ratio=0.3, is_primary_key=False, is_unique_index=False)
    r = detect_identifier(p, "number_of_employees", _CFG)
    assert r["is_identifier"] is False
    assert r["confidence"] == "none"


def test_order_quantity_not_identifier():
    p = _profile(distinct_count=30, unique_ratio=0.4)
    r = detect_identifier(p, "order_quantity", _CFG)
    assert r["is_identifier"] is False


def test_account_balance_not_identifier():
    p = _profile(distinct_count=40, unique_ratio=0.6)
    r = detect_identifier(p, "account_balance", _CFG)
    assert r["is_identifier"] is False


def test_name_only_moderate_uniqueness_is_ambiguous():
    p = _profile(distinct_count=10, unique_ratio=0.6, is_primary_key=False, is_unique_index=False)
    r = detect_identifier(p, "Shop_ID", _CFG)
    assert r["is_identifier"] is True
    assert r["confidence"] == "ambiguous"


def test_name_only_low_uniqueness_not_identifier():
    p = _profile(distinct_count=3, unique_ratio=0.2)
    r = detect_identifier(p, "status_code", _CFG)
    assert r["is_identifier"] is False


def test_name_only_no_stats_is_ambiguous():
    p = {"stats_available": False, "is_primary_key": False, "is_unique_index": False,
         "resolution": "field_not_found"}
    r = detect_identifier(p, "customer_id", _CFG)
    assert r["is_identifier"] is True
    assert r["confidence"] == "ambiguous"
    assert "stats_unavailable" in r["evidence"]


def test_unique_index_is_identifier_strong():
    p = _profile(is_unique_index=True, distinct_count=4, unique_ratio=1.0)
    r = detect_identifier(p, "code", _CFG)
    assert r["is_identifier"] is True
    assert r["confidence"] == "strong"


def test_foreign_key_is_identifier_strong_even_when_values_repeat():
    p = _profile(is_foreign_key=True, normalized_dtype="number", distinct_count=4, unique_ratio=0.1)
    r = detect_identifier(p, "department_ref", _CFG)
    assert r["is_identifier"] is True
    assert r["confidence"] == "strong"
    assert "foreign_key" in r["evidence"]


def test_numeric_entity_id_name_is_strong_but_numeric_count_is_not():
    repeated_id = _profile(normalized_dtype="number", distinct_count=12, unique_ratio=0.11)
    assert detect_identifier(repeated_id, "DEPARTMENT_ID", _CFG)["confidence"] == "strong"

    count_measure = _profile(normalized_dtype="number", distinct_count=12, unique_ratio=0.11)
    result = detect_identifier(count_measure, "number_of_employees", _CFG)
    assert result["is_identifier"] is False


def test_identity_qualified_card_number_is_strong_with_small_near_unique_profile():
    profile = _profile(
        normalized_dtype="number", distinct_count=15, unique_ratio=1.0,
        is_primary_key=False, is_foreign_key=False, is_unique_index=False,
    )
    result = detect_identifier(profile, "card_number", _CFG)
    assert result["is_identifier"] is True
    assert result["confidence"] == "strong"
    assert "entity_number_name" in result["evidence"]


def test_quantity_number_of_rooms_is_not_entity_number():
    profile = _profile(
        normalized_dtype="number", distinct_count=15, unique_ratio=0.95,
        is_primary_key=False, is_foreign_key=False, is_unique_index=False,
    )
    result = detect_identifier(profile, "number_of_rooms", _CFG)
    assert result["is_identifier"] is False


def test_ambiguous_table_resolution_carried_as_evidence():
    p = {"stats_available": False, "is_primary_key": False, "is_unique_index": False,
         "resolution": "ambiguous_table"}
    r = detect_identifier(p, "Name", _CFG)
    assert "field_table_ambiguous" in r["evidence"]
