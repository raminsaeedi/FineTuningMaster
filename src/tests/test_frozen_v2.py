"""Freeze, validation and not-for-training safeguards for dataset v2."""

import csv
from copy import deepcopy
from pathlib import Path

from src.core.schemas import ChartType, DashboardBrief, DesignOutput, GoldItem, TaskType
from src.data_pipeline.frozen_v2 import (
    STORED_SPLIT_TO_FILE,
    bucket_by_split,
    dedupe_by_fingerprint,
    gold_to_record,
)
from src.data_pipeline.frozen_validation import (
    distributions,
    find_duplicate_ids,
    leakage_report,
    read_jsonl_strict,
    sha256_of_file,
    validate_record,
)
from src.data_pipeline.splits import assign_split
from src.data_pipeline.synth_generator_v2 import generate_candidates
from src.utils.io import read_yaml, write_jsonl

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _gold(candidate) -> GoldItem:
    return GoldItem(
        item_id=candidate["item_id"],
        brief=DashboardBrief(**candidate["brief"]),
        recommendation=DesignOutput(**candidate["recommendation"]),
        split=assign_split(candidate["item_id"]),
    )


# --- split compatibility + freeze helpers ---------------------------------

def test_split_values_stay_legacy_literals():
    assert set(STORED_SPLIT_TO_FILE) == {"train", "val", "test"}
    assert STORED_SPLIT_TO_FILE["test"] == "internal_test.jsonl"
    for it in generate_candidates(n=40, seed=42):
        assert assign_split(it["item_id"]) in {"train", "val", "test"}


def test_bucket_and_serialise_roundtrip():
    items = [_gold(c) for c in generate_candidates(n=30, seed=42)]
    buckets = bucket_by_split(items)
    assert sum(len(v) for v in buckets.values()) == len(items)
    rec = gold_to_record(items[0])
    # enums serialised to plain strings
    for m in rec["recommendation"]["kpi_chart_mapping"]:
        assert isinstance(m["task_type"], str) and isinstance(m["chart_type"], str)


def test_dedupe_by_fingerprint_drops_content_duplicate():
    items = [_gold(c) for c in generate_candidates(n=5, seed=42)]
    dup = deepcopy(items[0])
    dup.item_id = "v2_different_id"
    kept, dropped = dedupe_by_fingerprint(items + [dup])
    assert len(kept) == len(items)
    assert dropped and dropped[0]["reason"] == "duplicate_fingerprint"


# --- validation checks -----------------------------------------------------

def test_validate_record_accepts_generated():
    for c in generate_candidates(n=10, seed=3):
        assert validate_record(gold_to_record(_gold(c))) == []


def test_validate_record_rejects_bad_enum():
    c = generate_candidates(n=1, seed=1)[0]
    rec = gold_to_record(_gold(c))
    rec["recommendation"]["kpi_chart_mapping"][0]["chart_type"] = "not_a_chart"
    problems = validate_record(rec)
    assert any("chart_type" in p for p in problems)


def test_validate_record_rejects_empty_required_field():
    c = generate_candidates(n=1, seed=1)[0]
    rec = gold_to_record(_gold(c))
    rec["brief"]["kpis"] = []
    problems = validate_record(rec)
    assert any("kpis" in p for p in problems)


def test_read_jsonl_strict_flags_malformed(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"ok": 1}\nNOT JSON\n', encoding="utf-8")
    records, errors = read_jsonl_strict(p)
    assert len(records) == 1 and len(errors) == 1


def test_find_duplicate_ids():
    recs = [{"item_id": "a"}, {"item_id": "a"}, {"item_id": "b"}]
    assert find_duplicate_ids(recs) == ["a"]


def test_distributions_counts_domain_and_charts():
    recs = [gold_to_record(_gold(c)) for c in generate_candidates(n=20, seed=42)]
    dist = distributions(recs)
    assert dist["domain"] and dist["task_type"] and dist["chart_type"]
    for chart in dist["chart_type"]:
        assert isinstance(chart, str)


# --- leakage + hashing -----------------------------------------------------

def test_leakage_report_detects_and_clears():
    cands = [gold_to_record(_gold(c)) for c in generate_candidates(n=10, seed=42)]
    train_val, eval_recs = cands[:6], cands[6:]
    assert leakage_report(train_val, eval_recs) == {
        "item_id_overlap": [], "fingerprint_overlap": []
    }
    # Injecting a train item into the eval set must be detected.
    leaked = leakage_report(train_val, eval_recs + [train_val[0]])
    assert leaked["item_id_overlap"] and leaked["fingerprint_overlap"]


def test_sha256_is_stable(tmp_path):
    p = tmp_path / "f.jsonl"
    write_jsonl([{"a": 1}], p)
    assert sha256_of_file(p) == sha256_of_file(p)


# --- not-for-training safeguard -------------------------------------------

def test_dashboard_v2_config_never_trains_eval_files():
    cfg = read_yaml(_REPO_ROOT / "src" / "config" / "data" / "dashboard_v2.yaml")
    trainable = {cfg["train_file"], cfg["val_file"]}
    never = set(cfg["not_for_training"])
    # trainable and never-train sets must be disjoint
    assert trainable.isdisjoint(never)
    # the diagnostic/eval files must all be guarded
    for key in ("test_file", "l1_effectiveness_file", "real_briefs_file"):
        assert cfg[key] in never


def test_l1_effectiveness_csv_uses_valid_enums():
    path = _REPO_ROOT / "data" / "eval" / "l1_chart_effectiveness_v1.csv"
    tasks = {t.value for t in TaskType}
    charts = {c.value for c in ChartType}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 15  # planned 30-50; template seeds a covered subset
    for row in rows:
        assert row["task_type"] in tasks, row["task_type"]
        for chart in row["effective_charts"].split("|"):
            assert chart in charts, chart


def test_real_briefs_v1_have_no_chart_labels():
    path = _REPO_ROOT / "data" / "eval" / "real_briefs_v1.jsonl"
    records, errors = read_jsonl_strict(path)
    assert not errors and records
    for r in records:
        # external briefs only — no recommendation / chart gold, marked never-train
        assert "recommendation" not in r
        assert r.get("extra", {}).get("not_for_training") is True
