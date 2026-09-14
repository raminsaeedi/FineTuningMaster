import copy
import json
from collections import Counter

import pytest

from src.evaluation.human import open_study
from src.evaluation.human.open_analysis import validate_and_summarize
from src.evaluation.human.rubric import RUBRIC_KEYS


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("items.jsonl", "study_manifest.json", "assignment.json"):
        (source / name).write_text("synthetic fixture", encoding="utf-8")
    items = [dict(item_id=f"item{i}", brief={"users": "Analysts", "goals": ["Compare groups"]},
                  outputs={m: {"parsed": {"layout": {"grid": "<script>bad</script>"}}} for m in "ABCD"}) for i in range(8)]
    manifest = dict(model="qwen3_8_27b", dataset="dashboard_v4", methods=list("ABCD"), seed=42, source_prediction_hashes={m: m for m in "ABCD"})
    monkeypatch.setattr(open_study, "load_study", lambda _: (manifest, items, {}))
    verified = []
    monkeypatch.setattr(open_study, "verify_source_predictions_unchanged", lambda *a, **k: verified.append(True))
    args = dict(source_dir=source, out_dir=tmp_path / "open", project_root=tmp_path, spreadsheet_id="a"*40)
    result = open_study.build_open_study(**args)
    assert verified
    key = json.loads((args["out_dir"] / "unblinding_key.json").read_text())
    return args, result, key


def test_packets_blinding_and_freeze(bundle):
    args, result, key = bundle
    assert len(key["units"]) == 32
    assert len({t for p in key["packets"] for t in p}) == 32
    for p in key["packets"]:
        assert Counter(key["units"][t]["method"] for t in p) == dict.fromkeys("ABCD", 2)
        assert len({key["units"][t]["item_id"] for t in p}) == 8
    server = json.loads((args["out_dir"] / "server_config.json").read_text(encoding="utf-8"))
    assert "method" not in json.dumps(server["units"])
    assert "<script>bad" not in json.dumps(server["units"])
    assert "&lt;script&gt;" in json.dumps(server["units"])
    assert open_study.build_open_study(**args)["study_id"] == result["study_id"]
    (args["out_dir"] / "Code.gs").write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        open_study.build_open_study(**args)


def records(manifest, key, n=12):
    people, assignments, ratings = [], [], []
    common = dict(study_id=manifest["study_id"], is_test="FALSE")
    for i in range(n):
        pid = f"p{i}"
        people.append(dict(common, participant_id=pid, nickname="PRIVATE", token_hash="PRIVATE", consent_version=manifest["consent_version"], consent_at="2026-09-14", status="complete"))
        for pos, token in enumerate(key["packets"][i%4], 1):
            assignments.append(dict(common, participant_id=pid, unit_token=token, position=str(pos)))
            ratings.append(dict(common, participant_id=pid, unit_token=token, position=str(pos), rating_id=pid+'|'+token,
                                rubric_version=manifest["rubric_version"], language="de", duration_ms="1000", elapsed_ms="2000", received_at="2026-09-14", **{k: str((i % 5) + 1) for k in RUBRIC_KEYS}))
    return people, assignments, ratings


def test_variable_n_and_no_private_fields(bundle):
    _, manifest, key = bundle
    for n in (0, 1, 6, 12, 17):
        summary, long, coverage = validate_and_summarize(manifest, key, *records(manifest, key, n))
        assert summary["participants"] == n
        assert summary["submitted_pages"] == n*8
        assert len(coverage) == 32
        assert len(long) == n*8*6
        assert summary["coverage_target_met"] == (n >= 12)
        assert "PRIVATE" not in json.dumps([summary, long, coverage])


def test_missing_and_constant_agreement(bundle):
    _, manifest, key = bundle
    p, a, r = records(manifest, key)
    for row in r:
        row.update({k: "3" for k in RUBRIC_KEYS})
    for k in RUBRIC_KEYS:
        r[0][k] = ""
    summary, long, coverage = validate_and_summarize(manifest, key, p, a, r)
    assert summary["missing_judgments"] == 6
    assert summary["coverage_target_met"] is False
    assert all(x["ordinal_alpha"] is None for x in summary["agreement"])
    assert len([x for x in long if x["score"] is None]) == 6


@pytest.mark.parametrize("fault", ["test", "study", "score", "duplicate", "consent", "assignment", "rubric", "timing"])
def test_fail_closed_exports(bundle, fault):
    _, manifest, key = bundle
    p, a, r = records(manifest, key, 1)
    if fault == "test": r[0]["is_test"] = "TRUE"
    if fault == "study": r[0]["study_id"] = "wrong"
    if fault == "score": r[0]["overall_usefulness"] = "NaN"
    if fault == "duplicate": r.append(copy.deepcopy(r[0]))
    if fault == "consent": p[0]["consent_at"] = ""
    if fault == "assignment": a.pop()
    if fault == "rubric": r[0]["rubric_version"] = "old"
    if fault == "timing": r[0]["elapsed_ms"] = "1800001"
    with pytest.raises(ValueError):
        validate_and_summarize(manifest, key, p, a, r)
