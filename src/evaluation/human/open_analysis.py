"""Validate private Sheet CSVs and create pseudonymous descriptive exports.

This deliberately does not reuse the fixed-six-rater inferential pipeline.
Missing judgments are not zero or neutral scores. No automatic significance
tests treat the 96 pages (or their six dimensions) as independent samples.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, median

from .rubric import RUBRIC_KEYS


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_and_summarize(manifest: dict, key: dict, participants: list[dict],
                           assignments: list[dict], ratings: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    if key["study_id"] != manifest["study_id"] or key["rubric_version"] != manifest["rubric_version"]:
        raise ValueError("Manifest/key version mismatch")
    for row in participants + assignments + ratings:
        if row.get("study_id") != manifest["study_id"]:
            raise ValueError("Mixed or wrong study IDs")
        if str(row.get("is_test", "")).lower() not in ("false", "0"):
            raise ValueError("Demo/test or unlabelled rows cannot enter research analysis")
    people = {p["participant_id"]: p for p in participants}
    if len(people) != len(participants):
        raise ValueError("Duplicate participant IDs")
    for p in participants:
        if p.get("consent_version") != manifest["consent_version"] or not p.get("consent_at"):
            raise ValueError("Missing/version-mismatched consent")
    assigned, by_person = {}, defaultdict(list)
    for a in assignments:
        pair = (a["participant_id"], a["unit_token"])
        if pair in assigned or pair[0] not in people or pair[1] not in key["units"]:
            raise ValueError("Duplicate or unknown assignment")
        assigned[pair] = a
        by_person[pair[0]].append(a)
    packet_sets = {frozenset(p) for p in key["packets"]}
    for pid in people:
        tasks = by_person[pid]
        if (frozenset(a["unit_token"] for a in tasks) not in packet_sets or len(tasks) != 8
                or sorted(int(a["position"]) for a in tasks) != list(range(1, 9))):
            raise ValueError("Incomplete/invalid assignment; export again after registration completes")
    long_rows, seen, per_unit, pages_by_person = [], set(), defaultdict(list), Counter()
    for r in ratings:
        pair = (r["participant_id"], r["unit_token"])
        if pair in seen:
            raise ValueError("Duplicate rating page; reconcile raw export, do not silently drop")
        seen.add(pair)
        if pair not in assigned or int(r["position"]) != int(assigned[pair]["position"]):
            raise ValueError("Rating not assigned or order mismatch")
        if r.get("rubric_version") != manifest["rubric_version"]:
            raise ValueError("Mixed rating rubrics")
        if r.get("rating_id") != "|".join(pair):
            raise ValueError("Rating ID mismatch")
        if r.get("language") not in ("de", "en"):
            raise ValueError("Invalid rating language")
        duration, elapsed = float(r["duration_ms"]), float(r["elapsed_ms"])
        if not (0 <= duration <= elapsed <= 1800000):
            raise ValueError("Invalid timing")
        record = {"participant_id": pair[0], "unit_token": pair[1], **key["units"][pair[1]],
                  "position": int(r["position"]), "language": r["language"],
                  "duration_ms": duration, "elapsed_ms": elapsed, "received_at": r["received_at"]}
        for dimension in RUBRIC_KEYS:
            raw = r.get(dimension)
            if raw in (None, ""):
                score = None
            else:
                score = float(raw)
                if score not in (1, 2, 3, 4, 5):
                    raise ValueError(f"Invalid score in {dimension}")
                score = int(score)
            record[dimension] = score
            long_rows.append({k: record[k] for k in ("participant_id", "unit_token", "item_id", "method", "position", "language", "duration_ms", "elapsed_ms", "received_at")} | {"dimension": dimension, "score": score})
        per_unit[pair[1]].append(record)
        pages_by_person[pair[0]] += 1
    coverage = []
    for token, unit in key["units"].items():
        unit_rows = per_unit[token]
        result = {"unit_token": token, **unit, "submitted_pages": len(unit_rows),
                  "usable_pages": sum(any(r[k] is not None for k in RUBRIC_KEYS) for r in unit_rows)}
        result.update({"n_" + k: sum(r[k] is not None for r in unit_rows) for k in RUBRIC_KEYS})
        coverage.append(result)
    descriptions = []
    unit_means = {}
    for method in "ABCD":
        for dimension in RUBRIC_KEYS:
            scores, item_means = [], []
            for token, unit in key["units"].items():
                if unit["method"] != method:
                    continue
                values = [r[dimension] for r in per_unit[token] if r[dimension] is not None]
                unit_means[(unit["item_id"], method, dimension)] = mean(values) if values else None
                if values:
                    scores.extend(values)
                    item_means.append(mean(values))
            descriptions.append(dict(method=method, dimension=dimension, n_valid=len(scores),
                                     n_briefs=len(item_means), score_counts={str(i): scores.count(i) for i in range(1, 6)},
                                     median_score=median(scores) if scores else None,
                                     equal_brief_mean=mean(item_means) if item_means else None))
    contrasts = []
    for a, b in combinations("ABCD", 2):
        for dimension in RUBRIC_KEYS:
            diffs = []
            for item_id in manifest["item_ids"]:
                left, right = unit_means.get((item_id, a, dimension)), unit_means.get((item_id, b, dimension))
                if left is not None and right is not None:
                    diffs.append(left - right)
            contrasts.append(dict(contrast=a + " - " + b, dimension=dimension, paired_briefs=len(diffs),
                                  mean_difference=mean(diffs) if diffs else None, inference="descriptive only"))
    # Agreement only over multiply-rated output units; constant data do not
    # demonstrate reliability. Use the established ordinal alpha implementation.
    from .irr import krippendorff_alpha
    agreement = []
    for dimension in RUBRIC_KEYS:
        groups = [[r[dimension] for r in per_unit[token] if r[dimension] is not None] for token in key["units"]]
        groups = [g for g in groups if len(g) >= 2]
        distinct = {s for g in groups for s in g}
        alpha = krippendorff_alpha(groups, level="ordinal") if groups and len(distinct) > 1 else None
        agreement.append(dict(dimension=dimension, multiply_rated_units=len(groups), ordinal_alpha=alpha))
    summary = dict(study_id=manifest["study_id"], analysis_type="exploratory_descriptive",
                   participants=len(people), completed_sessions=sum(n == 8 for n in pages_by_person.values()),
                   participants_with_no_ratings=sum(p not in pages_by_person for p in people),
                   submitted_pages=len(ratings), expected_output_units=len(key["units"]),
                   coverage_target_met=all(c["usable_pages"] >= 3 for c in coverage),
                   missing_judgments=sum(r["score"] is None for r in long_rows),
                   participant_statuses=dict(Counter(p.get("status", "unknown") for p in participants)),
                   rating_languages=dict(Counter(r["language"] for r in ratings)),
                   descriptions=descriptions, paired_contrasts=contrasts, agreement=agreement,
                   limitations=["Convenience recruitment, identity not verified; IDs are not proof of unique humans.",
                                "Eight briefs and shared raters: no independent-page significance test.",
                                "Timing is browser-reported, excludes pauses, hidden time and network waiting.",
                                "Coverage target is not power or precision assurance; examine each dimension's missingness.",
                                "Do not combine with earlier rubric/protocol versions."])
    return summary, long_rows, coverage


def analyze_exports(*, study_dir: Path, participant_csv: Path, assignment_csv: Path,
                    rating_csv: Path, out_dir: Path) -> dict:
    from .open_study import digest
    manifest = json.loads((study_dir / "open_manifest.json").read_text(encoding="utf-8"))
    key_path = study_dir / "unblinding_key.json"
    if digest(key_path) != manifest["artifact_hashes"]["unblinding_key.json"]:
        raise ValueError("Unblinding key was modified after study creation")
    key = json.loads(key_path.read_text(encoding="utf-8"))
    summary, long_rows, coverage = validate_and_summarize(
        manifest, key, read_csv(participant_csv), read_csv(assignment_csv), read_csv(rating_csv))
    summary["input_sha256"] = {name: digest(path) for name, path in
                               (("participants", participant_csv), ("assignments", assignment_csv), ("ratings", rating_csv))}
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("Use an empty analysis output directory to preserve previous snapshots")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, rows, fields in (("ratings_long.csv", long_rows, ["participant_id", "unit_token", "item_id", "method", "position", "language", "duration_ms", "elapsed_ms", "received_at", "dimension", "score"]),
                               ("coverage.csv", coverage, list(coverage[0]))):
        with (out_dir / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return summary
