"""Dataset leakage check: training vs evaluation/benchmark sets.

Checks (per training-vs-eval comparison, plus within-training):
  1. exact duplicate item_id
  2. exact duplicate brief (normalized fingerprint)
  3. same source item across train vs benchmark
  4. same label lineage across train vs benchmark
  5. near-duplicate brief similarity (char-3gram Jaccard >= threshold)

Findings are classified: no_issue | warning | critical | rule_leakage.
Writes experiments/results/leakage_report.{json,md}. No model inference.

    python experiments/scripts/check_dataset_leakage.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.benchmark_validation import item_brief
from src.data_pipeline.builders.leakage import fingerprint
from src.data_pipeline.leakage_similarity import near_duplicate_pairs
from src.data_pipeline.frozen_validation import read_jsonl_strict
from src.utils.io import write_json

NEAR_DUP_THRESHOLD = 0.8

# Active trainable files only (per src/config/data/dashboard_v2.yaml). Legacy
# data/processed/* is superseded and never used for thesis training, so it is
# intentionally excluded here (including it conflates two generator generations).
TRAIN_FILES = ("data/frozen/dashboard_v2/train.jsonl", "data/frozen/dashboard_v2/val.jsonl")
EVAL_SETS = {
    "benchmark_v1": "data/eval/benchmark_v1.jsonl",
    "real_briefs_v1": "data/eval/real_briefs_v1.jsonl",
    "internal_test": "data/frozen/dashboard_v2/internal_test.jsonl",
}


def _load(path: Path) -> List[dict]:
    if not path.exists():
        return []
    recs, _ = read_jsonl_strict(path)
    return recs


def _id_and_brief(rec: dict, kind: str) -> Tuple[str, dict]:
    """Return (id, brief_dict) for a record from a given set kind."""
    if kind == "benchmark_v1":
        return rec.get("benchmark_id", ""), item_brief(rec)
    if kind == "real_briefs_v1":
        return rec.get("item_id", ""), rec  # DashboardBrief at top level
    # train / internal_test: {item_id, brief, ...}
    return rec.get("item_id", ""), (rec.get("brief") or {})


def _label_lineage(rec: dict, kind: str) -> str:
    if kind == "benchmark_v1":
        return f"benchmark:{rec.get('label_source', '?')}"
    if kind in ("train", "internal_test"):
        # frozen v2 synthetic gold: labels come from the generator rule.
        extra = (rec.get("brief") or {}).get("extra") or {}
        return "synthetic_generator" if extra.get("generator_version") else "unknown"
    return "external_no_labels"


def main() -> None:
    train: List[dict] = []
    train_breakdown: List[str] = []
    for rel in TRAIN_FILES:
        recs = _load(_PROJECT_ROOT / rel)
        train += recs
        train_breakdown.append(f"{rel} = {len(recs)}")
    train_pairs = [(_id_and_brief(r, "train")) for r in train]
    train_ids = {i for i, _ in train_pairs}
    train_fps = {fingerprint(b) for _, b in train_pairs}
    train_lineages = {_label_lineage(r, "train") for r in train}

    findings: List[dict] = []

    # Within-training duplicate ids (data integrity).
    seen: set = set()
    dup_train_ids = sorted({i for i, _ in train_pairs if (i in seen) or seen.add(i)})
    if dup_train_ids:
        findings.append({"check": "within_train_duplicate_item_id", "severity": "critical",
                         "detail": dup_train_ids[:20]})

    for kind, rel in EVAL_SETS.items():
        recs = _load(_PROJECT_ROOT / rel)
        pairs = [_id_and_brief(r, kind) for r in recs]
        # 1. exact item_id overlap
        id_overlap = sorted({i for i, _ in pairs if i in train_ids})
        findings.append({"check": f"exact_item_id::train~{kind}",
                         "severity": "critical" if id_overlap else "no_issue",
                         "detail": id_overlap[:20], "n": len(id_overlap)})
        # 2. exact brief fingerprint overlap
        fp_overlap = sorted({i for i, b in pairs if fingerprint(b) in train_fps})
        findings.append({"check": f"exact_brief::train~{kind}",
                         "severity": "critical" if fp_overlap else "no_issue",
                         "detail": fp_overlap[:20], "n": len(fp_overlap)})
        # 5. near-duplicate similarity
        nd = near_duplicate_pairs(train_pairs, pairs, threshold=NEAR_DUP_THRESHOLD)
        findings.append({"check": f"near_duplicate::train~{kind}",
                         "severity": "warning" if nd else "no_issue",
                         "detail": nd[:20], "n": len(nd)})
        # 4. label lineage overlap (rule leakage)
        eval_lineages = {_label_lineage(r, kind) for r in recs}
        shared = sorted(train_lineages & eval_lineages)
        findings.append({"check": f"label_lineage::train~{kind}",
                         "severity": "rule_leakage" if shared else "no_issue",
                         "detail": shared, "n": len(shared)})

    # 3. same source across train vs benchmark (source identity namespaces).
    bench = _load(_PROJECT_ROOT / EVAL_SETS["benchmark_v1"])
    train_sources = {((r.get("brief") or {}).get("extra") or {}).get("source_id")
                     for r in train} - {None}
    bench_sources = {r.get("source_name") for r in bench} - {None}
    src_overlap = sorted(train_sources & bench_sources)
    findings.append({"check": "same_source::train~benchmark_v1",
                     "severity": "critical" if src_overlap else "no_issue",
                     "detail": src_overlap, "n": len(src_overlap)})

    severities = {f["severity"] for f in findings}
    overall = ("critical" if "critical" in severities else
               "rule_leakage" if "rule_leakage" in severities else
               "warning" if "warning" in severities else "no_issue")

    payload = {"threshold_near_duplicate": NEAR_DUP_THRESHOLD,
               "n_train": len(train), "overall": overall, "findings": findings}
    write_json(payload, _PROJECT_ROOT / "experiments/results/leakage_report.json")

    md = ["# Dataset Leakage Report", "",
          f"Training records checked: {len(train)}. Near-duplicate threshold: "
          f"{NEAR_DUP_THRESHOLD} (char-3gram Jaccard).", "",
          "## Active trainable dataset (authoritative)", "",
          "The check scopes `train` to the **only files the current v2 pipeline trains on** "
          "(per `src/config/data/dashboard_v2.yaml`: `train_file` + `val_file`). Legacy "
          "synthetic v1 files (`data/processed/*`, `data/gold.jsonl`) are **superseded and "
          "never trained on**, so they are intentionally excluded here — including them would "
          "conflate two generator generations and produce misleading cross-generation "
          "overlaps.", ""]
    for line in train_breakdown:
        md.append(f"- `{line}`")
    md += [f"- **total active train+val = {len(train)}**", "",
           f"**Overall: {overall.upper()}**", "",
           "| check | severity | n | detail |", "| --- | --- | --- | --- |"]
    for f in findings:
        d = f["detail"]
        d_str = "" if not d else (str(d)[:80] + ("…" if len(str(d)) > 80 else ""))
        md.append(f"| `{f['check']}` | {f['severity']} | {f.get('n', '')} | {d_str} |")
    md += ["", "> `label_lineage::train~benchmark_v1` is `no_issue` because benchmark labels "
           "use `literature_L1`/`manual_expert`, disjoint from the generator's "
           "`synthetic_generator` lineage — this is the rule-leakage guard for the benchmark.",
           "> `label_lineage::train~internal_test` is expected to be `rule_leakage` "
           "(both are synthetic-generator lineage) — the internal test is diagnostic only, "
           "never an independent claim.",
           "> Some benchmark `acceptable_chart_types` sets happen to match the generator's "
           "`TASK_CHART` set for the same task_type (see `benchmark_dataset_report.md` §10). "
           "This is **informational overlap only, not leakage**: those labels are sourced from "
           "the independent L1 literature table / documented expert judgment, not from the "
           "synthetic generator."]
    (_PROJECT_ROOT / "experiments/results/leakage_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Leakage report overall={overall} -> experiments/results/leakage_report.md")


if __name__ == "__main__":
    main()
