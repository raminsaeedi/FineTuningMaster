"""Build the hybrid dataset: enriched nvBench + validated synthetic (train/val only).

Inputs are the reconciled nvBench accepted sets from the full enrichment run and
the frozen synthetic v2 splits. The nvBench test split, the human-evaluation items
and the literature-based L1 gold are never read as inputs; they are loaded only as
held-out reference sets for the leakage report.

Nothing is frozen here and no model is trained.

Usage:
    python experiments/scripts/build_hybrid_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.enrichment import ENRICHABLE_FIELDS  # noqa: E402
from src.data_pipeline.enrichment_full import write_json_atomic, write_jsonl_atomic  # noqa: E402
from src.data_pipeline.hybrid_dataset import (  # noqa: E402
    HYBRID_SPEC_VERSION,
    SOURCE_NVBENCH,
    SOURCE_SYNTHETIC,
    cross_source_duplicates,
    distribution_rows,
    has_provenance,
    holdout_collisions,
    required_field_problems,
    schema_problems,
    split_overlap,
    summarize,
    tag_source,
)
from src.utils.io import read_jsonl  # noqa: E402

DEFAULT_ENRICHMENT_DIR = "data/staging/enrichment/full_train_val_v1"
DEFAULT_SYNTHETIC_DIR = "data/frozen/dashboard_v2"
DEFAULT_NVBENCH_DIR = "data/staging/dashboard_v3/nvbench_large_v2"
DEFAULT_OUT_DIR = "data/staging/dashboard_v3/hybrid_dataset_v1"
HUMAN_EVAL_FILE = "data/staging/dashboard_v3/nvbench_large_v2/reports/human_eval_test_items_40.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build hybrid train/val dataset")
    p.add_argument("--enrichment-dir", default=DEFAULT_ENRICHMENT_DIR)
    p.add_argument("--synthetic-dir", default=DEFAULT_SYNTHETIC_DIR)
    p.add_argument("--nvbench-dir", default=DEFAULT_NVBENCH_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return p.parse_args()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _PROJECT_ROOT / candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:  # noqa: C901 - linear build script
    args = parse_args()
    out_dir = _resolve(args.out_dir)
    enrichment_dir = _resolve(args.enrichment_dir)
    synthetic_dir = _resolve(args.synthetic_dir)
    nvbench_dir = _resolve(args.nvbench_dir)

    print("=" * 64)
    print("HYBRID DATASET v1 - BUILD")
    print("=" * 64)

    # ---- inputs -------------------------------------------------------
    nvbench = {
        split: [tag_source(r, SOURCE_NVBENCH)
                for r in read_jsonl(enrichment_dir / "final" / f"{split}_accepted_final.jsonl")]
        for split in ("train", "val")
    }
    synthetic_raw = {
        split: read_jsonl(synthetic_dir / f"{split}.jsonl") for split in ("train", "val")
    }
    permanently_rejected = read_jsonl(enrichment_dir / "final" / "permanently_rejected.jsonl") \
        if (enrichment_dir / "final" / "permanently_rejected.jsonl").exists() else []

    print(f"  nvbench accepted : train {len(nvbench['train'])} | val {len(nvbench['val'])}")
    print(f"  synthetic source : train {len(synthetic_raw['train'])} | val {len(synthetic_raw['val'])}"
          f"  ({synthetic_dir.name})")

    # ---- cross-source deduplication -----------------------------------
    synthetic: Dict[str, List[Dict[str, Any]]] = {}
    dropped: List[Dict[str, Any]] = []
    for split in ("train", "val"):
        kept, drops = cross_source_duplicates(nvbench[split], synthetic_raw[split])
        synthetic[split] = [tag_source(r, SOURCE_SYNTHETIC) for r in kept]
        dropped.extend(drops)
    print(f"  dropped synthetic: {len(dropped)} (cross-source duplicates)")

    hybrid = {split: nvbench[split] + synthetic[split] for split in ("train", "val")}

    # ---- held-out reference sets (never inputs) ------------------------
    nvbench_test = read_jsonl(nvbench_dir / "test.jsonl")
    human_eval_path = _resolve(HUMAN_EVAL_FILE)
    human_eval = read_jsonl(human_eval_path) if human_eval_path.exists() else []

    # ---- validation ----------------------------------------------------
    all_records = hybrid["train"] + hybrid["val"]
    ids = [str(r.get("item_id")) for r in all_records]
    duplicate_ids = sorted({i for i in ids if ids.count(i) > 1})
    overlap = split_overlap(hybrid["train"], hybrid["val"])
    test_collisions = holdout_collisions(all_records, nvbench_test)
    human_collisions = holdout_collisions(all_records, human_eval) if human_eval else {
        "shared_item_ids": [], "shared_normalized_goals": [], "shared_source_groups": [],
        "near_duplicates": [], "note": "human-evaluation file not present as JSONL",
    }
    schema_issues = schema_problems(all_records)
    field_issues = required_field_problems(all_records)
    missing_provenance = [str(r.get("item_id")) for r in all_records if not has_provenance(r)]

    checks = {
        "schema_valid": not schema_issues,
        "required_fields_non_empty": not field_issues,
        "item_ids_unique": not duplicate_ids,
        "provenance_and_lineage_present": not missing_provenance,
        "no_test_records_in_hybrid": not any(str(r.get("split")) == "test" for r in all_records),
        "no_test_item_collisions": not test_collisions["shared_item_ids"],
        "no_test_source_group_collisions": not test_collisions["shared_source_groups"],
        "no_test_near_duplicates": not test_collisions["near_duplicates"],
        "no_human_eval_collisions": not human_collisions["shared_item_ids"],
        "no_cross_split_source_group_overlap": not overlap["shared_source_groups"],
        "no_cross_split_duplicate_ids": not overlap["duplicate_item_ids"],
        "splits_preserved": all(str(r.get("split")) == split
                                for split in ("train", "val") for r in hybrid[split]),
        "counts_reconcile": len(nvbench["train"]) + len(nvbench["val"]) + len(permanently_rejected) == 1545,
    }
    failures = [name for name, ok in checks.items() if not ok]
    status = "PASS_HYBRID_DATASET_READY_FOR_FREEZE" if not failures else "HYBRID_VALIDATION_FAIL"

    # ---- outputs -------------------------------------------------------
    write_jsonl_atomic(hybrid["train"], out_dir / "train.jsonl")
    write_jsonl_atomic(hybrid["val"], out_dir / "val.jsonl")
    write_jsonl_atomic(nvbench["train"], out_dir / "nvbench_train_accepted.jsonl")
    write_jsonl_atomic(nvbench["val"], out_dir / "nvbench_val_accepted.jsonl")
    write_jsonl_atomic(synthetic["train"], out_dir / "synthetic_train_accepted.jsonl")
    write_jsonl_atomic(synthetic["val"], out_dir / "synthetic_val_accepted.jsonl")
    write_jsonl_atomic(dropped, out_dir / "dropped_cross_source_duplicates.jsonl")
    write_jsonl_atomic(permanently_rejected, out_dir / "permanently_rejected_enrichment.jsonl")

    with (out_dir / "distribution_report.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "dimension", "value", "records"])
        writer.writeheader()
        for split in ("train", "val"):
            writer.writerows(distribution_rows(hybrid[split], split))

    validation_report = {
        "hybrid_spec_version": HYBRID_SPEC_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checks": checks,
        "failures": failures,
        "schema_problems": schema_issues[:20],
        "required_field_problems": field_issues[:20],
        "duplicate_item_ids": duplicate_ids,
        "records_without_provenance": missing_provenance[:20],
        "counts": {
            "hybrid_train": len(hybrid["train"]),
            "hybrid_val": len(hybrid["val"]),
            "nvbench_train": len(nvbench["train"]),
            "nvbench_val": len(nvbench["val"]),
            "synthetic_train": len(synthetic["train"]),
            "synthetic_val": len(synthetic["val"]),
            "synthetic_dropped": len(dropped),
            "permanently_rejected_enrichment": len(permanently_rejected),
        },
        "summary": {"train": summarize(hybrid["train"]), "val": summarize(hybrid["val"])},
    }
    write_json_atomic(validation_report, out_dir / "validation_report.json")

    leakage = {
        "timestamp_utc": validation_report["timestamp_utc"],
        "against_nvbench_test": test_collisions,
        "against_human_eval_items": human_collisions,
        "cross_split": overlap,
        "held_out_sets_used_as_reference_only": True,
        "l1_gold_included": False,
    }
    write_json_atomic(leakage, out_dir / "leakage_report.json")

    duplicates = {
        "timestamp_utc": validation_report["timestamp_utc"],
        "duplicate_item_ids": duplicate_ids,
        "cross_source_dropped": dropped,
        "policy": "prefer source-grounded nvBench; drop conflicting synthetic record",
    }
    write_json_atomic(duplicates, out_dir / "duplicate_report.json")

    manifest = {
        "dataset": "hybrid_dataset_v1",
        "hybrid_spec_version": HYBRID_SPEC_VERSION,
        "timestamp_utc": validation_report["timestamp_utc"],
        "status": status,
        "inputs": {
            "nvbench_enrichment": str(enrichment_dir.relative_to(_PROJECT_ROOT)),
            "synthetic": str(synthetic_dir.relative_to(_PROJECT_ROOT)),
            "excluded": ["nvbench test.jsonl", "human_eval_test_items_40.csv",
                         "literature-based L1 gold", "rejected enrichment records",
                         "data/raw_legacy", "data/processed"],
        },
        "counts": validation_report["counts"],
        "field_lineage": {
            "nvbench_enrichment_fields": {f: "llm_generated" for f in ENRICHABLE_FIELDS},
            "sources": [SOURCE_NVBENCH, SOURCE_SYNTHETIC],
            "not_gold": ["human_gold", "expert_gold", "independent_evidence"],
        },
        "distributions": validation_report["summary"],
        "checks": checks,
    }
    write_json_atomic(manifest, out_dir / "manifest.json")

    hashes = {p.name: _sha256(p) for p in sorted(out_dir.glob("*.jsonl"))}
    write_json_atomic(hashes, out_dir / "hashes.json")

    lines = [
        "# Hybrid dataset v1 — validation report",
        "",
        f"- status: **{status}**",
        f"- spec: `{HYBRID_SPEC_VERSION}`",
        f"- built: {validation_report['timestamp_utc']}",
        "",
        "## Counts",
        "",
        f"- hybrid train: {len(hybrid['train'])} "
        f"(nvBench {len(nvbench['train'])} + synthetic {len(synthetic['train'])})",
        f"- hybrid val: {len(hybrid['val'])} "
        f"(nvBench {len(nvbench['val'])} + synthetic {len(synthetic['val'])})",
        f"- synthetic dropped as cross-source duplicates: {len(dropped)}",
        f"- permanently rejected enrichment records (excluded): {len(permanently_rejected)}",
        "",
        "## Checks",
        "",
    ]
    lines += [f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in checks.items()]
    lines += [
        "",
        "## Roles and lineage",
        "",
        "- nvBench records carry the source-grounded analytical content; their six "
        f"presentation fields ({', '.join(ENRICHABLE_FIELDS)}) are `llm_generated`.",
        "- synthetic records are deterministically generated design examples and are "
        "neither independent evidence nor human gold.",
        "- held-out sets (nvBench test, human-evaluation items, literature L1 gold) were "
        "used only as leakage references and never as inputs.",
    ]
    (out_dir / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"  hybrid train     : {len(hybrid['train'])} "
          f"(nvbench {len(nvbench['train'])} + synthetic {len(synthetic['train'])})")
    print(f"  hybrid val       : {len(hybrid['val'])} "
          f"(nvbench {len(nvbench['val'])} + synthetic {len(synthetic['val'])})")
    for name, ok in checks.items():
        if not ok:
            print(f"  FAILED CHECK     : {name}")
    print(f"  outputs          : {out_dir}")
    print(f"  status           : {status}")
    print("=" * 64)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
