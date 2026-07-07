"""Derive per-item provenance/lineage metadata for the datasets (derive-only).

No dataset records are mutated. Fields are derived from the record content
(`brief.extra`, benchmark fields), the file the record lives in, and the dataset
context. Used by `experiments/scripts/build_provenance_report.py`.
"""

from __future__ import annotations

from typing import Dict, Optional

# file -> (split, intended_use)
INTENDED_USE: Dict[str, tuple] = {
    "train.jsonl": ("train", "train"),
    "val.jsonl": ("val", "validation"),
    "internal_test.jsonl": ("test", "internal_diagnostic"),
    "benchmark_v1.jsonl": ("benchmark", "independent_benchmark"),
    "real_briefs_v1.jsonl": ("eval", "independent_eval"),
}


def derive_synthetic(record: dict, file_name: str, dataset_version: str) -> dict:
    """Provenance for a synthetic frozen GoldItem record."""
    split, intended_use = INTENDED_USE.get(file_name, ("?", "?"))
    extra = (record.get("brief") or {}).get("extra") or {}
    gen = extra.get("generator_version", "unknown")
    return {
        "item_id": record.get("item_id", ""),
        "source_name": extra.get("source_id", "synthetic_generator"),
        "source_type": "synthetic",
        "license": "synthetic (project-internal)",
        "generation_method": f"source_conditioned_generator:{gen}",
        "label_source": "synthetic_generator",
        "label_lineage_id": f"synthetic_generator:TASK_CHART:{gen}",
        "split": split,
        "intended_use": intended_use,
        "is_synthetic": True,
        "dataset_version": dataset_version,
        "independent_eval_safe": False,
        "notes": "circular for chart choice; internal diagnostic only",
    }


def derive_benchmark(record: dict, dataset_version: str) -> dict:
    """Provenance for a benchmark_v1 item."""
    label_source = record.get("label_source", "?")
    lineage = ("literature_L1:Saket2019+KimHeer2018" if label_source == "literature_L1"
               else f"{label_source}")
    return {
        "item_id": record.get("benchmark_id", ""),
        "source_name": record.get("source_name", ""),
        "source_type": record.get("source_type", "?"),
        "license": record.get("license_or_usage_note", ""),
        "generation_method": ("reused_public_brief" if record.get("source_type") == "real_public"
                              else "author_drafted"),
        "label_source": label_source,
        "label_lineage_id": lineage,
        "split": "benchmark",
        "intended_use": "independent_benchmark",
        "is_synthetic": False,
        "dataset_version": dataset_version,
        "independent_eval_safe": True,
        "notes": "evaluation-only (benchmark lock)",
    }


def derive_real_brief(record: dict, dataset_version: str) -> dict:
    """Provenance for a real_briefs_v1 item (external brief, no chart labels)."""
    extra = record.get("extra") or {}
    return {
        "item_id": record.get("item_id", ""),
        "source_name": extra.get("provenance_id", "real_brief"),
        "source_type": "real_brief",
        "license": "see docs/datasets/real_briefs_provenance.md",
        "generation_method": "curated_public_brief",
        "label_source": "none",
        "label_lineage_id": "none",
        "split": "eval",
        "intended_use": "independent_eval",
        "is_synthetic": False,
        "dataset_version": dataset_version,
        "independent_eval_safe": True,
        "notes": "external brief, no chart labels; never for training",
    }
