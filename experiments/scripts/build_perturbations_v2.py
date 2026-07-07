"""Build v2 perturbation sets from the frozen internal_test split.

Reads ``data/frozen/dashboard_v2/internal_test.jsonl`` and writes two variants
next to it, each row paired to its source item via ``extra.base_item_id`` and
tagged with ``extra.variant``:

    test_paraphrased.jsonl   (paraphrase_consistency / paraphrase_accuracy)
    test_missing_info.jsonl  (missing-info robustness)

The reference ``recommendation`` is left unchanged — it is the gold the perturbed
input is scored against. Reuses the deterministic perturbations in
``src.data_pipeline.perturbations`` (no LLM, no randomness).

Usage:
    python experiments/scripts/build_perturbations_v2.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.perturbations import drop_info_brief, paraphrase_brief
from src.utils.io import read_jsonl, write_jsonl


def _perturb_rows(rows, transform, variant):
    out = []
    for row in rows:
        brief = dict(row.get("brief", {}))
        base_id = row.get("item_id") or brief.get("item_id", "")
        new_brief = transform(brief)
        new_brief["item_id"] = base_id  # keep id stable so pairs line up
        extra = dict(new_brief.get("extra") or {})
        extra["variant"] = variant
        extra["base_item_id"] = base_id
        new_brief["extra"] = extra
        out.append({
            "item_id": base_id,
            "split": row.get("split"),
            "brief": new_brief,
            "recommendation": row.get("recommendation", {}),
        })
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate v2 paraphrase / missing-info variants")
    p.add_argument("--frozen-dir", default="data/frozen/dashboard_v2")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frozen_dir = (_PROJECT_ROOT / args.frozen_dir) if not Path(args.frozen_dir).is_absolute() else Path(args.frozen_dir)
    base_path = frozen_dir / "internal_test.jsonl"
    if not base_path.exists():
        raise SystemExit(f"Missing {base_path}. Run freeze_dataset_v2.py first.")

    rows = read_jsonl(base_path)
    variants = {
        "test_paraphrased.jsonl": _perturb_rows(rows, paraphrase_brief, "paraphrased"),
        "test_missing_info.jsonl": _perturb_rows(rows, drop_info_brief, "missing_info"),
    }
    print("=" * 56)
    print("DATASET V2 PERTURBATION SETS BUILT")
    print("=" * 56)
    print(f"  Source : {base_path} ({len(rows)} items)")
    for fname, out_rows in variants.items():
        write_jsonl(out_rows, frozen_dir / fname)
        print(f"  {fname}: {len(out_rows)} items")
    print("=" * 56)


if __name__ == "__main__":
    main()
