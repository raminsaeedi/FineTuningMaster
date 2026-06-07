"""Build perturbed test sets so the robustness metrics stop being null.

Reads the canonical test split and writes two perturbed variants next to it,
each row keeping the ORIGINAL ``item_id`` so the robustness metric can pair
original vs perturbed predictions:

    data/processed/test_paraphrased.jsonl   (paraphrase_consistency)
    data/processed/test_missing_info.jsonl  (missing_info_validity_rate)

Run after build_data.py; the experiment runner then picks these up automatically
(via data.paraphrased_file / data.missing_info_file) on the next run.

    python scripts/build_perturbations.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.perturbations import drop_info_brief, paraphrase_brief
from src.utils.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate paraphrase / missing-info test variants")
    p.add_argument("--processed-dir", default="data/processed")
    return p.parse_args()


def _perturb_rows(rows, transform):
    out = []
    for row in rows:
        brief = dict(row.get("brief", {}))
        item_id = row.get("item_id") or brief.get("item_id", "")
        new_brief = transform(brief)
        new_brief["item_id"] = item_id  # keep id stable so pairs line up
        out.append({
            "item_id": item_id,
            "split": row.get("split"),
            "brief": new_brief,
            "recommendation": row.get("recommendation", {}),
        })
    return out


def main() -> None:
    args = parse_args()
    proc = (_PROJECT_ROOT / args.processed_dir) if not Path(args.processed_dir).is_absolute() else Path(args.processed_dir)
    test_path = proc / "test.jsonl"
    if not test_path.exists():
        raise SystemExit(f"Missing {test_path}. Run `python scripts/build_data.py` first.")

    rows = read_jsonl(test_path)
    variants = {
        "test_paraphrased.jsonl": _perturb_rows(rows, paraphrase_brief),
        "test_missing_info.jsonl": _perturb_rows(rows, drop_info_brief),
    }
    print("=" * 56)
    print("PERTURBED TEST SETS BUILT")
    print("=" * 56)
    print(f"  Source : {test_path} ({len(rows)} items)")
    for fname, out_rows in variants.items():
        write_jsonl(out_rows, proc / fname)
        print(f"  {fname}: {len(out_rows)} items")
    print("=" * 56)


if __name__ == "__main__":
    main()
