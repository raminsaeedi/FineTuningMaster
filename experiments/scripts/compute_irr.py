"""Aggregate human ratings: inter-rater reliability + per-system scores + stats.

    python scripts/compute_irr.py

Reads results/human_ratings/*.jsonl, then writes to results/human_ratings/:
  - irr_alphas.csv     Krippendorff's ordinal alpha per rubric dimension
  - system_means.csv   mean score per method x dimension (unblinded)
  - human_stats.json   Friedman + pairwise Wilcoxon+Holm on the per-item overall
                       human score across methods (with Cliff's delta + bootstrap CI)
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.human.irr import krippendorff_alpha
from src.evaluation.human.rubric import RUBRIC_KEYS
from src.evaluation.human.storage import load_all_ratings
from src.evaluation.stats import (
    cliffs_delta,
    friedman_test,
    paired_bootstrap_diff,
    pairwise_wilcoxon,
)
from src.utils.io import write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute IRR and human-score statistics")
    p.add_argument("--ratings-dir", default="experiments/results/human_ratings")
    p.add_argument("--out-dir", default="experiments/results/human_ratings")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ratings_dir = _PROJECT_ROOT / args.ratings_dir
    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_all_ratings(ratings_dir)
    if not rows:
        raise SystemExit(f"No ratings found in {ratings_dir}.")

    # ---- IRR per dimension (units = unit_id; values = scores across raters) ----
    alphas = {}
    for dim in RUBRIC_KEYS:
        units = collections.defaultdict(list)
        for r in rows:
            if dim in r.get("scores", {}):
                units[r["unit_id"]].append(r["scores"][dim])
        alphas[dim] = krippendorff_alpha(list(units.values()), level="ordinal")

    # ---- Per-system means (unblinded via stored method) ----
    sys_dim_vals = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for dim, v in r.get("scores", {}).items():
            sys_dim_vals[r["method"]][dim].append(v)

    methods = sorted(sys_dim_vals)
    system_means = {}
    for m in methods:
        system_means[m] = {dim: round(statistics.mean(sys_dim_vals[m][dim]), 3)
                           for dim in RUBRIC_KEYS if sys_dim_vals[m][dim]}
        all_scores = [v for dim in RUBRIC_KEYS for v in sys_dim_vals[m][dim]]
        system_means[m]["overall_mean"] = round(statistics.mean(all_scores), 3) if all_scores else None

    # ---- Per-item overall human score per method, for cross-system stats ----
    # overall = mean across dimensions, averaged across raters, per (item, method).
    cell = collections.defaultdict(list)  # (item, method) -> list of per-rating means
    for r in rows:
        scores = [r["scores"][d] for d in RUBRIC_KEYS if d in r.get("scores", {})]
        if scores:
            cell[(r["item_id"], r["method"])].append(statistics.mean(scores))
    item_method_score = {k: statistics.mean(v) for k, v in cell.items()}

    items_all = sorted({i for (i, _m) in item_method_score})
    common_items = [i for i in items_all if all((i, m) in item_method_score for m in methods)]
    scores_by_method = {m: [item_method_score[(i, m)] for i in common_items] for m in methods}

    human_stats = {"methods": methods, "n_common_items": len(common_items)}
    if len(common_items) >= 2 and len(methods) >= 2:
        human_stats["friedman"] = friedman_test(scores_by_method)
        pw = pairwise_wilcoxon(scores_by_method)
        for rrow in pw:
            d, mag = cliffs_delta(scores_by_method[rrow["method_a"]], scores_by_method[rrow["method_b"]])
            rrow["cliffs_delta"] = d
            rrow["cliffs_magnitude"] = mag
            rrow["bootstrap_diff"] = paired_bootstrap_diff(
                scores_by_method[rrow["method_a"]], scores_by_method[rrow["method_b"]]
            )
        human_stats["pairwise_wilcoxon_holm"] = pw

    # ---- Write outputs ----
    write_json({"krippendorff_alpha_ordinal": alphas}, out_dir / "irr_alphas.json")
    write_json({"system_means": system_means}, out_dir / "system_means.json")
    write_json(human_stats, out_dir / "human_stats.json")
    try:
        import pandas as pd

        pd.DataFrame([{"dimension": k, "alpha": v} for k, v in alphas.items()]).to_csv(
            out_dir / "irr_alphas.csv", index=False)
        pd.DataFrame(system_means).T.to_csv(out_dir / "system_means.csv")
    except ImportError:
        pass

    print("=" * 56)
    print("HUMAN EVALUATION — IRR & SCORES")
    print("=" * 56)
    print(f"  Ratings        : {len(rows)} across {len({r['rater_id'] for r in rows})} raters")
    print("  Krippendorff's alpha (ordinal) per dimension:")
    for dim, a in alphas.items():
        flag = "" if (a is None or a >= 0.667) else "  <- below 0.667"
        print(f"     {dim:24} {a}{flag}")
    print("  Overall mean per system:")
    for m in methods:
        print(f"     {m:12} {system_means[m].get('overall_mean')}")
    if "friedman" in human_stats:
        print(f"  Friedman p = {human_stats['friedman'].get('p_value', 'n/a')}")
    print(f"\n  Saved to: {out_dir}")
    print("=" * 56)


if __name__ == "__main__":
    main()
