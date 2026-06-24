"""Statistical comparison of methods across experiments.

Given two or more experiments (each a method run on the same test split), this
builds matched per-item vectors and runs the protocol's tests:

  - Top-1 correctness (binary): Cochran's Q (k>=3) + pairwise exact McNemar+Holm
  - Schema completeness (continuous): Friedman (k>=3) + pairwise Wilcoxon+Holm,
    with Cliff's delta and a paired bootstrap CI for each pair

Results are printed and written as CSVs under results/stats/.

    python scripts/eval_stats.py --experiments E01_qwen0_5b_prompt E03_qwen0_5b_ft
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.core.schemas import GenerationResult  # noqa: E402
from src.data_pipeline.dataset import load_gold_items  # noqa: E402
from src.evaluation.metrics.base import normalise, predicted_charts, reference_charts  # noqa: E402
from src.evaluation.metrics.schema_compliance import completeness_fraction  # noqa: E402
from src.evaluation.stats import (  # noqa: E402
    cliffs_delta,
    cochran_q,
    cohen_dz,
    friedman_test,
    paired_bootstrap_diff,
    paired_rank_biserial,
    pairwise_mcnemar,
    pairwise_wilcoxon,
)
from src.inference.postprocess import extract_json_dict, reparse  # noqa: E402
from src.utils.artifacts import experiment_dir  # noqa: E402
from src.utils.config import load_cfg  # noqa: E402
from src.utils.io import read_jsonl, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Statistical tests across experiments")
    p.add_argument("--experiments", nargs="+", required=True,
                   help="Experiment config names to compare")
    p.add_argument("--out-dir", default="experiments/results/stats")
    return p.parse_args()


def _completeness(raw_text: str) -> float:
    # Corrected: a required key counts only if present AND non-empty.
    return completeness_fraction(extract_json_dict(raw_text))


def _top1_correct(result: GenerationResult, ref_charts: List[str]) -> int:
    preds = [normalise(c) for c in predicted_charts(result)]
    refs = [normalise(c) for c in ref_charts]
    if not preds or not refs:
        return 0
    return int(preds[0] == refs[0])


def main() -> None:
    args = parse_args()

    # Load each experiment's method name + predictions.
    method_names: Dict[str, str] = {}
    preds_by_method: Dict[str, Dict[str, GenerationResult]] = {}
    references = None

    for name in args.experiments:
        cfg = load_cfg(experiment=name)
        method = str(cfg.method.name)
        exp_dir = experiment_dir(cfg, _PROJECT_ROOT)
        pred_path = exp_dir / "predictions.jsonl"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions for {name}: {pred_path}")
        results = {r["item_id"]: reparse(GenerationResult(**r)) for r in read_jsonl(pred_path)}
        method_names[name] = method
        preds_by_method[method] = results

        if references is None:
            test_file = Path(str(cfg.data.test_file))
            if not test_file.is_absolute():
                test_file = _PROJECT_ROOT / test_file
            references = {
                it.item_id: reference_charts({"recommendation": it.recommendation.model_dump(mode="json")})
                for it in load_gold_items(test_file)
            }

    methods = list(preds_by_method)
    # Items present in every method AND with a reference.
    common = set(references)
    for m in methods:
        common &= set(preds_by_method[m])
    common_ids = sorted(common)
    if not common_ids:
        raise SystemExit("No overlapping items across experiments — nothing to compare.")

    top1: Dict[str, List[int]] = {m: [] for m in methods}
    completeness: Dict[str, List[float]] = {m: [] for m in methods}
    for item_id in common_ids:
        for m in methods:
            r = preds_by_method[m][item_id]
            top1[m].append(_top1_correct(r, references[item_id]))
            completeness[m].append(_completeness(r.raw_text))

    # ── Binary outcome (Top-1) ───────────────────────────────────────────────
    binary_report = {
        "metric": "top1_correct",
        "methods": methods,
        "n_items": len(common_ids),
        "cochran_q": cochran_q(top1),
        "pairwise_mcnemar": pairwise_mcnemar(top1),
    }

    # ── Continuous outcome (completeness) ────────────────────────────────────
    pairwise_w = pairwise_wilcoxon(completeness)
    for r in pairwise_w:
        a, b = completeness[r["method_a"]], completeness[r["method_b"]]
        # Paired effect sizes (correct for the matched-item design).
        r["rank_biserial"] = paired_rank_biserial(a, b)
        r["cohen_dz"] = cohen_dz(a, b)
        # Cliff's delta retained as an auxiliary (unpaired) effect size only.
        delta, mag = cliffs_delta(a, b)
        r["cliffs_delta_unpaired"] = delta
        r["cliffs_magnitude_unpaired"] = mag
        r["bootstrap_diff"] = paired_bootstrap_diff(a, b)
    continuous_report = {
        "metric": "schema_completeness",
        "methods": methods,
        "n_items": len(common_ids),
        "friedman": friedman_test(completeness),
        "pairwise_wilcoxon_holm": pairwise_w,
    }

    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json({"experiments": method_names, "binary": binary_report,
                "continuous": continuous_report}, out_dir / "stats_report.json")

    # Flat CSVs for the thesis tables.
    try:
        import pandas as pd

        pd.DataFrame(binary_report["pairwise_mcnemar"]).to_csv(out_dir / "posthoc_mcnemar.csv", index=False)
        pd.DataFrame([
            {k: v for k, v in r.items() if k != "bootstrap_diff"} | {
                "diff_ci_low": r["bootstrap_diff"]["ci_low"],
                "diff_ci_high": r["bootstrap_diff"]["ci_high"],
                "mean_diff": r["bootstrap_diff"]["mean_diff"],
            }
            for r in pairwise_w
        ]).to_csv(out_dir / "posthoc_wilcoxon.csv", index=False)
    except ImportError:
        pass

    print("=" * 60)
    print("STATISTICAL COMPARISON")
    print("=" * 60)
    print(f"  Methods   : {methods}")
    print(f"  Matched n : {len(common_ids)}")
    print(f"  Friedman  : {continuous_report['friedman'].get('p_value', 'n/a (k<3)')}")
    print(f"  Cochran Q : {binary_report['cochran_q'].get('p_value', 'n/a (k<3)')}")
    print("  Pairwise (completeness, Wilcoxon+Holm):")
    for r in pairwise_w:
        print(f"    {r['method_a']} vs {r['method_b']}: p_holm={r['p_holm']:.4f} "
              f"rank_biserial={r['rank_biserial']} d_z={r['cohen_dz']}")
    print("  Pairwise (top-1, McNemar+Holm):")
    for r in binary_report["pairwise_mcnemar"]:
        print(f"    {r['method_a']} vs {r['method_b']}: p_holm={r['p_holm']:.4f} "
              f"(b={r['b']}, c={r['c']})")
    print("=" * 60)
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
