"""Paired statistical comparison for independently executed experiment runs.

The command compares one selected seed at a time. Output-root, model and seed
are passed through the same Hydra-style override mechanism as the other scripts.
Run compatibility is checked before any paired statistic is computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
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
    per_method_bootstrap_cis,
)
from src.inference.postprocess import extract_json_dict, reparse  # noqa: E402
from src.utils.adapter import read_training_metadata, resolve_adapter_path  # noqa: E402
from src.utils.config_hash import hash_config  # noqa: E402
from src.utils.artifacts import experiment_dir  # noqa: E402
from src.utils.config import load_cfg  # noqa: E402
from src.utils.io import read_json, read_jsonl, write_json  # noqa: E402


class IncompatiblePairedRuns(RuntimeError):
    """Raised when runs cannot support valid paired statistics."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Statistical tests across experiments")
    p.add_argument("--experiments", nargs="+", required=True,
                   help="Experiment config names to compare")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Hydra-style overrides, e.g. output_root=experiments/outputs/final seed=43")
    p.add_argument("--out-dir", default="experiments/results/stats")
    return p.parse_args()


def _completeness(raw_text: str) -> float:
    return completeness_fraction(extract_json_dict(raw_text))


def _top1_correct(result: GenerationResult, ref_charts: List[str]) -> int:
    preds = [normalise(c) for c in predicted_charts(result)]
    refs = [normalise(c) for c in ref_charts]
    if not preds or not refs:
        return 0
    return int(preds[0] == refs[0])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(raw: str, project_root: Path = _PROJECT_ROOT) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else project_root / path


def _model_id(cfg, manifest: dict) -> str:
    model_cfg = cfg.get("model", {})
    return str(
        manifest.get("model_hf_id")
        or model_cfg.get("hf_id")
        or manifest.get("model")
        or model_cfg.get("name")
        or ""
    )


def _test_file(cfg, manifest: dict) -> Path:
    data_cfg = cfg.get("data", {})
    raw = data_cfg.get("test_file")
    if not raw:
        raise FileNotFoundError("Run config has no data.test_file")
    path = _resolve_path(str(raw))
    if not path.exists():
        raise FileNotFoundError(f"Test split not found: {path}")
    return path


def _read_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run manifest: {path}")
    try:
        manifest = read_json(path)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Invalid run manifest: {path}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"Invalid run manifest: {path}")
    return manifest


def _read_predictions(run_dir: Path) -> Dict[str, GenerationResult]:
    path = run_dir / "predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions for run: {path}")
    predictions: Dict[str, GenerationResult] = {}
    for record in read_jsonl(path):
        item_id = str(record.get("item_id", ""))
        if not item_id:
            raise RuntimeError(f"Prediction without item_id: {path}")
        if item_id in predictions:
            raise RuntimeError(f"Duplicate prediction item_id '{item_id}': {path}")
        predictions[item_id] = reparse(GenerationResult(**record))
    return predictions


def _adapter_info(cfg, manifest: dict, method: str) -> tuple[Path | None, dict]:
    if method not in {"ft", "ft_rag"}:
        return None, {}
    info = manifest.get("adapter", {}) or {}
    raw_path = info.get("adapter_path")
    path = _resolve_path(raw_path) if raw_path else resolve_adapter_path(cfg, _PROJECT_ROOT)
    metadata = info.get("adapter_training_metadata") or {}
    if not metadata and path is not None:
        metadata = read_training_metadata(path)
    return path, metadata if isinstance(metadata, dict) else {}


def bundle_from_config(cfg) -> dict:
    """Load one run and its exact provenance from a composed config."""
    run_dir = experiment_dir(cfg, _PROJECT_ROOT)
    manifest = _read_manifest(run_dir)
    predictions = _read_predictions(run_dir)
    test_file = _test_file(cfg, manifest)
    references = {
        item.item_id: reference_charts({
            "recommendation": item.recommendation.model_dump(mode="json")
        })
        for item in load_gold_items(test_file)
    }
    data_hashes = manifest.get("data_file_sha256", {}) or {}
    test_hash = manifest.get("test_file_sha256") or data_hashes.get("test")
    if not test_hash:
        test_hash = _sha256_file(test_file)
    method = str(manifest.get("method") or cfg.method.name)
    adapter_path, adapter_metadata = _adapter_info(cfg, manifest, method)
    eval_cfg = cfg.get("eval", {})
    protocol = {
        "report_schema_version": manifest.get("report_schema_version"),
        "metrics": list(eval_cfg.get("metrics", [])),
        "paraphrased_sha256": data_hashes.get("paraphrased"),
        "missing_info_sha256": data_hashes.get("missing_info"),
    }
    config_hash = manifest.get("config_hash")
    hash_path = run_dir / "config_hash.txt"
    if not config_hash and hash_path.exists():
        config_hash = hash_path.read_text(encoding="utf-8").strip() or None
    return {
        "cfg": cfg,
        "run_dir": run_dir,
        "run_id": str(manifest.get("experiment_id") or cfg.get("experiment_id") or run_dir.name),
        "method": method,
        "model": _model_id(cfg, manifest),
        "model_config": str(cfg.model.get("name", "")),
        "seed": int(manifest.get("seed", cfg.get("seed", 42))),
        "dataset_version": manifest.get("dataset_version") or cfg.data.get("dataset_version"),
        "dataset_hash": str(test_hash),
        "train_dataset_hash": data_hashes.get("train"),
        "training_config_hash": hash_config(cfg.get("training", {})),
        "test_file": test_file,
        "test_item_ids": set(references),
        "references": references,
        "predictions": predictions,
        "protocol": protocol,
        "config_hash": config_hash,
        "adapter_path": adapter_path,
        "adapter_metadata": adapter_metadata,
        "manifest": manifest,
    }


def load_run_bundle(name: str, overrides: list[str]) -> dict:
    """Compose one requested experiment using caller-selected overrides."""
    cfg = load_cfg(experiment=name, overrides=overrides)
    return bundle_from_config(cfg)


def load_run_bundles(names: list[str], overrides: list[str]) -> list[dict]:
    return [load_run_bundle(name, overrides) for name in names]


def _raise_incompatible(detail: str) -> None:
    raise IncompatiblePairedRuns(f"INCOMPATIBLE_PAIRED_RUNS: {detail}")


def _validate_adapter_pair(bundles: list[dict]) -> None:
    by_method = {bundle["method"]: bundle for bundle in bundles}
    c = by_method.get("ft")
    d = by_method.get("ft_rag")
    if c is None or d is None:
        return

    expected = (c["run_dir"] / "adapter").resolve()
    actual = d.get("adapter_path")
    if actual is None or actual.resolve() != expected:
        _raise_incompatible(
            f"Method D adapter path {actual} does not match Method C adapter {expected}"
        )
    metadata = d.get("adapter_metadata", {}) or {}
    checks = {
        "base_model": c["model"],
        "seed": c["seed"],
        "dataset_version": c["dataset_version"],
        "train_file_sha256": c["train_dataset_hash"],
        "training_config_hash": c["training_config_hash"],
        "experiment_id": c["run_id"],
    }
    for field, expected_value in checks.items():
        actual_value = metadata.get(field)
        if actual_value is None:
            continue
        if str(actual_value) != str(expected_value):
            _raise_incompatible(
                f"Method D adapter metadata {field}={actual_value!r} "
                f"does not match Method C value {expected_value!r}"
            )


def validate_paired_runs(bundles: list[dict]) -> dict:
    """Reject incompatible runs before constructing paired metric vectors."""
    if len(bundles) < 2:
        raise ValueError("At least two runs are required for paired statistics")
    methods = [bundle["method"] for bundle in bundles]
    if len(set(methods)) != len(methods):
        _raise_incompatible(f"duplicate method names: {methods}")

    baseline = bundles[0]
    for bundle in bundles[1:]:
        for field, label in (
            ("model", "base model"),
            ("model_config", "model configuration"),
            ("seed", "seed"),
            ("dataset_version", "dataset version"),
            ("dataset_hash", "test split hash"),
            ("protocol", "evaluation protocol"),
        ):
            if bundle[field] != baseline[field]:
                _raise_incompatible(
                    f"{label} differs: {baseline['run_id']}={baseline[field]!r}, "
                    f"{bundle['run_id']}={bundle[field]!r}"
                )

    expected_ids = set(baseline["test_item_ids"])
    for bundle in bundles:
        if set(bundle["test_item_ids"]) != expected_ids:
            _raise_incompatible(
                f"test split item IDs differ between {baseline['run_id']} and {bundle['run_id']}"
            )
        prediction_ids = set(bundle["predictions"])
        if prediction_ids != expected_ids:
            missing = sorted(expected_ids - prediction_ids)
            extra = sorted(prediction_ids - expected_ids)
            _raise_incompatible(
                f"prediction item IDs differ for {bundle['run_id']} "
                f"(missing={missing[:5]}, extra={extra[:5]})"
            )

    _validate_adapter_pair(bundles)
    return {
        "methods": methods,
        "n_items": len(expected_ids),
        "common_ids": sorted(expected_ids),
        "model": baseline["model"],
        "seed": baseline["seed"],
        "dataset_version": baseline["dataset_version"],
        "dataset_hash": baseline["dataset_hash"],
    }


def _statistics_payload(bundles: list[dict], validation: dict) -> dict:
    references = bundles[0]["references"]
    common_ids = validation["common_ids"]
    methods = validation["methods"]
    preds_by_method = {
        bundle["method"]: bundle["predictions"] for bundle in bundles
    }
    top1: Dict[str, List[int]] = {method: [] for method in methods}
    completeness: Dict[str, List[float]] = {method: [] for method in methods}
    for item_id in common_ids:
        for method in methods:
            result = preds_by_method[method][item_id]
            top1[method].append(_top1_correct(result, references[item_id]))
            completeness[method].append(_completeness(result.raw_text))

    per_method_ci = {
        "top1_accuracy_pct": per_method_bootstrap_cis(top1, scale=100.0),
        "completeness": per_method_bootstrap_cis(completeness, scale=1.0),
    }
    pairwise_w = pairwise_wilcoxon(completeness)
    for result in pairwise_w:
        a = completeness[result["method_a"]]
        b = completeness[result["method_b"]]
        result["rank_biserial"] = paired_rank_biserial(a, b)
        result["cohen_dz"] = cohen_dz(a, b)
        delta, magnitude = cliffs_delta(a, b)
        result["cliffs_delta_unpaired"] = delta
        result["cliffs_magnitude_unpaired"] = magnitude
        result["bootstrap_diff"] = paired_bootstrap_diff(a, b)

    return {
        "experiments": {bundle["run_id"]: bundle["method"] for bundle in bundles},
        "provenance": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "input_runs": [
                {
                    "run_id": bundle["run_id"],
                    "run_path": str(bundle["run_dir"].resolve()),
                    "method": bundle["method"],
                    "model": bundle["model"],
                    "model_config": bundle["model_config"],
                    "seed": bundle["seed"],
                    "dataset_version": bundle["dataset_version"],
                    "dataset_hash": bundle["dataset_hash"],
                    "test_file": str(bundle["test_file"]),
                    "config_hash": bundle["config_hash"],
                }
                for bundle in bundles
            ],
            "model": validation["model"],
            "seeds": [bundle["seed"] for bundle in bundles],
            "dataset_version": validation["dataset_version"],
            "dataset_hash": validation["dataset_hash"],
            "matched_test_items": validation["n_items"],
            "paired_statistics": True,
            "pooling_across_seeds": False,
        },
        "per_method_ci": per_method_ci,
        "binary": {
            "metric": "top1_correct",
            "methods": methods,
            "n_items": len(common_ids),
            "test": "Cochran's Q + exact McNemar post-hoc",
            "multiple_comparison_correction": "Holm",
            "cochran_q": cochran_q(top1),
            "pairwise_mcnemar": pairwise_mcnemar(top1),
        },
        "continuous": {
            "metric": "schema_completeness",
            "methods": methods,
            "n_items": len(common_ids),
            "test": "Friedman + paired Wilcoxon signed-rank post-hoc",
            "multiple_comparison_correction": "Holm",
            "friedman": friedman_test(completeness),
            "pairwise_wilcoxon_holm": pairwise_w,
        },
    }


def main() -> None:
    args = parse_args()
    try:
        bundles = load_run_bundles(args.experiments, args.override)
        validation = validate_paired_runs(bundles)
    except IncompatiblePairedRuns as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"STATISTICS_INPUT_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    report = _statistics_payload(bundles, validation)
    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(report, out_dir / "stats_report.json")

    binary_report = report["binary"]
    continuous_report = report["continuous"]
    pairwise_w = continuous_report["pairwise_wilcoxon_holm"]
    per_method_ci = report["per_method_ci"]
    try:
        import pandas as pd

        pd.DataFrame(binary_report["pairwise_mcnemar"]).to_csv(
            out_dir / "posthoc_mcnemar.csv", index=False
        )
        pd.DataFrame([
            {k: v for k, v in result.items() if k != "bootstrap_diff"}
            | {
                "diff_ci_low": result["bootstrap_diff"]["ci_low"],
                "diff_ci_high": result["bootstrap_diff"]["ci_high"],
                "mean_diff": result["bootstrap_diff"]["mean_diff"],
            }
            for result in pairwise_w
        ]).to_csv(out_dir / "posthoc_wilcoxon.csv", index=False)
        pd.DataFrame([
            {"metric": metric, "method": method, **ci}
            for metric, by_method in per_method_ci.items()
            for method, ci in by_method.items()
        ]).to_csv(out_dir / "per_method_ci.csv", index=False)
    except ImportError:
        pass

    print("=" * 60)
    print("STATISTICAL COMPARISON")
    print("=" * 60)
    print(f"  Methods   : {validation['methods']}")
    print(f"  Model     : {validation['model']}")
    print(f"  Seed      : {validation['seed']}")
    print(f"  Matched n : {validation['n_items']}")
    print("  Per-method 95% CI (top-1 accuracy %):")
    for method, ci in per_method_ci["top1_accuracy_pct"].items():
        print(f"    {method}: {ci['point']} [{ci['ci_low']}, {ci['ci_high']}]")
    print(f"  Friedman  : {continuous_report['friedman'].get('p_value', 'n/a (k<3)')}")
    print(f"  Cochran Q : {binary_report['cochran_q'].get('p_value', 'n/a (k<3)')}")
    print("  Pairwise (completeness, Wilcoxon+Holm):")
    for result in pairwise_w:
        print(
            f"    {result['method_a']} vs {result['method_b']}: "
            f"p_holm={result['p_holm']:.4f} "
            f"rank_biserial={result['rank_biserial']} d_z={result['cohen_dz']}"
        )
    print("  Pairwise (top-1, McNemar+Holm):")
    for result in binary_report["pairwise_mcnemar"]:
        print(
            f"    {result['method_a']} vs {result['method_b']}: "
            f"p_holm={result['p_holm']:.4f} (b={result['b']}, c={result['c']})"
        )
    print("=" * 60)
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
