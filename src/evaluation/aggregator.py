"""Collect per-experiment metrics into one flat table.

Walks the experiments root, reads each ``metrics_auto.json`` (plus the config
snapshot for method/model identity), and flattens the nested metric dicts into
one row per experiment. Useful for the final results table and figures.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.evaluation.stats.bootstrap_ci import bootstrap_ci
from src.utils.io import read_json, read_yaml


def _flatten(prefix: str, obj: Any, out: Dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    else:
        out[prefix] = obj


def collect_rows(outputs_root: str | Path) -> List[Dict[str, Any]]:
    """Return one flattened metrics row per completed experiment directory.

    Identity comes from the run manifest when available, with the resolved config
    snapshot as a legacy fallback. Metrics remain sourced only from
    ``metrics_auto.json``; missing values stay null.
    """
    root = Path(outputs_root)
    rows: List[Dict[str, Any]] = []
    if not root.exists():
        return rows

    metric_files = sorted(root.rglob("metrics_auto.json"))
    for metrics_path in metric_files:
        exp_dir = metrics_path.parent
        manifest = {}
        manifest_path = exp_dir / "manifest.json"
        if manifest_path.exists():
            try:
                value = read_json(manifest_path)
                manifest = value if isinstance(value, dict) else {}
            except (OSError, ValueError):
                manifest = {}
        snap = exp_dir / "config_snapshot.yaml"
        cfg = {}
        if snap.exists():
            try:
                cfg = read_yaml(snap)
            except (OSError, ValueError):
                cfg = {}
        method_cfg = cfg.get("method", {}) if isinstance(cfg, dict) else {}
        model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
        data_hashes = manifest.get("data_file_sha256", {}) or {}
        model_hf_id = (
            manifest.get("model_hf_id")
            or (model_cfg.get("hf_id") if isinstance(model_cfg, dict) else None)
            or manifest.get("model")
            or (model_cfg.get("name") if isinstance(model_cfg, dict) else None)
        )
        test_file = data_cfg.get("test_file") if isinstance(data_cfg, dict) else None
        config_hash = manifest.get("config_hash")
        config_hash_path = exp_dir / "config_hash.txt"
        if not config_hash and config_hash_path.exists():
            config_hash = config_hash_path.read_text(encoding="utf-8").strip() or None

        row: Dict[str, Any] = {"experiment_id": exp_dir.name}
        _flatten("", read_json(metrics_path), row)
        row.update({
            "experiment_id": manifest.get("experiment_id", row.get("experiment_id", exp_dir.name)),
            "experiment": manifest.get("experiment_name", exp_dir.name),
            "method": manifest.get("method") or (
                method_cfg.get("name") if isinstance(method_cfg, dict) else None
            ) or row.get("method"),
            "model": model_hf_id or row.get("model"),
            "model_key": manifest.get("model_key") or (
                model_cfg.get("key") if isinstance(model_cfg, dict) else None
            ),
            "model_config": model_cfg.get("name") if isinstance(model_cfg, dict) else None,
            "seed": manifest.get("seed", row.get("seed")),
            "method_key": manifest.get("method_key"),
            "profile": manifest.get("profile"),
            "run_id": manifest.get("run_id") or manifest.get("experiment_id"),
            "dataset_version": manifest.get("dataset_version") or (
                data_cfg.get("dataset_version") if isinstance(data_cfg, dict) else None
            ),
            "dataset_hash": manifest.get("dataset_hash") or data_hashes.get("test"),
            "test_file": test_file,
            "test_file_sha256": data_hashes.get("test"),
            "config_hash": config_hash,
            "report_schema_version": manifest.get("report_schema_version"),
            "run_status": manifest.get("status"),
            "source_c_run_id": manifest.get("source_c_run_id"),
            "adapter_manifest_hash": manifest.get("adapter_manifest_hash"),
            "cache_identity_hash": manifest.get("cache_identity_hash"),
            "run_path": str(exp_dir.resolve()),
        })
        rows.append(row)
    return rows


def aggregate(outputs_root: str | Path, csv_path: str | Path | None = None):
    """Build a DataFrame of all experiment metrics; optionally write a CSV."""
    df = pd.DataFrame(collect_rows(outputs_root))
    if csv_path is not None and not df.empty:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    return df


_SUMMARY_METRICS = {
    "json_parse_rate": "metrics.schema_compliance.json_parse_rate",
    "schema_validity_rate": "metrics.schema_compliance.schema_validity_rate",
    "completeness_score": "metrics.schema_compliance.completeness_score",
    "top_1_accuracy": "metrics.top_k_accuracy.top_1_accuracy",
    "macro_f1": "metrics.macro_f1.macro_f1",
    "exact_task_classification": (
        "metrics.structured_exact_match.exact_task_classification"
    ),
    "exact_kpi_selection": "metrics.structured_exact_match.exact_kpi_selection",
    "exact_mapping_count": "metrics.structured_exact_match.exact_mapping_count",
    "exact_encoding": "metrics.structured_exact_match.exact_encoding",
    "exact_aggregate": "metrics.structured_exact_match.exact_aggregate",
    "avg_latency_ms": "metrics.latency.avg_latency_ms",
    "paraphrase_accuracy": "metrics.robustness.paraphrase_accuracy",
    "paraphrase_consistency": "metrics.robustness.paraphrase_consistency",
    "missing_info_clarification_rate": (
        "metrics.robustness.missing_info_clarification_rate"
    ),
    "missing_info_schema_rate": "metrics.robustness.missing_info_schema_rate",
    "supported_claim_rate": "metrics.grounding.supported_claim_rate",
}


def _numeric_column(df: pd.DataFrame, name: str) -> str | None:
    if name in df.columns:
        return name
    return next((column for column in df.columns if column.endswith(name)), None)


def build_multi_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build descriptive per-seed summaries without pooling prediction rows.

    Each output row represents one metric for one method/model/dataset group.
    Raw seed columns remain visible. With fewer than five seeds, confidence
    intervals are deliberately left null because three-run thesis matrices do
    not support strong distributional claims.
    """
    identity = ["method", "model", "dataset_version", "test_file_sha256"]
    for column in identity:
        if column not in df.columns:
            df = df.copy()
            df[column] = None
    if "seed" not in df.columns:
        df = df.copy()
        df["seed"] = None
    if df.empty:
        columns = identity + ["metric", "n_seeds", "seed_list", "mean", "std", "ci_low", "ci_high", "ci_note"]
        return pd.DataFrame(columns=columns)

    seed_values = pd.to_numeric(df.get("seed"), errors="coerce")
    all_seeds = sorted({int(value) for value in seed_values.dropna().tolist()})
    seed_columns = [f"seed_{seed}" for seed in all_seeds]
    output: list[dict] = []
    grouped = df.groupby(identity, dropna=False, sort=True)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        base = dict(zip(identity, group_key))
        group_seed_values = pd.to_numeric(group.get("seed"), errors="coerce")
        group_seeds = sorted({int(value) for value in group_seed_values.dropna().tolist()})
        seed_list = ",".join(str(seed) for seed in group_seeds)
        for metric, suffix in _SUMMARY_METRICS.items():
            column = _numeric_column(group, suffix)
            by_seed: dict[int, float | None] = {}
            if column is not None:
                for _, row in group.iterrows():
                    raw_seed = row.get("seed")
                    if pd.isna(raw_seed):
                        continue
                    value = pd.to_numeric(row.get(column), errors="coerce")
                    by_seed[int(raw_seed)] = None if pd.isna(value) else float(value)

            values = [value for value in by_seed.values() if value is not None]
            summary = {
                **base,
                "metric": metric,
                "n_seeds": len(group_seeds),
                "n_values": len(values),
                "seed_list": seed_list,
                **{column_name: by_seed.get(seed) for seed, column_name in zip(all_seeds, seed_columns)},
            }
            if values:
                summary["mean"] = float(pd.Series(values).mean())
                summary["std"] = float(pd.Series(values).std(ddof=1)) if len(values) >= 2 else None
                if len(values) >= 5:
                    ci = bootstrap_ci(values, n_boot=2000)
                    summary["ci_low"] = float(ci["ci_low"])
                    summary["ci_high"] = float(ci["ci_high"])
                    summary["ci_note"] = "percentile bootstrap over independent seeds"
                else:
                    summary["ci_low"] = None
                    summary["ci_high"] = None
                    summary["ci_note"] = "not reported for fewer than 5 independent seeds"
            else:
                summary.update({
                    "mean": None,
                    "std": None,
                    "ci_low": None,
                    "ci_high": None,
                    "ci_note": "metric unavailable in source runs",
                })
            output.append(summary)

    columns = identity + ["metric", "n_seeds", "n_values", "seed_list"] + seed_columns + [
        "mean", "std", "ci_low", "ci_high", "ci_note"
    ]
    return pd.DataFrame(output, columns=columns)
