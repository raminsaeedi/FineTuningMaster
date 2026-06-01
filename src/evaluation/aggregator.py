"""Collect per-experiment metrics into one flat table.

Walks the experiments root, reads each ``metrics_auto.json`` (plus the config
snapshot for method/model identity), and flattens the nested metric dicts into
one row per experiment. Useful for the final results table and figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.utils.io import read_json, read_yaml


def _flatten(prefix: str, obj: Any, out: Dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    else:
        out[prefix] = obj


def collect_rows(outputs_root: str | Path) -> List[Dict[str, Any]]:
    """Return one flattened metrics row per experiment directory."""
    root = Path(outputs_root)
    rows: List[Dict[str, Any]] = []
    if not root.exists():
        return rows

    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        metrics_path = exp_dir / "metrics_auto.json"
        if not metrics_path.exists():
            continue
        row: Dict[str, Any] = {"experiment_id": exp_dir.name}

        snap = exp_dir / "config_snapshot.yaml"
        if snap.exists():
            cfg = read_yaml(snap)
            method = cfg.get("method", {})
            model = cfg.get("model", {})
            row["method"] = method.get("name") if isinstance(method, dict) else None
            row["model"] = model.get("name") if isinstance(model, dict) else None
            row["seed"] = cfg.get("seed")

        _flatten("", read_json(metrics_path), row)
        rows.append(row)
    return rows


def aggregate(outputs_root: str | Path, csv_path: str | Path | None = None):
    """Build a DataFrame of all experiment metrics; optionally write a CSV."""
    import pandas as pd

    df = pd.DataFrame(collect_rows(outputs_root))
    if csv_path is not None and not df.empty:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    return df
