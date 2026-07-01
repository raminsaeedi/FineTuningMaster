"""Backfill Task-9 reporting artifacts from EXISTING saved runs — no model execution.

Reads each run's ``predictions.jsonl`` + ``metrics_auto.json`` + ``config_snapshot.yaml``
under the flat root ``experiments/outputs/`` and writes (additively):

    <run>/metrics.json         # layered value/n/CI, backfill-annotated
    <run>/eval_per_item.jsonl  # per-item scored join (run-time stored fields only)

and, under ``experiments/results/``:

    comparison_table.csv, comparison_seeds.csv, final_report.md, backfill_report.md

Legacy carry-forward: references (``data/processed/test.jsonl``) are NEVER used here, even
when present on disk — recomputing against them would produce corrected numbers that don't
match the legacy ``metrics_auto.json`` point values. Gold-dependent per-item fields and all
CIs are therefore always marked ``not_available``; point values are carried unchanged from
``metrics_auto.json`` (legacy, pre-Task-7) and labelled diagnostic-only.

Raw ``predictions.jsonl`` / ``metrics_auto.json`` / ``errors.jsonl`` are NEVER modified;
``manifest.json`` is augmented additively only. Performs NO training / inference / evaluation /
human evaluation, and never moves or deletes runs.

    python scripts/build_run_reports.py
    python scripts/build_run_reports.py --outputs-root experiments/outputs --out-dir experiments/results
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.constants import REPORT_SCHEMA_VERSION  # noqa: E402
from src.core.schemas import GenerationResult  # noqa: E402
from src.evaluation.aggregator import aggregate  # noqa: E402
from src.evaluation.reporting import build_metrics_json, legacy_per_item, mark_backfill  # noqa: E402
from src.utils.io import read_json, read_jsonl, read_yaml, write_json, write_jsonl  # noqa: E402

HEADER_NOTE = (
    "This is a backfilled legacy internal-synthetic diagnostic report. It is not the final "
    "thesis-valid independent evaluation report. L1 human-effectiveness, L3 realism, and L4 "
    "human evaluation are pending."
)
RAW_FILES = ("predictions.jsonl", "metrics_auto.json", "errors.jsonl")
_LAYER_KEYS = {"L1_chart_selection", "L2_format_robustness", "L1c_grounding", "L3_realism", "L4_human"}
_REPORT_METRIC_COLS = [
    ("json_parse", "json_parse%"),
    ("schema_validity_rate", "schema_valid%"),
    ("completeness_score", "complete"),
    ("top_1_accuracy", "top1%(legacy,synthetic)"),
    ("macro_f1", "macro_f1(legacy)"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill reporting artifacts from saved runs")
    p.add_argument("--outputs-root", default="experiments/outputs")
    p.add_argument("--out-dir", default="experiments/results")
    return p.parse_args()


def _abs(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _sha(path: Path):
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _references_info(cfg: dict) -> dict:
    """Whether the run's test-gold file exists on disk (informational only — the
    legacy backfill does NOT use references, to avoid producing corrected numbers)."""
    data = cfg.get("data", {}) or {}
    test_file = data.get("test_file")
    info = {"test_file": test_file, "resolved": None, "exists": False}
    if test_file:
        path = _abs(str(test_file))
        info["resolved"] = str(path)
        info["exists"] = path.exists()
    return info


def _augment_manifest(run_dir: Path, cfg: dict) -> List[str]:
    mpath = run_dir / "manifest.json"
    if not mpath.exists():
        return []
    manifest = read_json(mpath)
    added: List[str] = []
    additions = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "eval_tier": str(cfg.get("eval_tier", "internal-synthetic")),
        "layer_status": {"L1": "pending", "L2": "implemented",
                         "L3": "pending-data", "L4": "pending-ratings"},
    }
    for k, v in additions.items():
        if k not in manifest:
            manifest[k] = v
            added.append(k)
    if added:
        write_json(manifest, mpath)
    return added


def _payload_from(cfg: dict, run_dir: Path, n_results: int) -> tuple[dict, bool]:
    auto = run_dir / "metrics_auto.json"
    if auto.exists():
        return read_json(auto), True
    return {
        "experiment_id": cfg.get("experiment_id", run_dir.name),
        "method": str((cfg.get("method", {}) or {}).get("name", "")),
        "model": str((cfg.get("model", {}) or {}).get("name", "")),
        "seed": cfg.get("seed"),
        "n_predictions": n_results,
        "metrics": {},
    }, False


def process_run(run_dir: Path) -> dict:
    res = {"run": run_dir.name, "status": None, "missing": [], "manifest_keys_added": [],
           "byte_identical": None, "n_items": None, "references_present": False,
           "metrics_auto_present": False}

    cfg = read_yaml(run_dir / "config_snapshot.yaml")
    pre = {f: _sha(run_dir / f) for f in RAW_FILES}

    # Load WITHOUT reparsing: keep the run-time stored parse state (legacy).
    results = [GenerationResult(**r) for r in read_jsonl(run_dir / "predictions.jsonl")]
    refinfo = _references_info(cfg)
    res["references_present"] = refinfo["exists"]

    payload, auto_present = _payload_from(cfg, run_dir, len(results))
    res["metrics_auto_present"] = auto_present
    if not auto_present:
        res["missing"].append(str(run_dir / "metrics_auto.json"))

    per_item = legacy_per_item(results)
    write_jsonl(per_item, run_dir / "eval_per_item.jsonl")

    metrics_json = build_metrics_json(payload, per_item, compute_ci=False)
    mark_backfill(metrics_json, metrics_auto_present=auto_present, references_present=refinfo["exists"])
    write_json(metrics_json, run_dir / "metrics.json")

    res["manifest_keys_added"] = _augment_manifest(run_dir, cfg)

    # Validation checks.
    res["n_items"] = len(results)
    assert len(per_item) == len(results), "eval_per_item line count != predictions"
    assert _LAYER_KEYS <= set(metrics_json["layers"]), "metrics.json missing layer keys"
    post = {f: _sha(run_dir / f) for f in RAW_FILES}
    res["byte_identical"] = all(pre[f] == post[f] for f in RAW_FILES)
    res["status"] = "legacy-carry-forward"
    return res


def _md_table(rows: List[dict], columns: List[str]) -> str:
    head = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in columns) + " |"
            for r in rows]
    return "\n".join([head, sep, *body])


def _select_cols(df):
    chosen = [c for c in ("experiment_id", "method", "model", "seed") if c in df.columns]
    labels = {c: c for c in chosen}
    for needle, label in _REPORT_METRIC_COLS:
        match = next((c for c in df.columns if needle in c), None)
        if match is not None and match not in chosen:
            chosen.append(match)
            labels[match] = label
    return chosen, labels


def main() -> None:
    args = parse_args()
    outputs_root = _abs(args.outputs_root)
    out_dir = _abs(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not outputs_root.exists():
        raise SystemExit(f"Outputs root not found: {outputs_root}")

    processed, skipped = [], []
    for run_dir in sorted(p for p in outputs_root.iterdir() if p.is_dir()):
        has_cfg = (run_dir / "config_snapshot.yaml").exists()
        has_pred = (run_dir / "predictions.jsonl").exists()
        if not (has_cfg and has_pred):
            skipped.append({"run": run_dir.name,
                            "reason": "missing config_snapshot.yaml and/or predictions.jsonl "
                                      "(left untouched; possibly a nested/stray folder)"})
            continue
        processed.append(process_run(run_dir))

    # Aggregate point values from existing metrics_auto.json (no references needed).
    df = aggregate(outputs_root, out_dir / "comparison_table.csv")
    seeds_written = False
    report_rows, report_cols = [], []
    if not df.empty:
        import pandas as pd

        cols, labels = _select_cols(df)
        report_cols = [labels[c] for c in cols]
        report_rows = [{labels[c]: row.get(c) for c in cols} for row in df.to_dict(orient="records")]
        group_keys = [c for c in ("model", "method") if c in df.columns]
        metric_cols = [c for c in cols if c not in ("experiment_id", "method", "model", "seed")]
        if group_keys and metric_cols:
            for c in metric_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            agg = df.groupby(group_keys)[metric_cols].agg(["mean", "std", "count"])
            agg.columns = [f"{a}_{b}" for a, b in agg.columns]
            agg.reset_index().to_csv(out_dir / "comparison_seeds.csv", index=False)
            seeds_written = True

    # final_report.md (mandated header note + legacy diagnostic table).
    final_md = (
        "# Experiment comparison report (backfilled)\n\n"
        f"> {HEADER_NOTE}\n\n"
        f"Backfilled from `{args.outputs_root}` — {len(processed)} run(s). "
        "All numbers are carried from existing `metrics_auto.json` (legacy, pre-Task-7) and are "
        "**internal-synthetic diagnostics only**. `top1%` is synthetic (circular) — not a validity "
        "claim. L1 human-effectiveness / L3 realism / L4 human evaluation are pending.\n\n"
        "## Per-run (legacy synthetic diagnostics)\n\n"
        + (_md_table(report_rows, report_cols) if report_rows else "_No metrics_auto.json found._")
        + "\n"
    )
    (out_dir / "final_report.md").write_text(final_md, encoding="utf-8")

    # backfill_report.md (provenance + missing artifacts + validation).
    missing_lines = []
    for r in processed:
        if r["missing"]:
            missing_lines.append(f"- `{r['run']}`: " + "; ".join(r["missing"]))
    lines = [
        "# Backfill report\n",
        f"> {HEADER_NOTE}\n",
        "## Provenance (legacy carry-forward)",
        "- **Point values** (parse/schema/completeness/top-1/macro-F1) in `metrics.json`: "
        "re-presented from each run's `metrics_auto.json` (legacy, pre-Task-7; internal-synthetic "
        "diagnostic). No fresh metric numbers were computed.",
        "- **`eval_per_item.jsonl`**: run-time STORED fields only "
        "(`parse_error`, `parsed`, `predicted_primary_chart`, `n_distinct_recs`).",
        "- **not_available (deferred to a corrected re-scoring task):** per-item `schema_valid`, "
        "`completeness`, `gold_primary_chart`, `synthetic_top1_correct`, and **all CIs** — computing "
        "them now would pair legacy values with current-code results.\n",
        "## Processed runs",
        "| run | status | n_items | metrics_auto | refs_on_disk | manifest_keys_added | raw_byte_identical |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in processed:
        lines.append(
            f"| {r['run']} | {r['status']} | {r['n_items']} | {r['metrics_auto_present']} | "
            f"{r['references_present']} | {','.join(r['manifest_keys_added']) or '-'} | "
            f"{r['byte_identical']} |"
        )
    lines.append("\n## Skipped (left untouched)")
    if skipped:
        for s in skipped:
            lines.append(f"- `{s['run']}`: {s['reason']}")
    else:
        lines.append("- none")
    lines.append("\n## Missing artifacts among processed runs (exact paths)")
    lines.extend(missing_lines or ["- none"])
    lines.append(
        "\n## Inputs for the deferred corrected re-scoring task"
        "\n- References (`data/processed/test.jsonl` etc.): presence per run is shown as "
        "`refs_on_disk` above; they are intentionally NOT used here (using them would produce "
        "corrected numbers)."
        "\n- Per-run `predictions_paraphrased.jsonl` / `predictions_missing_info.jsonl` are absent in "
        "the flat runs, so robustness re-derivation is not possible in a later re-scoring either "
        "unless those variant predictions are regenerated."
    )
    (out_dir / "backfill_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    all_identical = all(r["byte_identical"] for r in processed) if processed else True
    print("=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"  processed runs : {[r['run'] for r in processed]}")
    print(f"  skipped        : {[s['run'] for s in skipped]}")
    print(f"  comparison_table.csv : {out_dir / 'comparison_table.csv'}")
    print(f"  comparison_seeds.csv : {'written' if seeds_written else 'not written (no grouping)'}")
    print(f"  final_report.md      : {out_dir / 'final_report.md'}")
    print(f"  backfill_report.md   : {out_dir / 'backfill_report.md'}")
    print(f"  raw files byte-identical: {all_identical}")
    print("=" * 60)


if __name__ == "__main__":
    main()
