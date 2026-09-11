"""Thesis-level analysis of the consolidated final runs.

Reads the per-item evaluation records and run manifests under
``experiments/outputs/final/<dataset>`` and writes the artifacts that the
results chapter cites:

* ``run_provenance.csv``          — protocol identity per run (git, config,
                                    training-config and knowledge-base hashes)
* ``aggregate_by_model_method.csv`` / ``.md`` — descriptive per-condition table
* ``paired_tests.json`` / ``.md``  — Cochran's Q, exact McNemar with Holm
                                    correction and paired bootstrap CIs
* ``method_contrasts.csv``        — the B-A, C-A, D-C, D-B contrasts
* ``strict_schema_ceiling.json``  — how many frozen reference records satisfy
                                    the strict response schema

Nothing here re-runs a model. It only aggregates artifacts that already exist,
so it can be executed on a laptop.

Usage::

    python experiments/scripts/analyze_final_results.py --dataset dashboard_v4
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.stats import (  # noqa: E402
    cochran_q,
    paired_bootstrap_diff,
    pairwise_mcnemar,
    per_method_bootstrap_cis,
)

METHOD_ORDER = ["A", "B", "C", "D"]
CONTRASTS = [("B", "A"), ("C", "A"), ("D", "C"), ("D", "B"), ("D", "A")]

# Binary per-item outcomes that support McNemar / Cochran's Q.
BINARY_FIELDS = {
    # Chart agreement as scored by the pipeline (requires a parseable record).
    "top1_correct": "synthetic_top1_correct",
    # A JSON object was recoverable from the raw text. Recomputed from the
    # stored raw response so that runs of different vintages are comparable
    # (older per-item records do not carry this flag).
    "json_object_recovered": "json_object_recovered",
    # The recovered object also satisfied the lenient runtime contract.
    "runtime_object_parsed": "parsed",
    # The raw object satisfied the strict response contract.
    "strict_schema_valid": "schema_valid",
    # Every emitted mapping carries an object-valued encoding. Recomputed from
    # the stored raw response for the same reason.
    "encoding_object_recovered": "encoding_object_recovered",
    # Chart agreement re-scored from the raw JSON object, ignoring whether the
    # surrounding record satisfies the runtime contract (see
    # ``format_tolerant_rescore``).
    "format_tolerant_chart_correct": "format_tolerant_chart_correct",
    # Chart agreement read from the first ``chart_type`` token in the raw text,
    # which also survives a generation that was cut off by the token budget.
    "text_level_chart_correct": "text_level_chart_correct",
}

# First ``"chart_type": "<token>"`` occurrence in a raw model response.
CHART_TYPE_PATTERN = re.compile(r'"chart_type"\s*:\s*"([A-Za-z_]+)"')


def comparability(runs_by_method: dict[str, dict]) -> dict:
    """Describe how far a set of runs shares one executable protocol.

    ``homogeneous``       one known git state for every run, and one training
                          configuration for the trained methods
    ``mixed_code_state``  the runs come from more than one known git state
    ``unknown_code_state`` at least one run did not record its git commit
    """
    git_states = {r["git_hash"] for r in runs_by_method.values()}
    train_states = {
        r["training_config_hash"]
        for r in runs_by_method.values()
        if r["training_config_hash"] is not None
    }
    if "unknown" in git_states:
        label = "unknown_code_state"
    elif len(git_states) > 1 or len(train_states) > 1:
        label = "mixed_code_state"
    else:
        label = "homogeneous"
    return {
        "label": label,
        "homogeneous": label == "homogeneous",
        "git_states": sorted(git_states),
        "training_config_states": sorted(train_states),
    }


def _get(d: dict, *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return None


def load_gold_primary_charts(dataset_dir: Path) -> dict[str, str]:
    """Primary reference chart token per held-out item."""
    from src.evaluation.metrics.base import chart_token

    gold: dict[str, str] = {}
    test_path = dataset_dir / "test.jsonl"
    if not test_path.exists():
        return gold
    with test_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            mapping = (record.get("recommendation") or {}).get("kpi_chart_mapping") or []
            if mapping and isinstance(mapping[0], dict) and mapping[0].get("chart_type"):
                gold[record["item_id"]] = chart_token(mapping[0]["chart_type"])
    return gold


def attach_format_tolerant_outcome(run: dict, gold: dict[str, str]) -> None:
    """Add a per-item format-tolerant chart outcome to ``run['items']``."""
    from src.evaluation.metrics.base import chart_token
    from src.evaluation.metrics.schema_compliance import encoding_objects_valid
    from src.inference.postprocess import extract_json_dict

    predictions_path = run["run_dir"] / "predictions.jsonl"
    if not predictions_path.exists() or not gold:
        return
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("variant") not in (None, "original"):
                continue
            item = run["items"].get(record["item_id"])
            if item is None:
                continue
            raw = record.get("raw_text") or ""
            correct = 0
            obj = extract_json_dict(raw)
            mapping = (obj or {}).get("kpi_chart_mapping")
            if isinstance(mapping, list) and mapping and isinstance(mapping[0], dict):
                token = mapping[0].get("chart_type")
                if isinstance(token, str) and token.strip():
                    correct = int(chart_token(token) == gold.get(record["item_id"]))
            item["format_tolerant_chart_correct"] = correct

            match = CHART_TYPE_PATTERN.search(raw)
            item["text_level_chart_correct"] = int(
                bool(match) and chart_token(match.group(1)) == gold.get(record["item_id"])
            )
            item["chart_token_in_text"] = int(bool(match))
            item["json_object_recovered"] = int(bool(obj))
            item["encoding_object_recovered"] = int(bool(obj) and encoding_objects_valid(obj))
            # A response that does not close its outermost object was cut off by
            # the generation budget rather than completed by the model.
            item["truncated_generation"] = int(not raw.rstrip().endswith("}"))
    for item in run["items"].values():
        item.setdefault("format_tolerant_chart_correct", 0)
        item.setdefault("text_level_chart_correct", 0)
        item.setdefault("chart_token_in_text", 0)
        item.setdefault("truncated_generation", 0)
        item.setdefault("json_object_recovered", 0)
        item.setdefault("encoding_object_recovered", 0)


def load_runs(final_dir: Path, gold_charts: dict[str, str] | None = None) -> list[dict]:
    """Collect one record per completed run under ``final_dir``."""
    runs: list[dict] = []
    for manifest_path in sorted(final_dir.glob("*/*/seed_*/manifest.json")):
        run_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        per_item_path = run_dir / "eval_per_item.jsonl"
        metrics_path = run_dir / "metrics_auto.json"
        if not per_item_path.exists() or not metrics_path.exists():
            continue

        items: dict[str, dict] = {}
        with per_item_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("variant") not in (None, "original"):
                    continue
                items[record["item_id"]] = record

        adapter_meta = _get(manifest, "adapter", "adapter_training_metadata", default={}) or {}
        runs.append(
            {
                "run_dir": run_dir,
                "model_key": manifest.get("model_key"),
                "method_key": manifest.get("method_key"),
                "seed": manifest.get("seed"),
                "model": manifest.get("model"),
                "model_revision": manifest.get("model_revision"),
                "git_hash": (manifest.get("git_hash") or "unknown")[:12],
                "config_hash": manifest.get("config_hash"),
                "training_config_hash": adapter_meta.get("training_config_hash"),
                "model_config_hash": adapter_meta.get("model_config_hash"),
                "kb_version": _get(manifest, "knowledge_base", "kb_version"),
                "kb_chunks_sha256": _get(manifest, "knowledge_base", "chunks_sha256"),
                "test_sha256": _get(manifest, "dataset_hashes", "test"),
                "train_sha256": _get(manifest, "dataset_hashes", "train"),
                "source_c_run_id": manifest.get("source_c_run_id"),
                "adapter_path": manifest.get("adapter_path"),
                "lora_r": adapter_meta.get("lora_r"),
                "lora_alpha": adapter_meta.get("lora_alpha"),
                "epochs": adapter_meta.get("num_train_epochs"),
                "learning_rate": adapter_meta.get("learning_rate"),
                "batch_size": adapter_meta.get("per_device_train_batch_size"),
                "grad_accum": adapter_meta.get("gradient_accumulation_steps"),
                "max_seq_length": adapter_meta.get("max_seq_length"),
                "quantization": adapter_meta.get("quantization"),
                "trainable_parameters": adapter_meta.get("trainable_parameters"),
                "total_parameters": adapter_meta.get("total_parameters"),
                "training_seconds": adapter_meta.get("training_duration_seconds"),
                "precision": _get(adapter_meta, "precision", "mode"),
                "train_gpu": (_get(adapter_meta, "precision", "devices") or [None])[0],
                "infer_gpu": _get(manifest, "hardware", "gpu_name"),
                "run_seconds": manifest.get("duration_seconds"),
                "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
                "items": items,
            }
        )
        if gold_charts:
            attach_format_tolerant_outcome(runs[-1], gold_charts)
    return runs


def write_provenance(runs: list[dict], out_dir: Path) -> None:
    columns = [
        "model_key", "method_key", "seed", "model", "model_revision", "git_hash",
        "config_hash", "training_config_hash", "model_config_hash", "kb_version",
        "kb_chunks_sha256", "test_sha256", "train_sha256", "source_c_run_id",
        "lora_r", "lora_alpha", "epochs", "learning_rate", "batch_size",
        "grad_accum", "max_seq_length", "quantization", "trainable_parameters",
        "total_parameters", "training_seconds", "precision", "train_gpu",
        "infer_gpu", "run_seconds",
    ]
    path = out_dir / "run_provenance.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for run in sorted(runs, key=lambda r: (r["model_key"], r["method_key"], r["seed"])):
            writer.writerow(run)


def _metric(run: dict, *keys: str) -> Any:
    return _get(run["metrics"], "metrics", *keys)


def aggregate_table(runs: list[dict], out_dir: Path) -> list[dict]:
    """Descriptive mean/sd per (model, method) across the seeds actually present."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[(run["model_key"], run["method_key"])].append(run)

    metric_specs = [
        ("json_parse_rate", lambda r: _metric(r, "schema_compliance", "json_parse_rate")),
        ("strict_schema_validity_rate", lambda r: _metric(r, "schema_compliance", "schema_validity_rate")),
        ("encoding_object_rate", lambda r: _metric(r, "schema_compliance", "encoding_object_rate")),
        ("completeness_score", lambda r: _metric(r, "schema_compliance", "completeness_score")),
        ("top_1_accuracy", lambda r: _metric(r, "top_k_accuracy", "top_1_accuracy")),
        ("macro_f1", lambda r: _metric(r, "macro_f1", "macro_f1")),
        ("exact_task_classification", lambda r: _metric(r, "structured_exact_match", "exact_task_classification")),
        ("exact_encoding", lambda r: _metric(r, "structured_exact_match", "exact_encoding")),
        ("exact_encoding_strict", lambda r: _metric(r, "structured_exact_match", "exact_encoding_strict")),
        ("paraphrase_consistency", lambda r: _metric(r, "robustness", "paraphrase_consistency")),
        ("paraphrase_accuracy", lambda r: _metric(r, "robustness", "paraphrase_accuracy")),
        ("missing_info_clarification_rate", lambda r: _metric(r, "robustness", "missing_info_clarification_rate")),
        ("avg_latency_ms", lambda r: _metric(r, "latency", "avg_latency_ms")),
    ]

    rows: list[dict] = []
    for (model_key, method_key), group in sorted(grouped.items()):
        group = sorted(group, key=lambda r: r["seed"])
        comp = comparability({str(r["seed"]): r for r in group})
        for name, getter in metric_specs:
            values = [getter(r) for r in group]
            present = [v for v in values if isinstance(v, (int, float))]
            rows.append(
                {
                    "model_key": model_key,
                    "method_key": method_key,
                    "metric": name,
                    "n_seeds": len(group),
                    "seeds": ",".join(str(r["seed"]) for r in group),
                    "values": ",".join("NA" if v is None else f"{v:g}" for v in values),
                    "mean": round(statistics.fmean(present), 4) if present else None,
                    "sd": round(statistics.stdev(present), 4) if len(present) > 1 else None,
                    "seed_comparability": comp["label"],
                    "git_states": ";".join(comp["git_states"]),
                }
            )

    path = out_dir / "aggregate_by_model_method.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Descriptive aggregation by model and method",
        "",
        "Seeds are repeated runs of the same 274 held-out items, not extra items.",
        "`sd` is the sample standard deviation across the seeds actually present and",
        "is left empty for single-seed conditions. `seed comparability` reports whether",
        "the seeds of one condition were produced by one recorded code state.",
        "",
        "| model | method | metric | n_seeds | seeds | per-seed values | mean | sd | seed comparability |",
        "| --- | --- | --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {model_key} | {method_key} | {metric} | {n_seeds} | {seeds} | {values} | "
            "{mean} | {sd} | {seed_comparability} |".format(
                **{k: ("" if v is None else v) for k, v in row.items()}
            )
        )
    (out_dir / "aggregate_by_model_method.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def paired_analysis(runs: list[dict], out_dir: Path) -> dict:
    """Cochran's Q, exact McNemar with Holm correction and bootstrap CIs."""
    by_model_seed: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for run in runs:
        by_model_seed[(run["model_key"], run["seed"])][run["method_key"]] = run

    report: dict[str, Any] = {
        "unit": "one held-out dashboard brief",
        "pairing": "same item_id across methods for one model and one seed",
        "tests": {
            "omnibus": "Cochran's Q over the methods present (requires k >= 3)",
            "pairwise": "exact McNemar on discordant pairs, Holm-corrected within each family",
            "interval": "percentile bootstrap CI of the mean paired difference, 10,000 resamples",
        },
        "results": [],
    }
    contrast_rows: list[dict] = []

    for (model_key, seed), methods in sorted(by_model_seed.items()):
        available = [m for m in METHOD_ORDER if m in methods]
        if len(available) < 2:
            continue
        shared = set.intersection(*(set(methods[m]["items"]) for m in available))
        item_ids = sorted(shared)
        if not item_ids:
            continue

        protocol = {
            m: {
                "git_hash": methods[m]["git_hash"],
                "config_hash": methods[m]["config_hash"],
                "training_config_hash": methods[m]["training_config_hash"],
            }
            for m in available
        }

        for outcome_name, field in BINARY_FIELDS.items():
            vectors: dict[str, list[int]] = {}
            incomplete = False
            for method in available:
                values = [_as_int(methods[method]["items"][i].get(field)) for i in item_ids]
                if any(v is None for v in values):
                    # The outcome is not recorded for this method. Skipping only
                    # that method would silently change the comparison set, so
                    # the whole outcome is skipped for this block.
                    incomplete = True
                    break
                vectors[method] = values
            if incomplete or len(vectors) < 2:
                continue

            comp = comparability({m: methods[m] for m in vectors})
            entry: dict[str, Any] = {
                "model_key": model_key,
                "seed": seed,
                "outcome": outcome_name,
                "n_items": len(item_ids),
                "methods": list(vectors),
                "protocol_identity": {m: protocol[m] for m in vectors},
                "comparability": comp,
                "rates_percent": {m: round(100 * sum(v) / len(v), 2) for m, v in vectors.items()},
                "bootstrap_ci_percent": per_method_bootstrap_cis(vectors, scale=100.0),
            }
            if len(vectors) >= 3:
                entry["cochran_q"] = cochran_q(vectors)
            entry["mcnemar_holm"] = pairwise_mcnemar(vectors)

            paired = []
            for a, b in CONTRASTS:
                if a not in vectors or b not in vectors:
                    continue
                diff = paired_bootstrap_diff(vectors[a], vectors[b])
                mcnemar = next(
                    (
                        r
                        for r in entry["mcnemar_holm"]
                        if {r["method_a"], r["method_b"]} == {a, b}
                    ),
                    None,
                )
                pair_comp = comparability({a: methods[a], b: methods[b]})
                record = {
                    "contrast": f"{a}-{b}",
                    "comparability": pair_comp["label"],
                    "git_states": pair_comp["git_states"],
                    "mean_diff_percent": round(100 * diff["mean_diff"], 2),
                    "ci_low_percent": round(100 * diff["ci_low"], 2),
                    "ci_high_percent": round(100 * diff["ci_high"], 2),
                    "n_discordant": mcnemar["n_discordant"] if mcnemar else None,
                    "p_exact": mcnemar["p_value"] if mcnemar else None,
                    "p_holm": mcnemar["p_holm"] if mcnemar else None,
                    "significant_holm_0_05": mcnemar["reject_h0"] if mcnemar else None,
                }
                paired.append(record)
                contrast_rows.append(
                    {
                        "model_key": model_key,
                        "seed": seed,
                        "outcome": outcome_name,
                        "block_comparability": comp["label"],
                        "n_items": len(item_ids),
                        **record,
                    }
                )
            entry["contrasts"] = paired
            report["results"].append(entry)

    (out_dir / "paired_tests.json").write_text(
        json.dumps(report, indent=2, sort_keys=False), encoding="utf-8"
    )

    if contrast_rows:
        path = out_dir / "method_contrasts.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(contrast_rows[0]))
            writer.writeheader()
            writer.writerows(contrast_rows)

    lines = [
        "# Paired method comparisons on the held-out split",
        "",
        "Each block compares methods on the same items for one model and one seed.",
        "`p_holm` is Holm-corrected within the set of pairwise tests for that block.",
        "`comparability` is reported twice: once for the whole block (all methods in",
        "the omnibus test) and once for each pair actually contrasted. A pair labelled",
        "`homogeneous` came from one recorded code state and one training configuration;",
        "`mixed_code_state` or `unknown_code_state` means something besides the method",
        "differs between the two runs, so that contrast is descriptive only.",
        "",
    ]
    for entry in report["results"]:
        lines.append(
            f"## {entry['model_key']} · seed {entry['seed']} · {entry['outcome']} "
            f"(n = {entry['n_items']}, comparability = {entry['comparability']['label']})"
        )
        lines.append("")
        rates = ", ".join(f"{m} = {v}%" for m, v in entry["rates_percent"].items())
        lines.append(f"Rates: {rates}")
        if "cochran_q" in entry and entry["cochran_q"].get("applicable"):
            q = entry["cochran_q"]
            lines.append(
                f"Cochran's Q = {q['statistic']:.2f}, df = {q['df']}, p = {q['p_value']:.3g}"
            )
        lines.append("")
        lines.append("| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |")
        lines.append("| --- | ---: | --- | ---: | ---: | ---: | --- | --- |")
        for c in entry["contrasts"]:
            lines.append(
                f"| {c['contrast']} | {c['mean_diff_percent']:+.2f} | "
                f"[{c['ci_low_percent']:+.2f}, {c['ci_high_percent']:+.2f}] | "
                f"{c['n_discordant']} | {c['p_exact']:.3g} | {c['p_holm']:.3g} | "
                f"{'yes' if c['significant_holm_0_05'] else 'no'} |"
            )
        lines.append("")
    (out_dir / "paired_tests.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def error_analysis(runs: list[dict], out_dir: Path) -> dict:
    """Failure-mode counts per condition: parse errors and chart confusions."""
    per_condition: list[dict] = []
    confusion_rows: list[dict] = []
    for run in sorted(runs, key=lambda r: (r["model_key"], r["method_key"], r["seed"])):
        items = list(run["items"].values())
        n = len(items)
        parse_errors: dict[str, int] = defaultdict(int)
        confusions: dict[tuple[str, str], int] = defaultdict(int)
        n_no_chart = 0
        n_alternatives = 0
        for record in items:
            if not record.get("parsed"):
                parse_errors[str(record.get("parse_error") or "unknown")] += 1
            predicted = record.get("predicted_primary_chart")
            gold = record.get("gold_primary_chart")
            if predicted in (None, ""):
                n_no_chart += 1
            if (record.get("n_distinct_recs") or 0) > 1:
                n_alternatives += 1
            if predicted and gold and predicted != gold:
                confusions[(str(gold), str(predicted))] += 1
        per_condition.append(
            {
                "model_key": run["model_key"],
                "method_key": run["method_key"],
                "seed": run["seed"],
                "n_items": n,
                "n_parse_failures": sum(parse_errors.values()),
                "parse_error_types": dict(sorted(parse_errors.items(), key=lambda kv: -kv[1])),
                "n_without_primary_chart": n_no_chart,
                "n_items_with_more_than_one_distinct_recommendation": n_alternatives,
                "top_confusions": [
                    {"gold": g, "predicted": p_, "count": c}
                    for (g, p_), c in sorted(confusions.items(), key=lambda kv: -kv[1])[:8]
                ],
            }
        )
        for (gold, predicted), count in confusions.items():
            confusion_rows.append(
                {
                    "model_key": run["model_key"],
                    "method_key": run["method_key"],
                    "seed": run["seed"],
                    "gold_primary_chart": gold,
                    "predicted_primary_chart": predicted,
                    "count": count,
                }
            )

    (out_dir / "error_analysis.json").write_text(
        json.dumps({"conditions": per_condition}, indent=2), encoding="utf-8"
    )
    if confusion_rows:
        path = out_dir / "chart_confusions.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(confusion_rows[0]))
            writer.writeheader()
            writer.writerows(sorted(confusion_rows, key=lambda r: -r["count"]))

    lines = [
        "# Failure modes per condition",
        "",
        "`parse failures` counts items whose raw text could not be turned into the",
        "runtime object. `no primary chart` counts items with no readable primary",
        "recommendation, which the top-1 metric scores as incorrect.",
        "",
        "| model | method | seed | n | parse failures | dominant parse error | no primary chart | items with >1 distinct recommendation |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for c in per_condition:
        dominant = next(iter(c["parse_error_types"]), "-")
        lines.append(
            f"| {c['model_key']} | {c['method_key']} | {c['seed']} | {c['n_items']} | "
            f"{c['n_parse_failures']} | {dominant} | {c['n_without_primary_chart']} | "
            f"{c['n_items_with_more_than_one_distinct_recommendation']} |"
        )
    (out_dir / "error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"conditions": per_condition}


def format_tolerant_rescore(runs: list[dict], dataset_dir: Path, out_dir: Path) -> dict:
    """Re-score the primary chart token directly from the raw JSON object.

    The pipeline's top-1 metric reads the chart from the parsed runtime object.
    An output whose ``encoding`` field is a string rather than an object fails
    that parse, so its chart token is never scored and the item counts as
    incorrect. This function repeats the comparison on the raw extracted object
    and therefore separates *chart-token agreement* from *output-format
    validity*. Both numbers are reported; the pipeline value stays the headline.
    """
    from src.evaluation.metrics.base import chart_token
    from src.inference.postprocess import extract_json_dict

    test_path = dataset_dir / "test.jsonl"
    gold: dict[str, str] = {}
    with test_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            mapping = (record.get("recommendation") or {}).get("kpi_chart_mapping") or []
            if mapping and isinstance(mapping[0], dict) and mapping[0].get("chart_type"):
                gold[record["item_id"]] = chart_token(mapping[0]["chart_type"])

    rows: list[dict] = []
    for run in sorted(runs, key=lambda r: (r["model_key"], r["method_key"], r["seed"])):
        predictions_path = run["run_dir"] / "predictions.jsonl"
        if not predictions_path.exists():
            continue
        n = 0
        n_json = 0
        n_token = 0
        n_match = 0
        n_text_match = 0
        n_truncated = 0
        with predictions_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("variant") not in (None, "original"):
                    continue
                n += 1
                raw = record.get("raw_text") or ""
                if not raw.rstrip().endswith("}"):
                    n_truncated += 1
                match = CHART_TYPE_PATTERN.search(raw)
                if match and chart_token(match.group(1)) == gold.get(record["item_id"]):
                    n_text_match += 1
                obj = extract_json_dict(raw)
                if not obj:
                    continue
                n_json += 1
                mapping = obj.get("kpi_chart_mapping")
                if not (isinstance(mapping, list) and mapping and isinstance(mapping[0], dict)):
                    continue
                token = mapping[0].get("chart_type")
                if not isinstance(token, str) or not token.strip():
                    continue
                n_token += 1
                if chart_token(token) == gold.get(record["item_id"]):
                    n_match += 1
        pipeline_top1 = _metric(run, "top_k_accuracy", "top_1_accuracy")
        rows.append(
            {
                "model_key": run["model_key"],
                "method_key": run["method_key"],
                "seed": run["seed"],
                "n_items": n,
                "n_json_object_extracted": n_json,
                "n_chart_token_present": n_token,
                "pipeline_top_1_accuracy": pipeline_top1,
                "n_truncated_generations": n_truncated,
                "format_tolerant_chart_accuracy": round(100 * n_match / n, 2) if n else None,
                "text_level_chart_accuracy": round(100 * n_text_match / n, 2) if n else None,
                "gap_percentage_points": (
                    round(100 * n_text_match / n - pipeline_top1, 2)
                    if n and isinstance(pipeline_top1, (int, float))
                    else None
                ),
            }
        )

    path = out_dir / "format_tolerant_chart_accuracy.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Chart-token agreement with and without the output-format requirement",
        "",
        "`pipeline top-1` is the metric produced by the evaluation code: it reads the",
        "chart from the parsed runtime object, so an item whose output fails the",
        "runtime contract is scored as incorrect.",
        "`format-tolerant chart accuracy` repeats the same comparison on the raw",
        "extracted JSON object and ignores whether the surrounding record satisfies the",
        "runtime contract.",
        "`text-level chart accuracy` reads the first `chart_type` token out of the raw",
        "text, so it also survives a response that the token budget cut off.",
        "`truncated` counts responses whose outermost object was never closed.",
        "A large pipeline-to-text gap means the condition failed the output contract or",
        "the length budget rather than the chart decision.",
        "",
        "| model | method | seed | n | JSON object found | truncated | pipeline top-1 (%) | format-tolerant (%) | text-level (%) | gap pipeline->text (pp) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['model_key']} | {r['method_key']} | {r['seed']} | {r['n_items']} | "
            f"{r['n_json_object_extracted']} | {r['n_truncated_generations']} | "
            f"{r['pipeline_top_1_accuracy']} | {r['format_tolerant_chart_accuracy']} | "
            f"{r['text_level_chart_accuracy']} | {r['gap_percentage_points']} |"
        )
    (out_dir / "format_tolerant_chart_accuracy.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return {"rows": rows}


def strict_schema_ceiling(dataset_dir: Path, out_dir: Path) -> dict:
    """Check how many frozen reference recommendations satisfy the strict schema."""
    from pydantic import ValidationError

    from src.core.strict_response_schema import StrictDesignOutput

    test_path = dataset_dir / "test.jsonl"
    if not test_path.exists():
        return {"available": False, "reason": f"{test_path} not found"}

    total = 0
    passed = 0
    error_counts: dict[str, int] = defaultdict(int)
    with test_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            total += 1
            try:
                StrictDesignOutput.model_validate(record["recommendation"])
                passed += 1
            except ValidationError as exc:
                for err in exc.errors():
                    key = f"{err['type']}:{'.'.join(str(p) for p in err['loc'])}"
                    error_counts[key] += 1

    result = {
        "available": True,
        "test_file": str(test_path).replace("\\", "/"),
        "n_reference_records": total,
        "n_passing_strict_schema": passed,
        "pass_rate_percent": round(100 * passed / total, 2) if total else None,
        "top_violations": dict(sorted(error_counts.items(), key=lambda kv: -kv[1])[:20]),
        "interpretation": (
            "The strict response schema forbids extra keys. The frozen reference "
            "encoding objects carry additional source-grounded keys, so the reference "
            "itself does not satisfy the strict contract. Strict schema validity is "
            "therefore a diagnostic of output shape relative to that narrow contract, "
            "not a measure of agreement with the reference."
        ),
    }
    (out_dir / "strict_schema_ceiling.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dashboard_v4")
    parser.add_argument("--outputs-root", default="experiments/outputs/final")
    parser.add_argument("--results-root", default="experiments/results/final")
    args = parser.parse_args()

    final_dir = _PROJECT_ROOT / args.outputs_root / args.dataset
    out_dir = _PROJECT_ROOT / args.results_root / args.dataset / "statistics"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = _PROJECT_ROOT / "data" / "frozen" / args.dataset
    gold_charts = load_gold_primary_charts(dataset_dir)
    runs = load_runs(final_dir, gold_charts)
    if not runs:
        raise SystemExit(f"no completed runs with per-item records under {final_dir}")
    print(f"loaded {len(runs)} runs from {final_dir}")

    write_provenance(runs, out_dir)
    aggregate_table(runs, out_dir)
    paired_analysis(runs, out_dir)
    error_analysis(runs, out_dir)
    format_tolerant_rescore(runs, dataset_dir, out_dir)
    ceiling = strict_schema_ceiling(dataset_dir, out_dir)
    if ceiling.get("available"):
        print(
            "strict-schema ceiling: "
            f"{ceiling['n_passing_strict_schema']}/{ceiling['n_reference_records']} "
            "reference records pass"
        )
    print(f"wrote analysis artifacts to {out_dir}")


if __name__ == "__main__":
    main()
