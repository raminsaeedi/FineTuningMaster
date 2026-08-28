"""Phase-1 reporting artifacts: a per-item scored join file and a layered metrics.json.

Additive and non-destructive: these helpers DERIVE new artifacts from the
predictions + references + the already-computed ``metrics_auto`` payload. They never
mutate ``predictions.jsonl``, ``errors.jsonl`` or ``metrics_auto.json``.

Scientific reporting conventions (see ``docs/evaluation/evaluation_protocol.md``):
  * Synthetic chart accuracy is INTERNAL / CIRCULAR -> diagnostic only, never a
    primary validity claim (tagged ``tier: internal-circular``).
  * The independent L1 human-effectiveness scorer is not implemented yet -> the L1
    human layer and the per-item ``l1_*`` fields are reported as not-applicable / pending.
  * L3 realism is pending-data; L4 human is pending-ratings.
Confidence intervals are computed ONLY where a per-item vector is available; otherwise
they are marked ``pending`` or ``not_available`` with a reason (never fabricated).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.core.constants import REPORT_SCHEMA_VERSION
from src.core.schemas import ChartType
from src.evaluation.metrics.base import (
    index_references,
    predicted_alternatives,
    predicted_charts,
    reference_charts,
)
from src.evaluation.metrics.schema_compliance import (
    completeness_fraction,
    encoding_objects_valid,
    strict_response_valid,
)
from src.evaluation.stats.bootstrap_ci import bootstrap_ci
from src.inference.postprocess import extract_json_dict
from src.utils.io import write_json, write_jsonl

NOT_APPLICABLE = "not_applicable"
NOT_AVAILABLE = "not_available"
_VALID_CHARTS = {c.value for c in ChartType}


# --------------------------------------------------------------------------- #
# Per-item scored rows (eval_per_item.jsonl)
# --------------------------------------------------------------------------- #
def score_per_item(results, references) -> List[dict]:
    """Per-item scored join rows derived from predictions + references.

    One row per prediction. ``synthetic_top1_correct`` is the INTERNAL/CIRCULAR
    correctness against the synthetic gold (diagnostic only). The independent L1
    human-effectiveness fields are ``not_applicable`` until that scorer exists.
    """
    ref_by_id = index_references(references or [])
    rows: List[dict] = []
    for r in results:
        ref = ref_by_id.get(r.item_id)
        gold_charts = reference_charts(ref) if ref else []
        gold_primary = gold_charts[0] if gold_charts else None

        preds = predicted_charts(r)
        pred_primary = preds[0] if preds else None

        # Distinct valid recommendations: primary + distinct valid alternatives.
        recs: List[str] = []
        if pred_primary is not None:
            recs = [pred_primary]
            for alt in predicted_alternatives(r, 0):
                if alt in _VALID_CHARTS and alt not in recs:
                    recs.append(alt)

        if gold_primary is None:
            synthetic_top1 = None  # no gold reference -> not scored
        elif pred_primary is None:
            synthetic_top1 = 0     # parse/empty prediction counts as wrong
        else:
            synthetic_top1 = int(pred_primary == gold_primary)

        obj = extract_json_dict(r.raw_text)
        rows.append(
            {
                "item_id": r.item_id,
                "method_name": r.method_name,
                "model_name": r.model_name,
                "seed": r.seed,
                "variant": r.variant,
                "config_hash": r.config_hash,
                "parsed": r.parsed is not None,
                "json_object_extracted": obj is not None,
                "parse_error": r.parse_error,
                "schema_valid": strict_response_valid(obj),
                "encoding_object_valid": encoding_objects_valid(obj),
                "completeness": round(completeness_fraction(obj), 4),
                "predicted_primary_chart": pred_primary,
                "gold_primary_chart": gold_primary,
                "n_distinct_recs": len(recs),
                # Internal/circular diagnostic — NOT a validity claim:
                "synthetic_top1_correct": synthetic_top1,
                # Independent L1 human-effectiveness scorer not implemented yet:
                "l1_covered": NOT_APPLICABLE,
                "l1_correct": NOT_APPLICABLE,
            }
        )
    return rows


def legacy_per_item(results) -> List[dict]:
    """Per-item rows for a LEGACY backfill: only run-time-stored fields are used.

    No references and no current-code metric recomputation (which would produce
    corrected numbers). Fields that would require re-scoring are marked
    ``not_available`` so this file never mixes legacy aggregates with fresh values.
    ``results`` MUST be loaded WITHOUT reparsing (so ``parsed``/``parse_error``
    reflect the run-time state, not the current parser).
    """
    rows: List[dict] = []
    for r in results:
        preds = predicted_charts(r)  # from the STORED parsed output (run-time)
        pred_primary = preds[0] if preds else None
        recs: List[str] = []
        if pred_primary is not None:
            recs = [pred_primary]
            for alt in predicted_alternatives(r, 0):
                if alt in _VALID_CHARTS and alt not in recs:
                    recs.append(alt)
        rows.append(
            {
                "item_id": r.item_id,
                "method_name": r.method_name,
                "model_name": r.model_name,
                "seed": r.seed,
                "variant": r.variant,
                "config_hash": r.config_hash,
                "parsed": r.parsed is not None,
                "parse_error": r.parse_error,
                "predicted_primary_chart": pred_primary,
                "n_distinct_recs": len(recs),
                # Not recomputed for a legacy backfill (would require current-code re-scoring):
                "schema_valid": NOT_AVAILABLE,
                "completeness": NOT_AVAILABLE,
                "gold_primary_chart": NOT_AVAILABLE,
                "synthetic_top1_correct": NOT_AVAILABLE,
                "l1_covered": NOT_APPLICABLE,
                "l1_correct": NOT_APPLICABLE,
                "source": "run-time stored prediction",
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Layered metrics.json (value / n / ci)
# --------------------------------------------------------------------------- #
def _ci_from_vector(values: Sequence[Optional[float]], scale: float = 1.0) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"status": "not_available", "reason": "no per-item values"}
    ci = bootstrap_ci(vals)
    return {
        "ci_low": round(ci["ci_low"] * scale, 4),
        "ci_high": round(ci["ci_high"] * scale, 4),
        "ci_level": ci["ci_level"],
        "ci_method": "percentile_bootstrap",
        "n": ci["n"],
    }


def _entry(value: Any, n: Any, ci: Any) -> dict:
    return {"value": value, "n": n, "ci": ci}


def _pending(reason: str) -> dict:
    return {"status": "pending", "reason": reason}


def _not_available(reason: str) -> dict:
    return {"status": "not_available", "reason": reason}


_ROBUSTNESS_CI_REASON = "per-variant per-item vectors not persisted in Phase 1"


def build_metrics_json(payload: dict, per_item_rows: List[dict], compute_ci: bool = True) -> dict:
    """Assemble the additive, layer-grouped ``metrics.json`` with value/n/CI.

    ``payload`` is the existing ``metrics_auto`` dict (authoritative point values).
    When ``compute_ci`` is True, ``per_item_rows`` supply vectors for honest bootstrap
    CIs (live eval, same code as the point values). When ``compute_ci`` is False
    (backfill of legacy runs), **no** fresh CI is computed — pairing a legacy point
    value with a current-code CI would be inconsistent — so every CI is marked
    ``not_available`` with a reason.
    """
    m = payload.get("metrics", {}) or {}
    sc = m.get("schema_compliance", {}) or {}
    tk = m.get("top_k_accuracy", {}) or {}
    mf = m.get("macro_f1", {}) or {}
    rob = m.get("robustness", {}) or {}
    gr = m.get("grounding", {}) or {}
    rr = m.get("retrieval_relevance", {}) or {}

    if compute_ci:
        parsed_vec = [1 if row["json_object_extracted"] else 0 for row in per_item_rows]
        schema_vec = [1 if row["schema_valid"] else 0 for row in per_item_rows]
        encoding_object_vec = [
            1 if row["encoding_object_valid"] else 0 for row in per_item_rows
        ]
        complete_vec = [row["completeness"] for row in per_item_rows]
        synth_top1_vec = [
            row["synthetic_top1_correct"]
            for row in per_item_rows
            if row["synthetic_top1_correct"] is not None
        ]
        jpr_ci = _ci_from_vector(parsed_vec, 100.0)
        sv_ci = _ci_from_vector(schema_vec, 100.0)
        encoding_object_ci = _ci_from_vector(encoding_object_vec, 100.0)
        comp_ci = _ci_from_vector(complete_vec, 1.0)
        st1_ci = _ci_from_vector(synth_top1_vec, 100.0)
    else:
        _bf = _not_available(
            "backfill: CI not recomputed to avoid pairing a legacy point value with a "
            "current-code CI; corrected re-scoring is a separate task"
        )
        jpr_ci = sv_ci = encoding_object_ci = comp_ci = st1_ci = _bf

    gr_applicable = bool(gr) and gr.get("mode") is not None and (gr.get("n") or 0) > 0
    if rr.get("available") is True:
        rr_cis = rr.get("confidence_intervals") or {}
        retrieval_layer = {
            "recall_at_3": _entry(
                rr.get("recall_at_3"), rr.get("n_qrels"), rr_cis.get("recall_at_3")
            ),
            "mrr_at_3": _entry(
                rr.get("mrr_at_3"), rr.get("n_qrels"), rr_cis.get("mrr_at_3")
            ),
            "ndcg_at_3": _entry(
                rr.get("ndcg_at_3"), rr.get("n_qrels"), rr_cis.get("ndcg_at_3")
            ),
            "query_coverage": rr.get("query_coverage"),
            "top_3_retrieval_support_rate": rr.get("top_3_retrieval_support_rate"),
            "n_with_3_unique_retrieved_ids": rr.get("n_with_3_unique_retrieved_ids"),
        }
    elif rr.get("applicable") is True:
        retrieval_layer = _not_available(
            rr.get("reason") or "retrieval qrels are not configured"
        )
    else:
        retrieval_layer = {
            "status": NOT_APPLICABLE,
            "reason": rr.get("reason") or "non-RAG run (no retriever)",
        }

    layers = {
        "L2_format_robustness": {
            "json_parse_rate": _entry(sc.get("json_parse_rate"), sc.get("n"), jpr_ci),
            "schema_validity_rate": _entry(sc.get("schema_validity_rate"), sc.get("n"), sv_ci),
            "encoding_object_rate": _entry(
                sc.get("encoding_object_rate"), sc.get("n"), encoding_object_ci
            ),
            "completeness_score": _entry(sc.get("completeness_score"), sc.get("n"), comp_ci),
            "paraphrase_accuracy": _entry(rob.get("paraphrase_accuracy"), None,
                                          _pending(_ROBUSTNESS_CI_REASON)),
            "paraphrase_consistency": _entry(rob.get("paraphrase_consistency"), None,
                                             _pending(_ROBUSTNESS_CI_REASON)),
            "missing_info_clarification_rate": _entry(rob.get("missing_info_clarification_rate"), None,
                                                      _pending(_ROBUSTNESS_CI_REASON)),
            "missing_info_schema_rate": _entry(rob.get("missing_info_schema_rate"), None,
                                               _pending(_ROBUSTNESS_CI_REASON)),
        },
        "L1_chart_selection": {
            "synthetic_top1": {
                **_entry(tk.get("top_1_accuracy"), tk.get("n"), st1_ci),
                "tier": "internal-circular",
                "note": "Diagnostic only (synthetic gold == generator rule); NOT a validity claim.",
            },
            "synthetic_top3": {
                "value": tk.get("top_3_accuracy"),
                "n": tk.get("n"),
                "valid": tk.get("top_3_valid"),
                "support_rate": tk.get("top_3_support_rate"),
                "ci": _not_available("top-3 reported only when valid; CI deferred"),
                "tier": "internal-circular",
            },
            "macro_f1_synthetic": {
                **_entry(mf.get("macro_f1"), mf.get("n"),
                         _not_available("macro-F1 is not an item mean; item bootstrap is a later phase")),
                "tier": "internal-circular",
            },
            "l1_human_effectiveness": _pending("set-valued human-effectiveness scorer not implemented"),
        },
        "L1b_retrieval": retrieval_layer,
        "L1c_grounding": (
            {
                "supported_claim_rate": _entry(gr.get("supported_claim_rate"), gr.get("n"),
                                               _not_available("per-item claim vectors not persisted in Phase 1")),
                "mode": gr.get("mode"),
            }
            if gr_applicable
            else {"status": NOT_APPLICABLE, "reason": "non-RAG run (no retrieved docs)"}
        ),
        "L3_realism": _pending("Tableau Census not acquired/mapped"),
        "L4_human": _pending("human ratings not collected"),
    }

    return {
        "experiment_id": payload.get("experiment_id"),
        "method": payload.get("method"),
        "model": payload.get("model"),
        "seed": payload.get("seed"),
        "n_predictions": payload.get("n_predictions"),
        "report_schema_version": REPORT_SCHEMA_VERSION,
        # This run's predictions are scored against the synthetic gold; the
        # independent layers (L1 human-effectiveness, L4 human) are pending.
        "eval_tier": "internal-synthetic",
        "layers": layers,
    }


# --------------------------------------------------------------------------- #
# Orchestration (write both additive artifacts into a run directory)
# --------------------------------------------------------------------------- #
def mark_backfill(metrics_json: dict, *, metrics_auto_present: bool,
                  references_present: bool) -> dict:
    """Annotate a ``metrics.json`` produced by a LEGACY backfill of existing runs.

    Records honestly that point values are re-presented from ``metrics_auto.json``
    (pre-Task-7 metric code), that no fresh metric numbers or CIs were computed, and
    that per-item / gold-dependent fields and all CIs are ``not_available`` pending a
    separate corrected re-scoring task. Mutates and returns ``metrics_json``.
    """
    metrics_json["backfill"] = {
        "mode": "legacy-carry-forward",
        "legacy": True,
        "pre_task7_metrics_auto": metrics_auto_present,
        "point_values_source": (
            "metrics_auto.json (legacy, pre-Task-7)" if metrics_auto_present else "none"
        ),
        "references_present_on_disk": references_present,
        "references_used": False,
        "recomputed_from_predictions": [],
        "note": (
            "Internal-synthetic diagnostic only; NOT final corrected thesis evidence. "
            "Point values are re-presented from metrics_auto.json (pre-Task-7 metric code); "
            "no fresh metric numbers or CIs were computed, to avoid mixing legacy values with "
            "current-code results. Per-item schema_valid/completeness/gold/synthetic_top1 and "
            "all CIs are not_available. Corrected re-scoring against references is a separate task."
        ),
    }
    return metrics_json


def write_per_run_reports(exp_dir: Path, payload: dict, results, references) -> dict:
    """Write ``eval_per_item.jsonl`` and ``metrics.json`` into ``exp_dir`` (additive)."""
    exp_dir = Path(exp_dir)
    per_item = score_per_item(results, references)
    write_jsonl(per_item, exp_dir / "eval_per_item.jsonl")
    metrics_json = build_metrics_json(payload, per_item)
    write_json(metrics_json, exp_dir / "metrics.json")
    return metrics_json
