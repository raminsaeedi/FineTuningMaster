"""
evaluation/robustness.py — Robustness metrics under input perturbations.

Metrics:
    paraphrase_consistency     — fraction of (original, paraphrased) pairs
                                  where valid_schema agrees between both
    missing_info_validity_rate — schema_validity_rate on missing-info variant

Both metrics are None when the corresponding perturbed prediction files
do not exist (standard experiments without robustness variants).
"""

from __future__ import annotations


def compute_robustness_metrics(
    original_results:   list[dict],
    paraphrase_results: list[dict] | None,
    missing_info_results: list[dict] | None,
) -> dict:
    """
    Compute consistency metrics under input perturbations.

    Parameters
    ----------
    original_results : list[dict]
        Prediction rows from the unperturbed test set.
    paraphrase_results : list[dict] or None
        Predictions on paraphrased versions of the same briefs.
        Must be the same length as original_results when provided.
    missing_info_results : list[dict] or None
        Predictions on briefs with key information omitted.

    Returns
    -------
    dict with keys:
        paraphrase_consistency      (float 0–1 or None)
        missing_info_validity_rate  (float 0–1 or None)
    """
    paraphrase_consistency = None
    if paraphrase_results and len(paraphrase_results) == len(original_results):
        consistent = sum(
            1 for o, p in zip(original_results, paraphrase_results)
            if o.get("valid_schema") == p.get("valid_schema")
        )
        paraphrase_consistency = round(consistent / len(original_results), 4)

    missing_validity = None
    if missing_info_results:
        n  = len(missing_info_results)
        ok = sum(1 for r in missing_info_results if r.get("valid_schema"))
        missing_validity = round(ok / n, 4) if n > 0 else None

    return {
        "paraphrase_consistency":     paraphrase_consistency,
        "missing_info_validity_rate": missing_validity,
    }
