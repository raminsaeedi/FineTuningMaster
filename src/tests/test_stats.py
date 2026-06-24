"""Sanity checks for the statistics module."""

from src.evaluation.stats import (
    bootstrap_ci,
    cliffs_delta,
    cochran_q,
    cohen_dz,
    holm_correction,
    mcnemar_test,
    paired_rank_biserial,
    pairwise_wilcoxon,
)


def test_holm_monotone_and_bounded():
    adj = holm_correction([0.01, 0.04, 0.03])
    assert all(0.0 <= p <= 1.0 for p in adj)
    # Holm adjusted p-values are >= the raw values.
    assert adj[0] >= 0.01


def test_cliffs_delta_extremes():
    delta, mag = cliffs_delta([5, 6, 7], [1, 2, 3])
    assert delta == 1.0 and mag == "large"
    delta2, _ = cliffs_delta([1, 1, 1], [1, 1, 1])
    assert delta2 == 0.0


def test_pairwise_wilcoxon_keys():
    scores = {"A": [3, 4, 5, 4, 5, 4], "B": [1, 2, 2, 1, 2, 1]}
    res = pairwise_wilcoxon(scores)
    assert len(res) == 1
    assert {"method_a", "method_b", "p_value", "p_holm", "reject_h0"} <= set(res[0])


def test_cochran_requires_three():
    out = cochran_q({"A": [1, 0, 1], "B": [0, 0, 1]})
    assert out["applicable"] is False


def test_mcnemar_discordant():
    out = mcnemar_test([1, 1, 0, 1], [0, 1, 0, 0])
    assert out["b"] == 2 and out["c"] == 0


def test_bootstrap_ci_orders():
    out = bootstrap_ci([1, 2, 3, 4, 5], n_boot=500)
    assert out["ci_low"] <= out["point"] <= out["ci_high"]


def test_paired_effect_sizes():
    # x strictly above y on every pair -> max positive paired effect.
    x = [2, 4, 4, 6]
    y = [1, 2, 3, 4]
    assert paired_rank_biserial(x, y) == 1.0   # all signed ranks positive
    assert cohen_dz(x, y) > 0                   # consistent positive shift (nonzero var)
    # All differences zero -> no effect, no divide-by-zero.
    assert paired_rank_biserial([1, 1, 1], [1, 1, 1]) == 0.0
    assert cohen_dz([1, 1, 1], [1, 1, 1]) == 0.0
