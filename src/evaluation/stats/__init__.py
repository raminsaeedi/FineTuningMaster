"""Statistical tests for comparing methods on matched items.

All tests assume *paired* data: the same test items run through each method, so
scores/outcomes are aligned by position. This subpackage is the scientific core
of the evaluation and depends only on numpy/scipy.
"""

from src.evaluation.stats.bootstrap_ci import bootstrap_ci, paired_bootstrap_diff
from src.evaluation.stats.cliff_delta import cliffs_delta
from src.evaluation.stats.cochran_mcnemar import cochran_q, mcnemar_test, pairwise_mcnemar
from src.evaluation.stats.effect_size import cohen_dz, paired_effect_size, paired_rank_biserial
from src.evaluation.stats.friedman import friedman_test
from src.evaluation.stats.wilcoxon_holm import holm_correction, pairwise_wilcoxon

__all__ = [
    "friedman_test",
    "pairwise_wilcoxon",
    "holm_correction",
    "cliffs_delta",
    "paired_rank_biserial",
    "cohen_dz",
    "paired_effect_size",
    "cochran_q",
    "mcnemar_test",
    "pairwise_mcnemar",
    "bootstrap_ci",
    "paired_bootstrap_diff",
]
