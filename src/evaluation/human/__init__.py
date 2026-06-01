"""Human evaluation: rubric, blind balanced assignment, ratings, Krippendorff's alpha.

This subpackage is import-safe (no streamlit at import time except the app module
itself). The Streamlit app lives in ``streamlit_app.py`` and is launched via
``scripts/run_human_eval.py``.
"""

from src.evaluation.human.assignment import build_assignment, build_eval_items
from src.evaluation.human.irr import krippendorff_alpha
from src.evaluation.human.rubric import RUBRIC, RUBRIC_KEYS

__all__ = [
    "RUBRIC",
    "RUBRIC_KEYS",
    "build_assignment",
    "build_eval_items",
    "krippendorff_alpha",
]
