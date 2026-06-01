"""Reproducible seeding.

NOTE (Windows): ``torch.cuda.manual_seed_all`` is intentionally omitted. Calling
it before the dataset is loaded (pyarrow/CUDA DLL initialisation) triggers a
Windows 0xC0000005 access violation due to DLL load order. ``torch.manual_seed``
already covers the default CUDA device, which is sufficient for single-GPU runs.
Always load/format the dataset before calling this on Windows.
"""

from __future__ import annotations

import random


def set_seeds(seed: int) -> None:
    """Seed Python, NumPy and (if present) PyTorch."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
