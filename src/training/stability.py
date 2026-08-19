"""Stop training the moment loss or adapter weights become non-finite.

Experiment C kept stepping for ~900 updates after grad_norm went NaN because
HuggingFace logs NaN windows as 0.0 and still saves the adapter. Raise instead.

Duck-typed as a transformers TrainerCallback (no import at module load).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


def nonfinite_log_reason(logs: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Return a short reason if logged loss/grad_norm is NaN or Inf."""
    if not logs:
        return None
    for key in ("grad_norm", "loss"):
        if key not in logs or logs[key] is None:
            continue
        try:
            value = float(logs[key])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            return f"non-finite {key}={logs[key]!r}"
    return None


def _trainable_params_finite(model: Any) -> bool:
    if model is None:
        return True
    try:
        import torch
    except Exception:
        return True
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return True
    for parameter in parameters():
        if not getattr(parameter, "requires_grad", False):
            continue
        data = getattr(parameter, "data", None)
        if data is None:
            continue
        if not torch.isfinite(data).all().item():
            return False
    return True


# Keep import-time cost low. Transformers is large and importing it here pulls
# optional native packages (pandas/pyarrow) into CPU-only tests. Trainer
# callback dispatch is duck-typed, so this compatibility base supplies the
# no-op lifecycle hooks that newer Transformers versions may dispatch.
class TrainerCallbackCompat:
    """Duck-typed TrainerCallback compatible with changing event surfaces."""

    def __getattr__(self, name):  # noqa: ANN001
        if not name.startswith("on_"):
            raise AttributeError(name)

        def _noop(*args, **kwargs):  # noqa: ANN001
            if "control" in kwargs:
                return kwargs["control"]
            return args[2] if len(args) >= 3 else None

        return _noop


class AbortOnNonFiniteCallback(TrainerCallbackCompat):
    """Abort the run on NaN/Inf logs or poisoned LoRA weights."""

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        reason = nonfinite_log_reason(logs)
        if reason:
            step = getattr(state, "global_step", "?")
            raise FloatingPointError(
                f"Training aborted at step {step}: {reason}. "
                "This is the Experiment C failure mode (fp16 overflow / NaN adapter)."
            )
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
        if model is None:
            model = kwargs.get("model")
        if model is not None and not _trainable_params_finite(model):
            step = getattr(state, "global_step", "?")
            raise FloatingPointError(
                f"Training aborted at step {step}: non-finite trainable parameters. "
                "Adapter weights have overflowed; refusing to continue or save."
            )
        return control
