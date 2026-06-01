"""Method C — fine-tuned.

Loads the base model plus the PEFT adapter produced by training. The adapter
folder is read from ``cfg.method.adapter_path`` (filled in by the experiment
config or at runtime). Generation is otherwise identical to method A.
"""

from __future__ import annotations

from typing import Optional

from src.core.registry import METHODS
from src.methods.base import HFMethod, _get


@METHODS.register("ft")
class FineTunedMethod(HFMethod):
    name = "ft"

    def _adapter_path(self) -> Optional[str]:
        path = _get(self.method_cfg, "adapter_path")
        if not path:
            raise ValueError(
                "method 'ft' requires cfg.method.adapter_path to point at a "
                "trained adapter folder (e.g. outputs/experiments/<id>/adapter)."
            )
        return str(path)
