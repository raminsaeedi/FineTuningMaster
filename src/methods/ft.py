"""Method C — fine-tuned.

Loads the base model plus the PEFT adapter produced by training. The adapter
folder is resolved by :mod:`src.utils.adapter` — normally the run's own
``adapter/`` directory, since method C both trains and consumes it — and is
validated against the config before it is loaded, so a mismatched base model,
seed or dataset version fails loudly instead of being silently used.
"""

from __future__ import annotations

from typing import Optional

from src.core.registry import METHODS
from src.methods.base import HFMethod
from src.utils.adapter import resolve_adapter_path, validate_adapter


@METHODS.register("ft")
class FineTunedMethod(HFMethod):
    name = "ft"

    def _adapter_path(self) -> Optional[str]:
        adapter_dir = resolve_adapter_path(self.cfg)
        validate_adapter(adapter_dir, self.cfg)
        return str(adapter_dir)
