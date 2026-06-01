"""Training layer — the ONLY place that imports peft / trl / bitsandbytes.

These dependencies are imported lazily inside methods so that importing this
package's symbols stays cheap and so the inference side never pulls them in.
Only ``scripts/train.py`` (run on the GPU machine) touches this package.
"""

from src.training.galore_trainer import GaLoreSFTTrainer
from src.training.sft_trainer import QLoRASFTTrainer

__all__ = ["QLoRASFTTrainer", "GaLoreSFTTrainer"]
