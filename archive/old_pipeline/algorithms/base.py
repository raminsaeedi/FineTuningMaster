"""
Abstract base class for all fine-tuning algorithm implementations.

Every algorithm must implement three methods:
  setup()  — load model + tokenizer, apply adapter/quantization config
  train()  — run the training loop, return metrics dict
  save()   — persist adapter/model weights to experiment_dir/final_adapter/

The pipeline calls run() which invokes them in that order.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BaseFineTuner(ABC):
    """
    Abstract interface for fine-tuning algorithms.

    Subclasses receive a fully merged config dict and the experiment output
    directory, both provided by pipeline/train.py before training starts.
    """

    def __init__(self, config: dict, experiment_dir: Path):
        """
        Parameters
        ----------
        config:          Fully merged config dict (base + model + experiment + CLI).
        experiment_dir:  Path to the experiment output directory created by
                         utils.experiment.setup_experiment().
        """
        self.config = config
        self.experiment_dir = experiment_dir
        self.model = None
        self.tokenizer = None
        self.trainer = None

        self._log_dir = experiment_dir / "logs"
        self._adapter_dir = experiment_dir / "final_adapter"
        self._checkpoint_dir = experiment_dir / "checkpoints"

    @abstractmethod
    def setup(self) -> None:
        """
        Load model + tokenizer and apply any adapter/quantization config.

        After this method returns, self.model and self.tokenizer must be
        populated and ready for training.
        """

    @abstractmethod
    def train(self, train_dataset, eval_dataset=None) -> dict:
        """
        Execute the training loop.

        Parameters
        ----------
        train_dataset:  HuggingFace Dataset object with a 'text' column
                        (formatted instruction-response strings).
        eval_dataset:   Optional validation dataset with the same schema.

        Returns
        -------
        dict with at minimum:
          {train_loss, train_runtime, train_samples_per_second}
        """

    @abstractmethod
    def save(self) -> None:
        """
        Persist the trained model/adapter to self._adapter_dir.

        Must also write training_metadata.json to that directory.
        """

    # ------------------------------------------------------------------
    # Template method — called by pipeline/train.py
    # ------------------------------------------------------------------

    def run(self, train_dataset, eval_dataset=None) -> dict:
        """
        Execute the full fine-tuning pipeline: setup → train → save.

        Returns the metrics dict from train().
        """
        logger.info(f"[{self.__class__.__name__}] setup()")
        self.setup()

        logger.info(f"[{self.__class__.__name__}] train()")
        metrics = self.train(train_dataset, eval_dataset)

        logger.info(f"[{self.__class__.__name__}] save()")
        self.save()

        return metrics

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _write_metadata(self, extra: dict[str, Any] | None = None) -> None:
        """Write training_metadata.json to final_adapter dir."""
        self._adapter_dir.mkdir(parents=True, exist_ok=True)
        lora_cfg = self.config.get("lora", {})
        train_cfg = self.config.get("training", {})
        metadata: dict[str, Any] = {
            "base_model":   self.config.get("model", {}).get("name", "unknown"),
            "algorithm":    self.config.get("algorithm", {}).get("name", "unknown"),
            "lora_r":       lora_cfg.get("r"),
            "lora_alpha":   lora_cfg.get("lora_alpha"),
            "num_epochs":   train_cfg.get("num_train_epochs"),
            "learning_rate": train_cfg.get("learning_rate"),
            "seed":         self.config.get("meta", {}).get("seed", 42),
            "experiment_id": self.config.get("_runtime", {}).get("experiment_id"),
        }
        if extra:
            metadata.update(extra)
        out_path = self._adapter_dir / "training_metadata.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Metadata written to {out_path}")
