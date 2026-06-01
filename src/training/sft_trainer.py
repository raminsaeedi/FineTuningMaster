"""QLoRA supervised fine-tuning — migrated from the original working trainer.

This is a faithful restructuring of the proven ``algorithms/qlora.py`` plus the
``SFTConfig`` assembly from the old ``pipeline/train.py``. The hyperparameters
and the order of operations are unchanged; only the surrounding plumbing (config
access, registry, artifact paths) was adapted to the new architecture.

Technique: load the base model in 4-bit NF4 (bitsandbytes), prepare it for
k-bit training, attach LoRA adapters, and train with TRL's ``SFTTrainer`` on the
pre-formatted ``text`` column. All heavy imports are local to the methods so the
module stays import-safe.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, List, Mapping, Optional

from src.core.interfaces import BaseTrainer
from src.core.registry import TRAINERS

logger = logging.getLogger(__name__)

# Suppress tokenizer parallelism warnings from HuggingFace.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _get(cfg: Mapping[str, Any], key: str, default: Any = None) -> Any:
    try:
        return cfg.get(key, default)
    except AttributeError:
        return getattr(cfg, key, default)


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return list(OmegaConf.to_container(value, resolve=True))  # type: ignore[arg-type]
    except Exception:
        pass
    return list(value)


def build_lora_kwargs(lora_cfg: Mapping[str, Any]) -> dict:
    """Assemble LoraConfig kwargs, including DoRA / RSLoRA toggles.

    DoRA (use_dora=True) decomposes weights into magnitude + direction; RSLoRA
    (use_rslora=True) rank-stabilizes the LoRA scaling. Both are native PEFT
    flags, so the same trainer covers QLoRA / LoRA / DoRA / RSLoRA — selecting an
    algorithm is just a config change.
    """
    return dict(
        r=int(_get(lora_cfg, "r", 16)),
        lora_alpha=int(_get(lora_cfg, "lora_alpha", 32)),
        lora_dropout=float(_get(lora_cfg, "lora_dropout", 0.05)),
        bias=str(_get(lora_cfg, "bias", "none")),
        target_modules=_as_list(_get(lora_cfg, "target_modules")),
        use_dora=bool(_get(lora_cfg, "use_dora", False)),
        use_rslora=bool(_get(lora_cfg, "use_rslora", False)),
    )


@TRAINERS.register("qlora_sft")
class QLoRASFTTrainer(BaseTrainer):
    """4-bit quantized base model + LoRA adapters via PEFT + SFTTrainer."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.model_cfg = _get(cfg, "model", {})
        self.train_cfg = _get(cfg, "training", {})
        self.seed = int(_get(cfg, "seed", 42))
        self.model = None
        self.tokenizer = None

    # ------------------------------------------------------------------
    def _setup(self) -> None:
        """Load base model with 4-bit quantization and attach LoRA adapters."""
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        name = _get(self.model_cfg, "hf_id") or _get(self.model_cfg, "name")
        cache_dir = _get(self.model_cfg, "cache_dir")
        max_seq_length = int(_get(self.model_cfg, "max_seq_length", 2048))

        lora_cfg = _get(self.train_cfg, "lora", {})
        quant_cfg = _get(self.train_cfg, "quantization", {})

        # ── Tokenizer ────────────────────────────────────────────────────
        logger.info("Loading tokenizer: %s", name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            padding_side="right",  # required for SFT causal LM
            cache_dir=cache_dir,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = max_seq_length

        # ── Quantization config ──────────────────────────────────────────
        bnb_config = None
        if bool(_get(quant_cfg, "load_in_4bit", True)):
            logger.info("Configuring 4-bit NF4 quantization (QLoRA)")
            compute_dtype = getattr(
                torch, str(_get(quant_cfg, "bnb_4bit_compute_dtype", "float16"))
            )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=str(_get(quant_cfg, "bnb_4bit_quant_type", "nf4")),
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=bool(_get(quant_cfg, "bnb_4bit_use_double_quant", True)),
            )

        # ── Base model ───────────────────────────────────────────────────
        logger.info("Loading model: %s", name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.float16 if not bnb_config else None,
            cache_dir=cache_dir,
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        logger.info("Model loaded — %s parameters", f"{self.model.num_parameters():,}")

        # ── Prepare for k-bit training ───────────────────────────────────
        if bnb_config:
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=True
            )

        # ── LoRA adapters (QLoRA / LoRA / DoRA / RSLoRA via flags) ───────────
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            **build_lora_kwargs(lora_cfg),
        )
        self.model = get_peft_model(self.model, lora_config)

        trainable, total = self.model.get_nb_trainable_parameters()
        pct = 100.0 * trainable / total if total > 0 else 0.0
        logger.info("LoRA applied: %s trainable / %s total (%.2f%%)", f"{trainable:,}", f"{total:,}", pct)

    # ------------------------------------------------------------------
    def train(self, train_dataset, eval_dataset, output_dir: str) -> str:
        """Run SFT and save the adapter to ``output_dir``; return that path."""
        from trl import SFTConfig, SFTTrainer

        self._setup()

        sft = _get(self.train_cfg, "sft", {})
        adapter_dir = Path(output_dir)
        checkpoint_dir = adapter_dir.parent / "checkpoints"

        training_args = SFTConfig(
            output_dir=str(checkpoint_dir),
            num_train_epochs=int(_get(sft, "num_train_epochs", 3)),
            per_device_train_batch_size=int(_get(sft, "per_device_train_batch_size", 2)),
            gradient_accumulation_steps=int(_get(sft, "gradient_accumulation_steps", 4)),
            learning_rate=float(_get(sft, "learning_rate", 2.0e-4)),
            lr_scheduler_type=str(_get(sft, "lr_scheduler_type", "cosine")),
            warmup_ratio=float(_get(sft, "warmup_ratio", 0.1)),
            weight_decay=float(_get(sft, "weight_decay", 0.01)),
            fp16=bool(_get(sft, "fp16", False)),
            bf16=bool(_get(sft, "bf16", False)),
            logging_steps=int(_get(sft, "logging_steps", 10)),
            save_steps=int(_get(sft, "save_steps", 50)),
            save_total_limit=int(_get(sft, "save_total_limit", 2)),
            eval_strategy=str(_get(sft, "eval_strategy", "no")),
            report_to=str(_get(sft, "report_to", "none")),
            seed=self.seed,
            gradient_checkpointing=bool(_get(sft, "gradient_checkpointing", True)),
            dataloader_pin_memory=bool(_get(sft, "dataloader_pin_memory", False)),
            remove_unused_columns=True,
            dataset_text_field="text",
            packing=bool(_get(sft, "packing", False)),
        )

        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,  # replaces `tokenizer=` in recent TRL
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )

        logger.info("Training started…")
        result = trainer.train()
        self.metrics = result.metrics
        logger.info("Training complete. Loss: %s", self.metrics.get("train_loss", "N/A"))

        self._save(adapter_dir)
        return str(adapter_dir)

    # ------------------------------------------------------------------
    def _save(self, adapter_dir: Path) -> None:
        """Persist LoRA adapter, tokenizer and training metadata."""
        adapter_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving LoRA adapter to %s", adapter_dir)
        self.model.save_pretrained(str(adapter_dir))
        self.tokenizer.save_pretrained(str(adapter_dir))

        lora_cfg = _get(self.train_cfg, "lora", {})
        sft = _get(self.train_cfg, "sft", {})
        metadata = {
            "base_model": _get(self.model_cfg, "hf_id") or _get(self.model_cfg, "name"),
            "trainer": "qlora_sft",
            "quantization": "4bit_nf4",
            "lora_r": _get(lora_cfg, "r"),
            "lora_alpha": _get(lora_cfg, "lora_alpha"),
            "num_train_epochs": _get(sft, "num_train_epochs"),
            "learning_rate": _get(sft, "learning_rate"),
            "seed": self.seed,
            "train_metrics": getattr(self, "metrics", {}),
        }
        with (adapter_dir / "training_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("Metadata written to %s", adapter_dir / "training_metadata.json")
