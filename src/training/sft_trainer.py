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

import hashlib
import inspect
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, List, Mapping, Optional

from src.core.interfaces import BaseTrainer
from src.core.registry import TRAINERS
from src.data_pipeline.formatter import split_training_text
from src.training.stability import AbortOnNonFiniteCallback, TrainerCallbackCompat
from src.utils.config_hash import hash_config
from src.utils.checkpoints import mark_checkpoint_complete
from src.utils.gpu_precision import (
    align_trainable_parameters,
    assert_fp16_amp_safe,
    lora_param_dtype_name,
    resolve_training_precision,
)
from src.models.hf_utils import (
    from_pretrained_kwargs,
    load_pretrained_with_cache_repair,
    model_identifier,
    safe_model_access_error,
)
from src.utils.numerics import (
    nonfinite_scalar_items,
    raise_if_nonfinite_parameters,
    validate_checkpoint_weights_finite,
)

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
    if isinstance(value, str):
        return [value]
    return list(value)


def build_lora_kwargs(lora_cfg: Mapping[str, Any]) -> dict:
    """Assemble LoraConfig kwargs, including DoRA / RSLoRA toggles.

    DoRA (use_dora=True) decomposes weights into magnitude + direction; RSLoRA
    (use_rslora=True) rank-stabilizes the LoRA scaling. Both are native PEFT
    flags, so the same trainer covers QLoRA / LoRA / DoRA / RSLoRA — selecting an
    algorithm is just a config change.
    """
    target_modules = _get(lora_cfg, "target_modules")
    if isinstance(target_modules, str) and target_modules == "all-linear":
        normalized_targets: Any = target_modules
    else:
        normalized_targets = _as_list(target_modules)
    return dict(
        r=int(_get(lora_cfg, "r", 16)),
        lora_alpha=int(_get(lora_cfg, "lora_alpha", 32)),
        lora_dropout=float(_get(lora_cfg, "lora_dropout", 0.05)),
        bias=str(_get(lora_cfg, "bias", "none")),
        target_modules=normalized_targets,
        use_dora=bool(_get(lora_cfg, "use_dora", False)),
        use_rslora=bool(_get(lora_cfg, "use_rslora", False)),
    )


def _require_completion_only_api(sft_config_cls: type) -> None:
    """Fail before GPU setup when installed TRL cannot mask prompt labels."""
    try:
        parameters = inspect.signature(sft_config_cls).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "TRL compatibility check failed for completion_only_loss. "
            "Install the pinned training environment with trl==1.3.0."
        ) from exc
    if "completion_only_loss" not in parameters:
        raise RuntimeError(
            "Installed TRL does not expose SFTConfig.completion_only_loss. "
            "Install the pinned training environment with trl==1.3.0, or set "
            "training.sft.completion_only_loss=false."
        )


def _to_prompt_completion_dataset(dataset, eos_token: str, dataset_name: str):
    """Convert legacy formatted text to TRL's native prompt-completion schema."""
    if dataset is None:
        return None
    sample = next(iter(dataset), None)
    if sample is None:
        raise ValueError(f"{dataset_name} dataset is empty; cannot run SFT.")
    if "prompt" in sample and "completion" in sample:
        return dataset
    if "text" not in sample:
        raise ValueError(
            f"{dataset_name} dataset must contain either 'text' or both "
            "'prompt' and 'completion' when completion_only_loss=true."
        )
    if not hasattr(dataset, "map"):
        raise TypeError(
            f"{dataset_name} dataset does not support map(); expected a "
            "Hugging Face Dataset or IterableDataset."
        )

    def split_row(example):
        return split_training_text(example["text"], eos_token=eos_token)

    return dataset.map(split_row, remove_columns=["text"])


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
        self.final_global_step = 0
        self.trainable_parameters = 0
        self.total_parameters = 0
        self.training_started_at = None
        self.training_duration_seconds = None
        self.precision = None

    # ------------------------------------------------------------------
    def _setup(self) -> None:
        """Load base model with 4-bit quantization and attach LoRA adapters."""
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        name = model_identifier(self.model_cfg)
        cache_dir = _get(self.model_cfg, "cache_dir")
        max_seq_length = int(_get(self.model_cfg, "max_seq_length", 2048))

        lora_cfg = _get(self.train_cfg, "lora", {})
        quant_cfg = _get(self.train_cfg, "quantization", {})
        sft_cfg = _get(self.train_cfg, "sft", {})
        gradient_checkpointing = bool(_get(sft_cfg, "gradient_checkpointing", True))

        # Resolve from the weakest visible GPU. Configured model dtype is only
        # a preference; mixed jobs must use one safe precision intersection.
        self.precision = resolve_training_precision(sft_cfg)
        logger.info("Training precision: %s", self.precision.reason)

        # ── Tokenizer ────────────────────────────────────────────────────
        logger.info("Loading tokenizer: %s", name)
        load_kwargs = from_pretrained_kwargs(
            self.model_cfg,
            cache_dir=cache_dir,
        )
        try:
            self.tokenizer = load_pretrained_with_cache_repair(
                AutoTokenizer.from_pretrained,
                name,
                kwargs={
                    **load_kwargs,
                    "padding_side": "right",  # required for SFT causal LM
                },
                logger=logger,
                component="training tokenizer",
            )
        except Exception as exc:
            raise safe_model_access_error(name, exc) from exc
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = max_seq_length

        # ── Quantization config ──────────────────────────────────────────
        bnb_config = None
        if bool(_get(quant_cfg, "load_in_4bit", True)):
            compute_name = self.precision.compute_dtype
            logger.info(
                "Configuring 4-bit NF4 quantization (QLoRA, compute_dtype=%s)",
                compute_name,
            )
            compute_dtype = getattr(torch, compute_name)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=str(_get(quant_cfg, "bnb_4bit_quant_type", "nf4")),
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=bool(_get(quant_cfg, "bnb_4bit_use_double_quant", True)),
            )

        # ── Base model ───────────────────────────────────────────────────
        logger.info("Loading model: %s", name)
        try:
            # Never leave dtype unset: Transformers may honor a bfloat16 model
            # config even when the visible GPU only supports fp16.
            load_dtype = getattr(torch, self.precision.compute_dtype)
            self.model = load_pretrained_with_cache_repair(
                AutoModelForCausalLM.from_pretrained,
                name,
                kwargs={
                    **load_kwargs,
                    "quantization_config": bnb_config,
                    "device_map": "auto" if torch.cuda.is_available() else None,
                    "dtype": load_dtype,
                },
                logger=logger,
                component="training model",
            )
        except Exception as exc:
            raise safe_model_access_error(name, exc) from exc
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        self.model.config.torch_dtype = load_dtype
        logger.info("Model loaded — %s parameters", f"{self.model.num_parameters():,}")

        # ── Prepare for k-bit training ───────────────────────────────────
        if bnb_config:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=gradient_checkpointing,
            )

        # ── LoRA adapters (QLoRA / LoRA / DoRA / RSLoRA via flags) ───────────
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            **build_lora_kwargs(lora_cfg),
        )
        self.model = get_peft_model(self.model, lora_config)
        align_trainable_parameters(self.model, self.precision)

        trainable, total = self.model.get_nb_trainable_parameters()
        self.trainable_parameters = int(trainable)
        self.total_parameters = int(total)
        pct = 100.0 * trainable / total if total > 0 else 0.0
        logger.info("LoRA applied: %s trainable / %s total (%.2f%%)", f"{trainable:,}", f"{total:,}", pct)

    # ------------------------------------------------------------------
    def train(
        self,
        train_dataset,
        eval_dataset,
        output_dir: str,
        resume_from_checkpoint: Optional[str] = None,
    ) -> str:
        """Run SFT and save the adapter to ``output_dir``; return that path."""
        from trl import SFTConfig, SFTTrainer

        sft = _get(self.train_cfg, "sft", {})
        completion_only_loss = _get(sft, "completion_only_loss", False)
        if not isinstance(completion_only_loss, bool):
            raise ValueError("training.sft.completion_only_loss must be true or false.")
        if completion_only_loss:
            _require_completion_only_api(SFTConfig)

        self._setup()
        if self.precision is None:
            # Keeps the public trainer contract safe for lightweight injected
            # setup implementations used by tests and downstream callers.
            import torch

            sft_cfg = _get(self.train_cfg, "sft", {})
            self.precision = resolve_training_precision(sft_cfg)

        adapter_dir = Path(output_dir)
        checkpoint_dir = adapter_dir.parent / "checkpoints"
        max_seq_length = int(_get(self.model_cfg, "max_seq_length", 2048))

        if completion_only_loss:
            eos_token = getattr(self.tokenizer, "eos_token", "") or ""
            train_dataset = _to_prompt_completion_dataset(
                train_dataset, eos_token, "Training"
            )
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    name: _to_prompt_completion_dataset(dataset, eos_token, f"Evaluation '{name}'")
                    for name, dataset in eval_dataset.items()
                }
            else:
                eval_dataset = _to_prompt_completion_dataset(
                    eval_dataset, eos_token, "Evaluation"
                )

        training_kwargs = dict(
            output_dir=str(checkpoint_dir),
            num_train_epochs=int(_get(sft, "num_train_epochs", 3)),
            per_device_train_batch_size=int(_get(sft, "per_device_train_batch_size", 2)),
            per_device_eval_batch_size=int(_get(sft, "per_device_eval_batch_size", 1)),
            gradient_accumulation_steps=int(_get(sft, "gradient_accumulation_steps", 4)),
            learning_rate=float(_get(sft, "learning_rate", 2.0e-4)),
            lr_scheduler_type=str(_get(sft, "lr_scheduler_type", "cosine")),
            optim=str(_get(sft, "optim", "adamw_torch")),
            warmup_ratio=float(_get(sft, "warmup_ratio", 0.1)),
            weight_decay=float(_get(sft, "weight_decay", 0.01)),
            fp16=self.precision.amp_fp16,
            bf16=self.precision.amp_bf16,
            max_length=max_seq_length,
            max_grad_norm=float(_get(sft, "max_grad_norm", 1.0)),
            logging_steps=int(_get(sft, "logging_steps", 10)),
            save_steps=int(_get(sft, "save_steps", 50)),
            save_total_limit=int(_get(sft, "save_total_limit", 4)),
            eval_strategy=str(_get(sft, "eval_strategy", "no")),
            save_strategy=str(_get(sft, "save_strategy", "steps")),
            load_best_model_at_end=bool(_get(sft, "load_best_model_at_end", False)),
            metric_for_best_model=str(_get(sft, "metric_for_best_model", "eval_loss")),
            greater_is_better=bool(_get(sft, "greater_is_better", False)),
            report_to=str(_get(sft, "report_to", "none")),
            max_steps=int(_get(sft, "max_steps", -1)),
            seed=self.seed,
            gradient_checkpointing=bool(_get(sft, "gradient_checkpointing", True)),
            dataloader_pin_memory=bool(_get(sft, "dataloader_pin_memory", False)),
            remove_unused_columns=True,
            packing=bool(_get(sft, "packing", False)),
        )
        if completion_only_loss:
            training_kwargs["completion_only_loss"] = True
        else:
            training_kwargs["dataset_text_field"] = "text"
        training_args = SFTConfig(**training_kwargs)

        owner = self

        class FiniteTrainingCallback(TrainerCallbackCompat):
            """Stop before a bad log/checkpoint can become a saved adapter."""

            def on_log(self, args, state, control, logs=None, **kwargs):
                del args, state, kwargs
                logs = logs or {}
                bad = nonfinite_scalar_items(logs)
                if bad:
                    raise FloatingPointError(
                        "Non-finite training metrics detected "
                        f"({', '.join(bad)}); refusing to continue or save weights."
                    )
                return control

            def on_save(self, args, state, control, **kwargs):
                model = kwargs.get("model")
                if model is None:
                    model = owner.model
                raise_if_nonfinite_parameters(
                    model,
                    trainable_only=True,
                    context="Training checkpoint",
                )
                # Runs after Trainer has finished writing the checkpoint
                # directory. Publishing the marker is the last step and is
                # itself atomic, so a checkpoint interrupted at any earlier
                # point never becomes markable — and a run that crashes between
                # two checkpoints still finds the previous complete one.
                mark_checkpoint_complete(
                    Path(args.output_dir) / f"checkpoint-{int(state.global_step)}",
                    global_step=int(state.global_step),
                )
                return control

        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,  # replaces `tokenizer=` in recent TRL
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            callbacks=[FiniteTrainingCallback(), AbortOnNonFiniteCallback()],
        )
        trainer_model = getattr(trainer, "model", self.model)
        align_trainable_parameters(trainer_model, self.precision)
        assert_fp16_amp_safe(trainer_model, self.precision)
        logger.info(
            "AMP: fp16=%s bf16=%s LoRA=%s (%s)",
            self.precision.amp_fp16,
            self.precision.amp_bf16,
            lora_param_dtype_name(self.precision),
            self.precision.reason,
        )

        logger.info("Training started…")
        self.training_started_at = time.perf_counter()
        train_kwargs = {}
        if resume_from_checkpoint:
            validate_checkpoint_weights_finite(resume_from_checkpoint)
            train_kwargs["resume_from_checkpoint"] = resume_from_checkpoint
        result = trainer.train(**train_kwargs)
        self.metrics = result.metrics or {}
        bad_metrics = nonfinite_scalar_items(self.metrics)
        if bad_metrics:
            raise FloatingPointError(
                "Training returned non-finite metrics "
                f"({', '.join(bad_metrics)}); refusing to save the adapter."
            )
        log_history = getattr(getattr(trainer, "state", None), "log_history", []) or []
        for history_entry in log_history:
            bad_history = nonfinite_scalar_items(history_entry)
            if bad_history:
                raise FloatingPointError(
                    "Training history contains non-finite metrics "
                    f"({', '.join(bad_history)}); refusing to save the adapter."
                )
        self.eval_history = [entry for entry in log_history if "eval_loss" in entry]
        trainer_state = getattr(trainer, "state", None)
        self.best_eval_metric = getattr(trainer_state, "best_metric", None)
        self.best_checkpoint = getattr(trainer_state, "best_model_checkpoint", None)
        raise_if_nonfinite_parameters(
            self.model,
            trainable_only=True,
            context="Final trainable adapter",
        )
        state_step = getattr(getattr(trainer, "state", None), "global_step", None)
        self.final_global_step = int(state_step or self.metrics.get("global_step", 0) or 0)
        self.training_duration_seconds = round(time.perf_counter() - self.training_started_at, 3)
        logger.info("Training complete. Loss: %s", self.metrics.get("train_loss", "N/A"))

        self._save(adapter_dir)
        return str(adapter_dir)

    # ------------------------------------------------------------------
    def _save(self, adapter_dir: Path) -> None:
        """Persist LoRA adapter, tokenizer and training metadata."""
        raise_if_nonfinite_parameters(
            self.model,
            trainable_only=True,
            context="Adapter being saved",
        )
        adapter_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving LoRA adapter to %s", adapter_dir)
        self.model.save_pretrained(str(adapter_dir))
        self.tokenizer.save_pretrained(str(adapter_dir))

        lora_cfg = _get(self.train_cfg, "lora", {})
        sft = _get(self.train_cfg, "sft", {})
        data_cfg = _get(self.cfg, "data", {})
        train_file = _get(data_cfg, "train_file")
        train_file_sha256 = None
        if train_file:
            train_path = Path(str(train_file))
            if not train_path.is_absolute():
                train_path = Path.cwd() / train_path
            if train_path.is_file():
                digest = hashlib.sha256()
                with train_path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1 << 16), b""):
                        digest.update(chunk)
                train_file_sha256 = digest.hexdigest()
        # dataset_version / experiment_id are what src.utils.adapter checks before
        # method D reuses this adapter, so they must be recorded at save time.
        metadata = {
            "base_model": _get(self.model_cfg, "hf_id") or _get(self.model_cfg, "name"),
            "model_key": _get(self.model_cfg, "key"),
            "model_revision": _get(self.model_cfg, "revision"),
            "model_config_hash": hash_config(self.model_cfg),
            "trainer": "qlora_sft",
            "quantization": "4bit_nf4",
            "lora_r": _get(lora_cfg, "r"),
            "lora_alpha": _get(lora_cfg, "lora_alpha"),
            "num_train_epochs": _get(sft, "num_train_epochs"),
            "learning_rate": _get(sft, "learning_rate"),
            "seed": self.seed,
            "dataset_version": _get(data_cfg, "dataset_version"),
            "train_file": train_file,
            "train_file_sha256": train_file_sha256,
            "training_config_hash": hash_config(self.train_cfg),
            "max_steps": _get(sft, "max_steps", -1),
            "per_device_train_batch_size": _get(sft, "per_device_train_batch_size"),
            "gradient_accumulation_steps": _get(sft, "gradient_accumulation_steps"),
            "optimizer": _get(sft, "optim", "adamw_torch"),
            "scheduler": _get(sft, "lr_scheduler_type"),
            "warmup_ratio": _get(sft, "warmup_ratio"),
            "max_seq_length": _get(self.model_cfg, "max_seq_length"),
            "completion_only_loss": bool(_get(sft, "completion_only_loss", False)),
            "precision": {
                "mode": None if self.precision is None else self.precision.mode,
                "dtype": None if self.precision is None else self.precision.inference_dtype,
                "fp16": None if self.precision is None else self.precision.amp_fp16,
                "bf16": None if self.precision is None else self.precision.amp_bf16,
                "compute_dtype": None if self.precision is None else self.precision.compute_dtype,
                "reason": None if self.precision is None else self.precision.reason,
                "devices": None if self.precision is None else list(self.precision.devices),
                "effective": self.precision.as_metadata() if self.precision else None,
            },
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "training_duration_seconds": self.training_duration_seconds,
            "experiment_id": _get(self.cfg, "experiment_id"),
            "experiment_name": _get(self.cfg, "experiment_name"),
            "train_metrics": getattr(self, "metrics", {}),
            "validation": {
                "eval_strategy": _get(sft, "eval_strategy", "no"),
                "metric_for_best_model": _get(sft, "metric_for_best_model", "eval_loss"),
                "best_metric": getattr(self, "best_eval_metric", None),
                "best_checkpoint": getattr(self, "best_checkpoint", None),
                "history": getattr(self, "eval_history", []),
            },
        }
        with (adapter_dir / "training_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("Metadata written to %s", adapter_dir / "training_metadata.json")
