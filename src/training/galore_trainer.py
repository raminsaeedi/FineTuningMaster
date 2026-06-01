"""GaLore full-parameter fine-tuning (memory-efficient, no adapters).

GaLore (Gradient Low-Rank Projection, ICML 2024) trains ALL weights but projects
gradients to a low-rank subspace, cutting optimizer memory enough to full-fine-
tune on a single GPU. Unlike LoRA it produces a full model (no adapter), so it is
saved as a complete checkpoint. Used as a fine-tuning ablation against QLoRA.

Requires the optional ``galore-torch`` package (the ``[galore]`` extra);
Transformers wires the optimizer in via ``optim="galore_adamw"``. Imports are
lazy, so this module is safe to import without galore installed.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, List, Mapping

from src.core.interfaces import BaseTrainer
from src.core.registry import TRAINERS
from src.training.sft_trainer import _as_list, _get

logger = logging.getLogger(__name__)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@TRAINERS.register("galore_sft")
class GaLoreSFTTrainer(BaseTrainer):
    """Full-parameter SFT with the GaLore optimizer."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.model_cfg = _get(cfg, "model", {})
        self.train_cfg = _get(cfg, "training", {})
        self.seed = int(_get(cfg, "seed", 42))
        self.model = None
        self.tokenizer = None

    def train(self, train_dataset, eval_dataset, output_dir: str) -> str:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        name = _get(self.model_cfg, "hf_id") or _get(self.model_cfg, "name")
        cache_dir = _get(self.model_cfg, "cache_dir")
        galore = _get(self.train_cfg, "galore", {})
        sft = _get(self.train_cfg, "sft", {})

        self.tokenizer = AutoTokenizer.from_pretrained(
            name, trust_remote_code=True, padding_side="right", cache_dir=cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            name, device_map="auto", trust_remote_code=True, dtype=dtype, cache_dir=cache_dir
        )
        self.model.config.use_cache = False

        target_modules: List[str] = _as_list(_get(galore, "target_modules")) or ["attn", "mlp"]
        optim_args = (
            f"rank={int(_get(galore, 'rank', 128))},"
            f"update_proj_gap={int(_get(galore, 'update_proj_gap', 200))},"
            f"scale={float(_get(galore, 'scale', 0.25))}"
        )

        training_args = SFTConfig(
            output_dir=str(Path(output_dir).parent / "checkpoints"),
            num_train_epochs=int(_get(sft, "num_train_epochs", 3)),
            per_device_train_batch_size=int(_get(sft, "per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(_get(sft, "gradient_accumulation_steps", 8)),
            learning_rate=float(_get(sft, "learning_rate", 1.0e-5)),
            lr_scheduler_type=str(_get(sft, "lr_scheduler_type", "cosine")),
            warmup_ratio=float(_get(sft, "warmup_ratio", 0.03)),
            weight_decay=float(_get(sft, "weight_decay", 0.0)),
            bf16=bool(_get(sft, "bf16", torch.cuda.is_available())),
            logging_steps=int(_get(sft, "logging_steps", 10)),
            save_steps=int(_get(sft, "save_steps", 50)),
            save_total_limit=int(_get(sft, "save_total_limit", 1)),
            gradient_checkpointing=bool(_get(sft, "gradient_checkpointing", True)),
            seed=self.seed,
            remove_unused_columns=True,
            dataset_text_field="text",
            packing=bool(_get(sft, "packing", False)),
            optim="galore_adamw",                 # requires galore-torch
            optim_target_modules=target_modules,
            optim_args=optim_args,
        )

        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )
        logger.info("GaLore training started (optim_args=%s)…", optim_args)
        result = trainer.train()
        self.metrics = result.metrics

        # GaLore has no adapter — save the full model.
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))
        with (out / "training_metadata.json").open("w", encoding="utf-8") as f:
            json.dump({
                "base_model": name, "trainer": "galore_sft", "full_finetune": True,
                "galore": {"target_modules": target_modules, "optim_args": optim_args},
                "seed": self.seed, "train_metrics": self.metrics,
            }, f, indent=2, default=str)
        logger.info("GaLore model saved to %s", out)
        return str(out)
