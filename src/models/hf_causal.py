"""HuggingFace causal-LM wrapper used by the prompt-only and fine-tuned methods.

This consolidates the base-model and fine-tuned-model loading paths from the
original ``pipeline/inference.py`` into one object with ``load(adapter_path)``
and ``chat(system, user, **gen)``.

Decoupling: torch and transformers are imported lazily inside ``load``; ``peft``
is imported only inside the adapter branch. Importing this module therefore does
not require the training stack — inference can run in a base-only environment.

CPU note: when no CUDA device is present (the local inference setup) the model
is loaded in float32 on CPU, because float16 generation is unreliable on CPU.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

from src.models.hf_utils import (
    chat_template_kwargs,
    from_pretrained_kwargs,
    load_pretrained_with_cache_repair,
    model_identifier,
    safe_model_access_error,
)
from src.utils.numerics import raise_if_nonfinite_parameters
from src.utils.precision import resolve_precision

logger = logging.getLogger(__name__)


def _get(cfg: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Read a key from a dict or an OmegaConf DictConfig uniformly."""
    try:
        return cfg.get(key, default)  # works for dict and DictConfig
    except AttributeError:
        return getattr(cfg, key, default)


class HFCausalModel:
    """Thin wrapper around a HuggingFace causal LM + tokenizer."""

    def __init__(self, model_cfg: Mapping[str, Any]) -> None:
        self.cfg = model_cfg
        self.model = None
        self.tokenizer = None
        self.chat_kwargs = chat_template_kwargs(model_cfg)

    # ------------------------------------------------------------------
    def load(self, adapter_path: Optional[str] = None) -> "HFCausalModel":
        """Load tokenizer + model, optionally applying a PEFT adapter."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = model_identifier(self.cfg)
        cache_dir = _get(self.cfg, "cache_dir")
        trust_remote_code = bool(_get(self.cfg, "trust_remote_code", True))
        max_seq_length = int(_get(self.cfg, "max_seq_length", 2048))

        dtype_str = _get(self.cfg, "dtype") or _get(self.cfg, "torch_dtype", "float16")
        precision = resolve_precision(dtype_str, torch, logger=logger)
        logger.info(
            "Using %s for inference (requested %s)",
            precision.effective_dtype_name,
            precision.requested_dtype,
        )
        dtype = precision.effective_dtype
        device_map = "auto" if torch.cuda.is_available() else None

        # Tokenizer comes from the adapter dir when present (it was saved there).
        tokenizer_src = adapter_path if adapter_path else name
        logger.info("Loading tokenizer: %s", tokenizer_src)
        tokenizer_kwargs = from_pretrained_kwargs(
            self.cfg, cache_dir=cache_dir, trust_remote_code=trust_remote_code
        )
        if adapter_path:
            tokenizer_kwargs.pop("revision", None)
        try:
            self.tokenizer = load_pretrained_with_cache_repair(
                AutoTokenizer.from_pretrained,
                tokenizer_src,
                kwargs=tokenizer_kwargs,
                logger=logger,
                component="inference tokenizer",
            )
        except Exception as exc:
            raise safe_model_access_error(name, exc) from exc
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.max_seq_length = max_seq_length

        logger.info("Loading model: %s (adapter=%s)", name, adapter_path)
        model_kwargs = from_pretrained_kwargs(
            self.cfg, cache_dir=cache_dir, trust_remote_code=trust_remote_code
        )
        model_kwargs.update(device_map=device_map, dtype=dtype)
        try:
            self.model = load_pretrained_with_cache_repair(
                AutoModelForCausalLM.from_pretrained,
                name,
                kwargs=model_kwargs,
                logger=logger,
                component="inference model",
            )
        except Exception as exc:
            raise safe_model_access_error(name, exc) from exc

        if adapter_path:
            # Lazy import keeps peft out of the base inference dependency set.
            from peft import PeftModel

            logger.info("Applying PEFT adapter from %s", adapter_path)
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
            # A corrupted adapter can otherwise fail much later inside
            # ``torch.multinomial`` with a misleading CUDA device-side assert.
            raise_if_nonfinite_parameters(
                self.model,
                name_contains="lora_",
                context=f"PEFT adapter {adapter_path}",
            )

        self.model.eval()
        return self

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Render messages with the profile's model-compatible chat template."""
        template_kwargs = dict(self.chat_kwargs)
        template_kwargs.update(kwargs)
        return self.tokenizer.apply_chat_template(messages, **template_kwargs)

    # ------------------------------------------------------------------
    def chat(self, system: str, user: str, **gen_kwargs: Any) -> str:
        """Run one chat turn and return the decoded assistant text."""
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        max_new = int(gen_kwargs.get("max_new_tokens", 1024))
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max(self.max_seq_length - max_new, 16),
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_args = dict(
            max_new_tokens=max_new,
            temperature=gen_kwargs.get("temperature", 0.1),
            top_p=gen_kwargs.get("top_p", 0.9),
            do_sample=gen_kwargs.get("do_sample", True),
            repetition_penalty=gen_kwargs.get("repetition_penalty", 1.15),
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        with torch.no_grad():
            out_ids = self.model.generate(**inputs, **gen_args)

        new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    def teardown(self) -> None:
        """Free GPU memory between methods/runs."""
        self.model = None
        self.tokenizer = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
