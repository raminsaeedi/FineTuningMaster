"""Shared implementation for the HuggingFace-backed methods (A and C).

Methods A (prompt-only) and C (fine-tuned) differ only in whether a PEFT adapter
is loaded; the prompt construction, generation call, JSON parsing and result
assembly are identical and live here. RAG variants (B, D) extend this contract
separately once retrieval is implemented.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional

from src.core.interfaces import BaseMethod
from src.core.prompts import SYSTEM_PROMPT, build_user_message
from src.core.schemas import DashboardBrief, GenerationResult
from src.inference.postprocess import parse_json_safe
from src.models.hf_causal import HFCausalModel


def _get(cfg: Mapping[str, Any], key: str, default: Any = None) -> Any:
    try:
        return cfg.get(key, default)
    except AttributeError:
        return getattr(cfg, key, default)


def _to_plain_dict(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:  # OmegaConf DictConfig
        from omegaconf import OmegaConf

        return OmegaConf.to_container(value, resolve=True)  # type: ignore[return-value]
    except Exception:
        return dict(value)


class HFMethod(BaseMethod):
    """Base for methods that run a single HuggingFace model locally."""

    name = "hf_base"

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.model_cfg = _get(cfg, "model", {})
        self.method_cfg = _get(cfg, "method", {})
        self.seed = int(_get(cfg, "seed", 42))
        self.config_hash = str(_get(cfg, "config_hash", ""))
        self.model: Optional[HFCausalModel] = None
        self._gen_kwargs = _to_plain_dict(_get(self.method_cfg, "generate", {}))
        self._constrained = bool(self._gen_kwargs.get("constrained", False))
        self._decoder = None

    # Subclasses override to point at an adapter folder (method C).
    def _adapter_path(self) -> Optional[str]:
        return None

    def setup(self) -> None:
        self.model = HFCausalModel(self.model_cfg)
        self.model.load(self._adapter_path())
        if self._constrained:
            from src.inference.decoders import ConstrainedDecoder

            self._decoder = ConstrainedDecoder(self._gen_kwargs.get("max_new_tokens", 1024))
            self._decoder.setup(self.model.model, self.model.tokenizer)

    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def _raw_generate(self, system: str, user: str) -> str:
        """Generate raw text, using constrained JSON decoding when enabled."""
        if self._decoder is not None:
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": user}]
            prompt = self.model.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return self._decoder.generate(prompt)
        return self.model.chat(system, user, **self._gen_kwargs)

    def generate(self, brief: DashboardBrief) -> GenerationResult:
        t0 = time.perf_counter()
        raw = self._raw_generate(self._system_prompt(), build_user_message(brief))
        parsed, err = parse_json_safe(raw)
        return GenerationResult(
            item_id=brief.item_id or "",
            method_name=self.name,
            model_name=str(_get(self.model_cfg, "name", "")),
            config_hash=self.config_hash,
            raw_text=raw,
            parsed=parsed,
            parse_error=err,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            seed=self.seed,
        )

    def teardown(self) -> None:
        if self.model is not None:
            self.model.teardown()
            self.model = None


class RAGHFMethod(HFMethod):
    """Base for retrieval-augmented methods (B and D).

    Extends the local HF method by retrieving guideline passages for each brief
    and injecting them into the system prompt. Method D additionally loads an
    adapter (via ``_adapter_path``); everything else is shared.
    """

    name = "rag_base"

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.retriever_cfg = _get(self.method_cfg, "retriever", {})
        self.top_k = int(_get(self.retriever_cfg, "top_k", 3))
        self.retriever = None

    def setup(self) -> None:
        super().setup()  # load model (+ adapter for method D)
        from src.core.registry import RETRIEVERS
        import src.retrievers  # noqa: F401  (register retrievers)

        retriever_name = str(_get(self.retriever_cfg, "name", "tfidf"))
        self.retriever = RETRIEVERS.get(retriever_name)(self.retriever_cfg)
        self.retriever.setup()

    def _brief_to_query(self, brief: DashboardBrief) -> str:
        parts = [brief.users, " ".join(brief.goals or []), " ".join(brief.kpis or [])]
        return " ".join(p for p in parts if p).strip()

    def _system_prompt_with(self, passages: list) -> str:
        from src.retrievers.base import format_passages

        if not passages:
            return SYSTEM_PROMPT
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"--- Relevant Design Guidelines ---\n"
            f"{format_passages(passages)}\n"
            f"--- End of Guidelines ---"
        )

    def generate(self, brief: DashboardBrief) -> GenerationResult:
        passages = self.retriever.retrieve(self._brief_to_query(brief), self.top_k)
        t0 = time.perf_counter()
        raw = self._raw_generate(self._system_prompt_with(passages), build_user_message(brief))
        parsed, err = parse_json_safe(raw)
        return GenerationResult(
            item_id=brief.item_id or "",
            method_name=self.name,
            model_name=str(_get(self.model_cfg, "name", "")),
            config_hash=self.config_hash,
            raw_text=raw,
            parsed=parsed,
            parse_error=err,
            retrieved_docs=passages,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            seed=self.seed,
        )
