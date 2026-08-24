"""Shared implementation for the HuggingFace-backed methods (A and C).

Methods A (prompt-only) and C (fine-tuned) differ only in whether a PEFT adapter
is loaded; the prompt construction, generation call, JSON parsing and result
assembly are identical and live here. RAG variants (B, D) extend this contract
separately once retrieval is implemented.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping, Optional

from src.core.interfaces import BaseMethod
from src.core.prompts import SYSTEM_PROMPT, build_user_message
from src.core.schemas import DashboardBrief, GenerationResult
from src.inference.postprocess import parse_json_safe
from src.models.hf_causal import HFCausalModel

logger = logging.getLogger(__name__)


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
        inference_cfg = _get(self.method_cfg, "inference", {})
        self.model = HFCausalModel(self.model_cfg, inference_cfg)
        self.model.load(self._adapter_path())
        if self._constrained:
            from src.inference.decoders import ConstrainedDecoder

            self._decoder = ConstrainedDecoder(self._gen_kwargs.get("max_new_tokens", 1024))
            self._decoder.setup(self.model.model, self.model.tokenizer)

    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def _raw_generate(
        self,
        system: str,
        user: str,
        prepared_inputs: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Generate raw text, using constrained JSON decoding when enabled."""
        if self._decoder is not None:
            prompt, _inputs, _tokens, _budget = self.model.prepare_prompt(
                system,
                user,
                int(self._gen_kwargs.get("max_new_tokens", 1024)),
            )
            return self._decoder.generate(prompt)
        if prepared_inputs is not None:
            return self.model.generate_prepared(prepared_inputs, **self._gen_kwargs)
        return self.model.chat(system, user, **self._gen_kwargs)

    def generate(self, brief: DashboardBrief) -> GenerationResult:
        system = self._system_prompt()
        user = build_user_message(brief)
        max_new = int(self._gen_kwargs.get("max_new_tokens", 1024))
        prepared_inputs = None
        if self._decoder is not None:
            prompt_tokens = self.model.prompt_token_count(system, user)
            prompt_budget = self.model.input_token_budget(max_new)
        else:
            _prompt, prepared_inputs, prompt_tokens, prompt_budget = self.model.prepare_prompt(
                system, user, max_new
            )
        t0 = time.perf_counter()
        raw = self._raw_generate(system, user, prepared_inputs)
        parsed, err = parse_json_safe(raw)
        return GenerationResult(
            item_id=brief.item_id or "",
            method_name=self.name,
            model_name=str(_get(self.model_cfg, "name", "")),
            config_hash=self.config_hash,
            raw_text=raw,
            parsed=parsed,
            parse_error=err,
            prompt_input_tokens=prompt_tokens,
            prompt_input_budget=prompt_budget,
            rag_context_truncated=False,
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

    def _fit_system_prompt(self, passages: list, user: str) -> tuple[str, bool]:
        """Fit equal token excerpts from every retrieved passage into the prompt."""
        full_system = self._system_prompt_with(passages)
        max_new = int(self._gen_kwargs.get("max_new_tokens", 1024))
        budget = self.model.input_token_budget(max_new)
        full_tokens = self.model.prompt_token_count(full_system, user)
        if full_tokens <= budget:
            return full_system, False

        tokenizer = self.model.tokenizer
        passage_token_ids = [
            tokenizer(str(p.get("text", "")), add_special_tokens=False)["input_ids"]
            for p in passages
        ]

        def system_with_limit(per_passage_limit: int) -> str:
            clipped = []
            for passage, token_ids in zip(passages, passage_token_ids):
                item = dict(passage)
                item["text"] = tokenizer.decode(
                    token_ids[:per_passage_limit], skip_special_tokens=True
                ).strip()
                clipped.append(item)
            return self._system_prompt_with(clipped)

        headings_only = system_with_limit(0)
        if self.model.prompt_token_count(headings_only, user) > budget:
            raise ValueError(
                "Prompt and RAG passage headings exceed the input-token budget; "
                "shorten the base prompt or increase max_seq_length."
            )

        low = 0
        high = max((len(ids) for ids in passage_token_ids), default=0)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = system_with_limit(middle)
            if self.model.prompt_token_count(candidate, user) <= budget:
                low = middle
            else:
                high = middle - 1

        fitted = system_with_limit(low)
        fitted_tokens = self.model.prompt_token_count(fitted, user)
        logger.info(
            "RAG context fitted to prompt budget: %d -> %d tokens; "
            "%d text tokens per passage across %d passages",
            full_tokens,
            fitted_tokens,
            low,
            len(passages),
        )
        return fitted, True

    def generate(self, brief: DashboardBrief) -> GenerationResult:
        passages = self.retriever.retrieve(self._brief_to_query(brief), self.top_k)
        user = build_user_message(brief)
        system, context_truncated = self._fit_system_prompt(passages, user)
        max_new = int(self._gen_kwargs.get("max_new_tokens", 1024))
        prepared_inputs = None
        if self._decoder is not None:
            prompt_tokens = self.model.prompt_token_count(system, user)
            prompt_budget = self.model.input_token_budget(max_new)
        else:
            _prompt, prepared_inputs, prompt_tokens, prompt_budget = self.model.prepare_prompt(
                system, user, max_new
            )
        t0 = time.perf_counter()
        raw = self._raw_generate(system, user, prepared_inputs)
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
            prompt_input_tokens=prompt_tokens,
            prompt_input_budget=prompt_budget,
            rag_context_truncated=context_truncated,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            seed=self.seed,
        )
