"""Turn gold items into the single training-text string SFT expects.

The formatter reuses the exact prompt builder used at inference time
(``src.core.prompts``) so the model trains on the same prompt format it will
later see. The completion is the pretty-printed reference JSON followed by the
tokenizer's end-of-sequence token.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Union

from src.core.prompts import SYSTEM_PROMPT, build_messages, build_user_message
from src.core.schemas import DashboardBrief, DesignOutput

BriefLike = Union[DashboardBrief, Dict[str, Any]]
RecLike = Union[DesignOutput, Dict[str, Any]]


def _rec_to_dict(recommendation: RecLike) -> Dict[str, Any]:
    if isinstance(recommendation, DesignOutput):
        # mode="json" serialises enums (TaskType/ChartType) to their string values.
        return recommendation.model_dump(mode="json")
    return dict(recommendation)


def build_prompt(brief: BriefLike, tokenizer=None, chat_template_kwargs=None) -> str:
    """Render the prompt up to (and including) the assistant-turn opener."""
    messages = build_messages(brief)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        kwargs = dict(chat_template_kwargs or {})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
    # Fallback for tokenizers without a chat template.
    return (
        f"### System:\n{SYSTEM_PROMPT}\n\n"
        f"### User:\n{build_user_message(brief)}\n\n"
        f"### Assistant:\n"
    )


def format_training_example(
    brief: BriefLike,
    recommendation: RecLike,
    tokenizer=None,
    chat_template_kwargs=None,
) -> str:
    """Full SFT text = prompt + reference JSON + EOS."""
    prompt = build_prompt(brief, tokenizer, chat_template_kwargs)
    response_json = json.dumps(_rec_to_dict(recommendation), ensure_ascii=False, indent=2)

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        eos = getattr(tokenizer, "eos_token", "") or ""
        return prompt + response_json + eos
    return prompt + response_json + "\n"


def split_training_text(text: str, eos_token: str = "") -> Dict[str, str]:
    """Recover TRL prompt-completion fields from this module's full SFT text.

    The gold completion is the final JSON object. Parsing candidates instead of
    splitting on the first ``{`` keeps JSON examples inside the prompt intact.
    Returned fields concatenate to the original text exactly.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("Training text must be a non-empty string.")

    content_end = len(text)
    if eos_token and text.endswith(eos_token):
        content_end -= len(eos_token)
    json_end = len(text[:content_end].rstrip())
    decoder = json.JSONDecoder()

    for start in range(json_end - 1, -1, -1):
        if text[start] != "{":
            continue
        candidate = text[start:json_end]
        try:
            value, consumed = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and not candidate[consumed:].strip():
            return {"prompt": text[:start], "completion": text[start:]}

    raise ValueError(
        "Cannot create prompt-completion training data: final response is not "
        "a valid JSON object. Keep completion_only_loss=false for legacy data "
        "or regenerate the formatted dataset."
    )
