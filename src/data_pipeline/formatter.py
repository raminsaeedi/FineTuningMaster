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


def build_prompt(brief: BriefLike, tokenizer=None) -> str:
    """Render the prompt up to (and including) the assistant-turn opener."""
    messages = build_messages(brief)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
) -> str:
    """Full SFT text = prompt + reference JSON + EOS."""
    prompt = build_prompt(brief, tokenizer)
    response_json = json.dumps(_rec_to_dict(recommendation), ensure_ascii=False, indent=2)

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        eos = getattr(tokenizer, "eos_token", "") or ""
        return prompt + response_json + eos
    return prompt + response_json + "\n"
