"""Prompt construction — pure text, no heavy dependencies.

Single source of truth for turning a brief into a chat prompt. Both the training
formatter and the inference methods import it, so train-time and inference-time
prompts match. The user message lists the required output keys and the allowed
task/chart vocabularies, giving even the prompt-only model a fair chance at the
typed schema.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

from src.core.constants import CHART_TYPES, TASK_TYPES
from src.core.schemas import DashboardBrief

BriefLike = Union[DashboardBrief, Dict[str, Any]]

SYSTEM_PROMPT = (
    "You are an expert dashboard design consultant.\n"
    "Given a dashboard brief, you generate structured, professional design "
    "recommendations.\n"
    "Always respond with a single valid JSON object following the exact schema "
    "provided.\n"
    "Do not wrap the JSON in markdown fences. Do not add commentary outside the "
    "JSON object."
)


def _as_dict(brief: BriefLike) -> Dict[str, Any]:
    if isinstance(brief, DashboardBrief):
        return brief.model_dump(mode="json")
    return dict(brief)


def _fmt_list(values) -> str:
    if not values:
        return "N/A"
    return ", ".join(str(v) for v in values)


def _fmt_columns(columns) -> str:
    if not columns:
        return "N/A"
    parts = []
    for c in columns:
        if isinstance(c, dict):
            parts.append(f"{c.get('name', '?')} ({c.get('dtype', '?')})")
        else:
            parts.append(str(c))
    return ", ".join(parts)


def build_user_message(brief: BriefLike) -> str:
    """Render a brief into the user-turn instruction text."""
    b = _as_dict(brief)
    lines = [
        "Generate one dashboard recommendation for this brief.",
        f"Users: {b.get('users', 'N/A')}",
        f"Goals: {_fmt_list(b.get('goals', []))}",
        f"KPIs: {_fmt_list(b.get('kpis', []))}",
        f"Data columns: {_fmt_columns(b.get('columns', []))}",
        f"Constraints: {b.get('constraints') or 'None'}",
        "",
        "Return only one JSON object with exactly this shape and value types. "
        "Every field must be filled in, never left as {} or []:",
        '{"context_summary":{"goal":"track signups"},'
        '"kpi_chart_mapping":[{"kpi":"",'
        '"task_type":"trend","chart_type":"line","alternatives":["bar"],'
        '"encoding":{"x":"month","y":"signups"}}],'
        '"layout":{"summary":"one chart at the top"},'
        '"styling":{"theme":"minimal"},"interactions":["filter by region"],'
        '"rationales":[{"claim":"line chart shows change over time",'
        '"principle":"trend visibility"}]}',
        "All six top-level fields are required and non-empty. context_summary, "
        "layout, styling, and encoding are objects. kpi_chart_mapping, "
        "alternatives, interactions, and rationales are arrays.",
        "Each kpi_chart_mapping item requires kpi, task_type, chart_type, "
        "alternatives, encoding.",
        "Each rationale requires claim and principle.",
        f"Allowed task_type: {', '.join(TASK_TYPES)}.",
        f"Allowed chart_type: {', '.join(CHART_TYPES)}.",
    ]
    return "\n".join(lines)


def build_messages(brief: BriefLike, system_prompt: str = SYSTEM_PROMPT) -> List[Dict[str, str]]:
    """Return chat messages ready for ``tokenizer.apply_chat_template``."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_message(brief)},
    ]
