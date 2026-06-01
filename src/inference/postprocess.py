"""Parse possibly-messy model output into a ``DesignOutput``.

Two-stage approach:
1. JSON extraction — handles code fences, prose wrappers, control characters.
2. Lenient normalisation — before Pydantic validation, normalise near-miss enum
   values (typos, synonyms) so the model is not penalised for spelling errors when
   it understood the concept correctly:
     - task_type: fuzzy-match against the 9 TaskType values (e.g. 'comparision' → 'comparison')
     - chart_type: same for ChartType values
     - alternatives: silently drop entries that are not valid chart tokens
     - interactions: accept list-of-dicts (richer than spec but not wrong)

Genuinely bad JSON (missing braces, unterminated strings) is still reported as
``no_json_found`` — this is a real model failure worth measuring separately.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from src.core.schemas import ChartType, DesignOutput, TaskType

# ── enum vocabularies ──────────────────────────────────────────────────────────
_TASK_VALUES: List[str] = [t.value for t in TaskType]
_CHART_VALUES: List[str] = [c.value for c in ChartType]

# Common synonyms / alternate spellings the 0.5B uses that are NOT in the enum.
_TASK_ALIASES: Dict[str, str] = {
    "comparision": "comparison",
    "comparative": "comparison",
    "compoation": "composition",
    "part-to-whole": "part_to_whole",
    "part to whole": "part_to_whole",
    "temporal": "trend",
    "time series": "trend",
    "deviation analysis": "deviation",
    "rank": "ranking",
}
_CHART_ALIASES: Dict[str, str] = {
    "kpi card": "kpi_card",
    "kpi_card / scorecard": "kpi_card",
    "kpi card / scorecard": "kpi_card",
    "scorecard": "kpi_card",
    "grouped bar": "grouped_bar",
    "stacked bar": "stacked_bar",
    "line graph": "line",
    "bar graph": "bar",
    "column chart": "bar",
    "column": "bar",
    "bubble": "scatter",
    "funnel": "sankey",    # 'funnel' not in ChartType; closest is sankey
    "waterfall": "bar",    # 'waterfall' not in ChartType; closest is bar
    "bullet": "gauge",     # 'bullet' not in ChartType; closest is gauge
}


def _normalise_enum(raw: Any, valid_values: List[str], aliases: Dict[str, str]) -> Optional[str]:
    """Try to map ``raw`` to a valid enum string. Returns None if hopeless."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in valid_values:
        return s
    if s in aliases:
        return aliases[s]
    # Prefix match — handles 'comparison' when model wrote 'comparison-based'
    for v in valid_values:
        if s.startswith(v) or v.startswith(s):
            return v
    return None


def _normalise_mapping(m: Any) -> Optional[dict]:
    """Return a normalised kpi_chart_mapping entry or None to drop it."""
    if not isinstance(m, dict):
        return None
    task_raw = m.get("task_type")
    chart_raw = m.get("chart_type")
    task = _normalise_enum(task_raw, _TASK_VALUES, _TASK_ALIASES)
    chart = _normalise_enum(chart_raw, _CHART_VALUES, _CHART_ALIASES)
    if task is None or chart is None:
        return None  # can't score this entry — drop it
    # Silently keep only alternatives that are valid chart tokens.
    raw_alts = m.get("alternatives", []) or []
    if isinstance(raw_alts, dict):
        raw_alts = []  # model put a dict here — just discard
    alts = [
        v for a in raw_alts
        if isinstance(a, str)
        and (v := _normalise_enum(a, _CHART_VALUES, _CHART_ALIASES)) is not None
    ]
    out = dict(m)
    out["task_type"] = task
    out["chart_type"] = chart
    out["alternatives"] = alts
    return out


def _normalise_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Apply lenient normalisation before Pydantic validation."""
    out = dict(obj)
    # kpi_chart_mapping: normalise each entry, drop those that can't be fixed.
    raw_mapping = out.get("kpi_chart_mapping")
    if isinstance(raw_mapping, list):
        out["kpi_chart_mapping"] = [n for m in raw_mapping if (n := _normalise_mapping(m)) is not None]
    # interactions: accept list-of-dicts by extracting a string label if present;
    # otherwise keep as-is (List[Any] in schema now accepts both).
    raw_ix = out.get("interactions")
    if isinstance(raw_ix, dict):
        # Model returned a single dict instead of a list — wrap it.
        out["interactions"] = [raw_ix]
    # rationales: accept list-of-dicts or list-of-strings uniformly.
    raw_rat = out.get("rationales")
    if isinstance(raw_rat, list):
        fixed = []
        for r in raw_rat:
            if isinstance(r, str):
                fixed.append({"claim": r, "principle": ""})
            elif isinstance(r, dict):
                fixed.append(r)
        out["rationales"] = fixed
    return out


# ── JSON extraction ────────────────────────────────────────────────────────────

def _try_parse(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text, strict=False)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def extract_json_dict(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object from raw model text."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        obj = _try_parse(fenced.group(1))
        if obj is not None:
            return obj
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        obj = _try_parse(brace.group(0))
        if obj is not None:
            return obj
    return _try_parse(text.strip())


def parse_json_safe(raw: str) -> Tuple[Optional[DesignOutput], Optional[str]]:
    """Extract JSON, normalise, validate. Return ``(DesignOutput, None)`` or ``(None, error)``."""
    obj = extract_json_dict(raw)
    if obj is None:
        return None, "no_json_found"
    try:
        normalised = _normalise_output(obj)
        return DesignOutput(**normalised), None
    except Exception as exc:
        return None, f"schema_error: {exc}"


def reparse(result):
    """Re-run the current parser over cached raw_text (parser improvements apply retroactively)."""
    parsed, err = parse_json_safe(result.raw_text)
    result.parsed = parsed
    result.parse_error = err
    return result
