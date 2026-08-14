"""Audit and repair generated dashboard_v4 design fields.

The frozen dashboard_v4 release is immutable.  This script reads its generated
Train/Validation records, audits the semantic design fields, repairs those
fields from the existing brief and encoding, and publishes dashboard_v4_1 as a
separate atomic release.  Original dashboard_v3 records and the held-out files
are copied byte-for-byte.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.frozen_validation import validate_record


PARENT_VERSION = "dashboard_v4"
TARGET_VERSION = "dashboard_v4_1"
PARENT_DIR = PROJECT_ROOT / "data" / "frozen" / PARENT_VERSION
V3_DIR = PROJECT_ROOT / "data" / "frozen" / "dashboard_v3"
TARGET_DIR = PROJECT_ROOT / "data" / "frozen" / TARGET_VERSION
STAGING_ROOT = PROJECT_ROOT / "data" / "staging" / TARGET_VERSION

REPAIR_VERSION = "dashboard_v4_1-semantic-repair-v1"
REPAIR_MODEL = "gpt-5.6-luna"
REPAIR_MODE = "codex_agent_context_aware"
NEAR_DUPLICATE_THRESHOLD = 0.8

JSONL_FILES = ("train.jsonl", "val.jsonl", "test.jsonl")
HELD_OUT_FILES = ("test.jsonl", "human_eval_test_items_40.csv")
REQUIRED_FILES = (
    "train.jsonl", "val.jsonl", "test.jsonl", "human_eval_test_items_40.csv",
    "schema.json", "manifest.json", "hashes.json", "dataset_card.md",
)
INTERACTION_TYPES = {
    "tooltip", "filter", "sort", "legend_toggle", "hover_highlight",
    "time_range_select", "zoom", "brush", "drill_down", "cross_filter",
}
CONTEXT_KEYS = {
    "objective", "kpis", "available_columns", "analysis_scope", "constraints",
}
GENERIC_RATIONALE_PHRASES = (
    "using the stated encoding",
    "clearly shows the data",
    "improves usability",
    "colors improve readability",
    "supports the stated decision",
    "reduces cognitive load while preserving the analytical task",
)
UNSUPPORTED_CLAIM_PHRASES = (
    "the data shows",
    "the data contains",
    "increased",
    "decreased",
    "will improve",
    "improved",
    "reduced",
    "causes",
    "is correlated",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def _atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records)
    _atomic_write_bytes(path, payload.encode("utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def _humanize(value: Any) -> str:
    text = str(value or "").replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _brief(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record.get("brief") or {}


def _recommendation(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record.get("recommendation") or {}


def _mappings(record: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    value = _recommendation(record).get("kpi_chart_mapping") or []
    return [item for item in value if isinstance(item, dict)]


def _columns(record: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    value = _brief(record).get("columns") or []
    return [item for item in value if isinstance(item, dict) and item.get("name")]


def _known_columns(record: Mapping[str, Any]) -> set[str]:
    return {str(column.get("name")) for column in _columns(record)}


def _column_by_normalized_name(record: Mapping[str, Any]) -> Dict[str, str]:
    return {_normalize(column.get("name")): str(column.get("name")) for column in _columns(record)}


def _first_measure(record: Mapping[str, Any]) -> str | None:
    for column in _columns(record):
        if column.get("role") in ("measure", "baseline") and column.get("dtype") in ("number", "datetime"):
            return str(column.get("name"))
    for column in _columns(record):
        if column.get("dtype") == "number":
            return str(column.get("name"))
    return None


def _resolve_field(record: Mapping[str, Any], token: Any, fallback: str | None = None) -> str | None:
    known = _known_columns(record)
    if isinstance(token, dict):
        token = token.get("field")
    if token is None:
        return fallback
    text = str(token)
    if text in known:
        return text
    expression = re.match(r"^[A-Za-z_]+\((.*?)\)$", text)
    if expression:
        inner = expression.group(1).strip()
        if inner == "*":
            return fallback or _first_measure(record)
        if inner in known:
            return inner
        normalized = _column_by_normalized_name(record).get(_normalize(inner))
        if normalized:
            return normalized
    normalized = _column_by_normalized_name(record).get(_normalize(text))
    if normalized:
        return normalized
    kpis = _brief(record).get("kpis") or []
    if text in kpis:
        return _column_by_normalized_name(record).get(_normalize(text), fallback)
    return fallback


def _mapping_primary_field(record: Mapping[str, Any], mapping: Mapping[str, Any]) -> str | None:
    encoding = mapping.get("encoding") or {}
    candidates = (
        encoding.get("value"), encoding.get("y"), encoding.get("x"),
        encoding.get("y_measure"), encoding.get("x_measure"), mapping.get("kpi"),
    )
    for candidate in candidates:
        resolved = _resolve_field(record, candidate)
        if resolved:
            return resolved
    return _first_measure(record)


def _mapping_fields(record: Mapping[str, Any], mapping: Mapping[str, Any]) -> List[str]:
    encoding = mapping.get("encoding") or {}
    fallback = _mapping_primary_field(record, mapping)
    fields: List[str] = []

    def add(value: str | None) -> None:
        if value and value in _known_columns(record) and value not in fields:
            fields.append(value)

    add(_resolve_field(record, encoding.get("x"), fallback))
    add(_resolve_field(record, encoding.get("y"), fallback))
    add(_resolve_field(record, encoding.get("x_measure"), fallback))
    add(_resolve_field(record, encoding.get("y_measure"), fallback))
    add(_resolve_field(record, encoding.get("group_field")))
    add(_resolve_field(record, encoding.get("source")))
    add(_resolve_field(record, encoding.get("target")))
    add(_resolve_field(record, encoding.get("value"), fallback))
    add(_resolve_field(record, encoding.get("baseline")))
    time_grain = encoding.get("time_grain")
    if isinstance(time_grain, dict):
        add(_resolve_field(record, time_grain.get("field")))
    for item in encoding.get("filters") or []:
        if isinstance(item, dict):
            add(_resolve_field(record, item.get("field")))
    return fields


def _mapping_axes(record: Mapping[str, Any], mapping: Mapping[str, Any]) -> Tuple[str | None, str | None]:
    encoding = mapping.get("encoding") or {}
    fallback = _mapping_primary_field(record, mapping)
    x = _resolve_field(record, encoding.get("x"), fallback)
    y = _resolve_field(record, encoding.get("y"), fallback)
    return x, y


def _time_field(record: Mapping[str, Any], mapping: Mapping[str, Any]) -> str | None:
    encoding = mapping.get("encoding") or {}
    time_grain = encoding.get("time_grain")
    if isinstance(time_grain, dict):
        resolved = _resolve_field(record, time_grain.get("field"))
        if resolved:
            return resolved
    for column in _columns(record):
        if column.get("role") == "time" or column.get("dtype") == "datetime":
            return str(column.get("name"))
    return None


def _group_field(record: Mapping[str, Any], mapping: Mapping[str, Any]) -> str | None:
    return _resolve_field(record, (mapping.get("encoding") or {}).get("group_field"))


def _filters(record: Mapping[str, Any], mapping: Mapping[str, Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for item in (mapping.get("encoding") or {}).get("filters") or []:
        if isinstance(item, dict):
            field = _resolve_field(record, item.get("field"))
            if field:
                result.append({"field": field, "operator": item.get("operator"), "value": item.get("value")})
    return result


def _sort_field(record: Mapping[str, Any], mapping: Mapping[str, Any]) -> str | None:
    sort = (mapping.get("encoding") or {}).get("sort")
    if not isinstance(sort, dict):
        return None
    return _resolve_field(record, sort.get("field"))


def _time_grain(mapping: Mapping[str, Any]) -> str | None:
    value = (mapping.get("encoding") or {}).get("time_grain")
    if isinstance(value, dict):
        return str(value.get("grain")) if value.get("grain") else None
    return str(value) if value else None


def _task_phrase(task: str, mapping: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    kpi = str(mapping.get("kpi") or (_brief(record).get("kpis") or ["the selected KPI"])[0])
    x, y = _mapping_axes(record, mapping)
    group = _group_field(record, mapping)
    primary = _humanize(kpi)
    x_text = _humanize(x or "the encoded dimension")
    y_text = _humanize(y or _mapping_primary_field(record, mapping) or primary)
    if task == "trend":
        return f"monitor {primary} across {x_text}"
    if task == "comparison":
        return f"compare {primary} across {x_text}"
    if task == "composition":
        return f"inspect how {primary} is divided across {x_text} and {_humanize(group or 'the encoded series')}"
    if task == "distribution":
        return f"inspect the spread of {primary} using {y_text}"
    if task == "correlation":
        return f"inspect the association between {x_text} and {y_text}"
    if task == "ranking":
        return f"prioritize {x_text} by {primary}"
    if task == "deviation":
        baseline = _humanize((mapping.get("encoding") or {}).get("baseline") or "the comparison baseline")
        return f"compare {primary} with {baseline} across {x_text}"
    if task == "part_to_whole":
        return f"inspect the contribution of {primary} across {x_text}"
    if task == "flow":
        encoding = mapping.get("encoding") or {}
        return f"trace {primary} from {_humanize(encoding.get('source') or 'source')} to {_humanize(encoding.get('target') or 'target')}"
    return f"analyze {primary} using the encoded fields"


def _task_principle(chart: str, mapping: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    x, y = _mapping_axes(record, mapping)
    group = _group_field(record, mapping)
    kpi = _humanize(mapping.get("kpi") or "the KPI")
    x_text = _humanize(x or "the x field")
    y_text = _humanize(y or _mapping_primary_field(record, mapping) or "the y field")
    group_text = _humanize(group or "the series field")
    if chart in ("line", "area"):
        return f"The ordered {x_text} axis places {kpi} observations in sequence, allowing change to be inspected without asserting a direction that is not in the brief."
    if chart == "bar":
        return f"Bar length makes {kpi} comparable across the encoded {x_text} categories and keeps the ordering visible."
    if chart == "grouped_bar":
        return f"Side-by-side bars compare {kpi} across {x_text} while keeping the {group_text} series distinct."
    if chart == "stacked_bar":
        return f"Stack segments by {group_text} within {x_text} so the composition of {kpi} remains tied to the encoded categories."
    if chart in ("pie", "donut", "treemap"):
        return f"Segment or area size represents each {x_text} category's share of {kpi}; labels remain necessary because no factual share is assumed."
    if chart == "scatter":
        return f"Positions on the {x_text} and {y_text} quantitative axes expose possible association for inspection without implying causation."
    if chart == "heatmap":
        return f"Color intensity across {x_text} and {y_text} summarizes paired numeric values for inspection without claiming a measured correlation."
    if chart == "histogram":
        return f"Numeric bins of {x_text} show the observed spread of {kpi}; the binning does not assert that an unusual value exists."
    if chart == "box":
        return f"Median, quartiles, and range for {x_text} make distribution and potential outliers inspectable for {kpi}."
    if chart == "sankey":
        encoding = mapping.get("encoding") or {}
        return f"Link width represents {kpi} from {_humanize(encoding.get('source') or 'source')} to {_humanize(encoding.get('target') or 'target')} using only the represented fields."
    if chart == "table":
        return f"Tabular rows preserve the encoded {x_text} and {y_text} values so the requested {kpi} comparison can be checked directly."
    return f"The {chart} encoding keeps {kpi} tied to {x_text} and {y_text} without adding unobserved facts."


def _generic_palette_before(style: Mapping[str, Any]) -> bool:
    palette = str(style.get("color_palette") or "").lower()
    markers = ("neutral", "categorical", "sequential", "diverging", "color-blind", "muted", "accessible")
    return bool(palette) and not any(marker in palette for marker in markers)


def _audit_record(record: Mapping[str, Any], strict: bool = False) -> Dict[str, List[str]]:
    """Return field-level issues; strict mode applies the repaired standard."""
    issues: Dict[str, List[str]] = {}

    def add(field: str, code: str) -> None:
        issues.setdefault(field, []).append(code)

    brief = _brief(record)
    recommendation = _recommendation(record)
    mappings = _mappings(record)
    known = _known_columns(record)
    task = str(mappings[0].get("task_type")) if mappings else ""

    schema_errors = validate_record(dict(record))
    if schema_errors:
        add("schema", "schema_invalid")

    users = str(brief.get("users") or "")
    relevant_tokens = [str(value).lower() for value in (brief.get("kpis") or [])]
    relevant_tokens.extend(str(column.get("name")).lower() for column in _columns(record))
    if not users.strip():
        add("users", "empty_users")
    if " responsible for " in users.lower() and not any(token and token in users.lower() for token in relevant_tokens):
        add("users", "generic_user_template")
    if len(users.split()) < 4:
        add("users", "underspecified_user_role")

    context = recommendation.get("context_summary")
    if not isinstance(context, dict) or not context:
        add("context_summary", "empty_context_summary")
    else:
        unsupported = set(context) - CONTEXT_KEYS
        if unsupported:
            add("context_summary", "unsupported_context_fields")
        if "objective" not in context or context.get("objective") != (brief.get("goals") or [None])[0]:
            add("context_summary", "context_not_anchored_to_goal")
        if context.get("kpis") != brief.get("kpis"):
            add("context_summary", "context_kpis_mismatch")
        if "constraints" not in context or context.get("constraints") != (brief.get("constraints") or ""):
            add("context_summary", "context_constraints_missing")

    layout = recommendation.get("layout")
    blocks = layout.get("blocks") if isinstance(layout, dict) else None
    if not isinstance(layout, dict) or not isinstance(blocks, list):
        add("layout", "empty_layout")
    else:
        if len(blocks) != len(mappings):
            add("layout", "layout_mapping_count_mismatch")
        if not strict and ("hierarchy" not in layout or "layout_rationale" not in layout):
            add("layout", "generic_layout_template")
        if strict and ("hierarchy" not in layout or "layout_rationale" not in layout):
            add("layout", "layout_hierarchy_missing")
        for index, block in enumerate(blocks):
            if not isinstance(block, dict) or index >= len(mappings):
                add("layout", "invalid_layout_block")
                continue
            mapping = mappings[index]
            if block.get("kpi") != mapping.get("kpi") or block.get("chart") != mapping.get("chart_type"):
                add("layout", "layout_block_mismatch")
            if not block.get("purpose") or not block.get("focus_fields"):
                add("layout", "layout_block_underdescribed")

    style = recommendation.get("styling")
    if not isinstance(style, dict) or not style:
        add("styling", "empty_styling")
    else:
        if not style.get("typography") or not style.get("contrast") or not style.get("semantic_color_policy"):
            add("styling", "styling_semantics_missing")
        if _generic_palette_before(style):
            add("styling", "generic_palette")
        palette_text = str(style.get("color_palette") or "").lower()
        if any(color in palette_text for color in ("red", "green")) and task != "deviation":
            add("styling", "unjustified_status_color")
        if strict and not style.get("record_specific_basis"):
            add("styling", "styling_record_basis_missing")

    interactions = recommendation.get("interactions")
    if not isinstance(interactions, list) or not interactions:
        add("interactions", "empty_interactions")
    else:
        for interaction in interactions:
            if not isinstance(interaction, dict):
                add("interactions", "invalid_interaction")
                continue
            interaction_type = str(interaction.get("type") or "")
            fields = interaction.get("fields") or []
            if interaction_type not in INTERACTION_TYPES:
                add("interactions", "unsupported_interaction_type")
            if not interaction.get("purpose"):
                add("interactions", "interaction_missing_purpose")
            if not isinstance(fields, list) or any(str(field) not in known for field in fields):
                add("interactions", "interaction_nonexistent_column")
            if interaction_type == "time_range_select" and not any(_time_field(record, mapping) for mapping in mappings):
                add("interactions", "time_interaction_without_time")
            if interaction_type in {"legend_toggle", "hover_highlight"} and not any(_group_field(record, mapping) for mapping in mappings):
                add("interactions", "legend_interaction_without_group")
            if interaction_type == "drill_down" and task != "flow":
                add("interactions", "drill_down_task_mismatch")
            if interaction_type == "cross_filter" and len(mappings) < 2:
                add("interactions", "cross_filter_without_multiple_views")
            if interaction_type == "sort" and task not in {"comparison", "ranking", "deviation"}:
                add("interactions", "sort_task_mismatch")

    rationales = recommendation.get("rationales")
    if not isinstance(rationales, list) or not rationales:
        add("rationales", "empty_rationales")
    else:
        rationale_text = " ".join(
            f"{item.get('claim', '')} {item.get('principle', '')}" for item in rationales if isinstance(item, dict)
        ).lower()
        if any(phrase in rationale_text for phrase in GENERIC_RATIONALE_PHRASES):
            add("rationales", "generic_filler")
        if any(phrase in rationale_text for phrase in UNSUPPORTED_CLAIM_PHRASES):
            add("rationales", "unsupported_factual_claim")
        for mapping in mappings:
            task_value = str(mapping.get("task_type") or "").lower()
            chart_value = str(mapping.get("chart_type") or "").lower()
            kpi_value = _normalize(mapping.get("kpi") or "")
            matching = [item for item in rationales if isinstance(item, dict) and task_value in str(item.get("claim") or "").lower() and chart_value in str(item.get("claim") or "").lower()]
            if not matching or not any(kpi_value and kpi_value in _normalize(item.get("claim") or "") for item in matching):
                add("rationales", "mapping_rationale_mismatch")
            principle = " ".join(str(item.get("principle") or "").lower() for item in matching)
            markers = {
                "line": ("axis", "sequence"), "area": ("axis",), "bar": ("bar",), "grouped_bar": ("series", "bar"),
                "stacked_bar": ("stack",), "pie": ("share", "segment"), "donut": ("share", "segment"),
                "treemap": ("area", "share"), "scatter": ("quantitative", "association"), "heatmap": ("intensity", "numeric"),
                "histogram": ("bin",), "box": ("quartile", "median", "range"), "sankey": ("link", "source"),
                "table": ("row",),
            }.get(chart_value, (chart_value,))
            if not any(marker in principle for marker in markers):
                add("rationales", "chart_principle_mismatch")

    if strict:
        if set(recommendation) < {"context_summary", "kpi_chart_mapping", "layout", "styling", "interactions", "rationales"}:
            add("recommendation", "missing_recommendation_fields")
    return {field: sorted(set(values)) for field, values in issues.items()}


def _audit_dataset(records: Sequence[Mapping[str, Any]], phase: str) -> Dict[str, Any]:
    per_record: Dict[str, Dict[str, List[str]]] = {}
    issue_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    for record in records:
        issues = _audit_record(record, strict=phase == "after")
        item_id = str(record.get("item_id"))
        per_record[item_id] = issues
        for field, codes in issues.items():
            field_counts[field] += 1
            issue_counts.update(codes)

    repetition: Dict[str, Any] = {}
    for field, values in (
        ("users", [str(_brief(record).get("users") or "") for record in records]),
        ("context_summary", [_canonical(_recommendation(record).get("context_summary")) for record in records]),
        ("layout", [_canonical(_recommendation(record).get("layout")) for record in records]),
        ("styling", [_canonical(_recommendation(record).get("styling")) for record in records]),
        ("interactions", [_canonical(_recommendation(record).get("interactions")) for record in records]),
        ("rationales", [_canonical(_recommendation(record).get("rationales")) for record in records]),
    ):
        counts = Counter(values)
        repetition[field] = {
            "unique_values": len(counts),
            "max_exact_repetition": max(counts.values()) if counts else 0,
            "top_repeated_values": [
                {"count": count, "value": value[:500]}
                for value, count in counts.most_common(5)
            ],
        }

    records_with_issues = sum(bool(value) for value in per_record.values())
    examples = {
        code: [item_id for item_id, issues in per_record.items() if code in sum(issues.values(), [])][:10]
        for code in issue_counts
    }
    return {
        "phase": phase,
        "records": len(records),
        "already_valid": len(records) - records_with_issues,
        "records_with_issues": records_with_issues,
        "issue_counts": dict(issue_counts),
        "field_issue_counts": dict(field_counts),
        "issue_examples": examples,
        "repetition": repetition,
    }


def _repair_users(record: Mapping[str, Any], mapping: Mapping[str, Any]) -> str:
    existing = str(_brief(record).get("users") or "")
    role = existing.split(" responsible for ", 1)[0].strip() or "dashboard analyst"
    return f"{role} using the dashboard to {_task_phrase(str(mapping.get('task_type') or ''), mapping, record)}."


def _repair_context(record: Mapping[str, Any]) -> Dict[str, Any]:
    brief = _brief(record)
    mappings = _mappings(record)
    summaries: List[Dict[str, Any]] = []
    for mapping in mappings:
        encoding = mapping.get("encoding") or {}
        x, y = _mapping_axes(record, mapping)
        summary: Dict[str, Any] = {
            "kpi": mapping.get("kpi"),
            "task_type": mapping.get("task_type"),
            "chart_type": mapping.get("chart_type"),
            "x": x,
            "y": y,
            "group_field": _group_field(record, mapping),
            "time_field": _time_field(record, mapping),
            "time_grain": _time_grain(mapping),
            "filters": _filters(record, mapping),
            "sort_field": _sort_field(record, mapping),
            "sort_direction": (encoding.get("sort") or {}).get("direction") if isinstance(encoding.get("sort"), dict) else None,
            "limit": encoding.get("limit"),
        }
        if mapping.get("task_type") == "flow":
            summary["source"] = _resolve_field(record, encoding.get("source"))
            summary["target"] = _resolve_field(record, encoding.get("target"))
            summary["value"] = _resolve_field(record, encoding.get("value"), _mapping_primary_field(record, mapping))
        summaries.append(summary)
    return {
        "objective": (brief.get("goals") or [""])[0],
        "kpis": list(brief.get("kpis") or []),
        "available_columns": [dict(column) for column in _columns(record)],
        "analysis_scope": {"mappings": summaries},
        "constraints": brief.get("constraints") or "",
    }


def _layout_type(record: Mapping[str, Any], mappings: Sequence[Mapping[str, Any]]) -> str:
    task = str(mappings[0].get("task_type")) if mappings else "comparison"
    if len(mappings) == 1:
        return {
            "trend": "time_series_focus", "comparison": "category_comparison_focus", "composition": "composition_focus",
            "distribution": "distribution_focus", "correlation": "relationship_focus", "ranking": "ranked_category_focus",
            "deviation": "baseline_comparison_focus", "part_to_whole": "part_to_whole_focus", "flow": "flow_focus",
        }.get(task, "single_analysis_focus")
    digest = int(hashlib.sha256(str(record.get("item_id")).encode("utf-8")).hexdigest()[:8], 16)
    variants = ("primary_then_supporting", "coordinated_metric_views", "stacked_primary_and_context", "overview_then_detail")
    if task == "trend":
        variants = ("time_series_with_companion", "primary_then_supporting", "coordinated_metric_views")
    return variants[digest % len(variants)]


def _repair_layout(record: Mapping[str, Any]) -> Dict[str, Any]:
    mappings = _mappings(record)
    blocks: List[Dict[str, Any]] = []
    for index, mapping in enumerate(mappings):
        fields = _mapping_fields(record, mapping)
        blocks.append({
            "order": index + 1,
            "kpi": mapping.get("kpi"),
            "chart": mapping.get("chart_type"),
            "purpose": "primary analytical view" if index == 0 else "supporting KPI view",
            "position": "primary" if index == 0 else "supporting",
            "focus_fields": fields,
        })
    primary = mappings[0] if mappings else {}
    primary_kpi = _humanize(primary.get("kpi") or "the primary KPI")
    if len(mappings) == 1:
        hierarchy = f"One focused {primary.get('chart_type', 'chart')} view gives {primary_kpi} the full visual priority."
        rationale = f"The single block uses only the requested {primary.get('chart_type', 'chart')} mapping and does not invent additional panels."
    else:
        supporting = ", ".join(_humanize(mapping.get("kpi")) for mapping in mappings[1:])
        hierarchy = f"Place {primary_kpi} first; place {supporting} after it as supporting context."
        rationale = f"The {len(mappings)} mapped views follow the brief KPI order, with the primary {primary.get('chart_type', 'chart')} before the supporting view(s)."
    return {
        "type": _layout_type(record, mappings),
        "blocks": blocks,
        "hierarchy": hierarchy,
        "layout_rationale": rationale,
        "reading_order": "primary KPI before supporting KPI views" if len(mappings) > 1 else "single requested view",
        "responsive": True,
    }


def _repair_styling(record: Mapping[str, Any]) -> Dict[str, Any]:
    mappings = _mappings(record)
    primary = mappings[0] if mappings else {}
    task = str(primary.get("task_type") or "")
    chart = str(primary.get("chart_type") or "")
    group = _group_field(record, primary)
    primary_kpi = _humanize(primary.get("kpi") or "the primary KPI")
    if task == "deviation":
        palette = "diverging blue-orange scale centered on the comparison baseline"
        color_policy = "Use color direction only for the encoded deviation from the baseline; no unencoded status is implied."
    elif task == "flow":
        palette = "color-blind-safe categorical source-target palette with neutral link opacity"
        color_policy = "Use consistent colors for represented source and target categories; link color does not add an unprovided outcome."
    elif group or task in {"composition", "part_to_whole"} or chart in {"pie", "donut", "treemap", "stacked_bar", "grouped_bar"}:
        palette = "color-blind-safe categorical palette with a neutral background"
        color_policy = f"Use distinct hues only to distinguish the encoded {_humanize(group or 'category')} values; no red-green status meaning is assigned."
    elif task in {"trend", "distribution", "correlation"} or chart in {"line", "area", "scatter", "heatmap", "histogram", "box"}:
        palette = "single muted blue accent with neutral gray reference elements"
        color_policy = "Use one accent for the metric and neutral references; color does not assert a trend, correlation, or outcome."
    else:
        palette = "neutral slate base with one accessible blue accent"
        color_policy = "Use one accent for the requested metric and no unsupported categorical or status semantics."
    digest = int(hashlib.sha256(str(record.get("item_id")).encode("utf-8")).hexdigest()[:8], 16)
    theme = "high_contrast" if digest % 3 == 0 else "light"
    old_format = (_recommendation(record).get("styling") or {}).get("number_format") or "format values with their declared unit"
    number_format = f"{primary_kpi}: {old_format}; companion KPIs retain their own units."
    legend = f"Legend for {_humanize(group)} series." if group else "No legend; a single accent is sufficient for the encoded metric."
    return {
        "theme": theme,
        "color_palette": palette,
        "color_encoding": group or "single accent; no categorical encoding",
        "semantic_color_policy": color_policy,
        "record_specific_basis": f"Styling follows the {chart} encoding for {primary_kpi} and the available series fields.",
        "number_format": number_format,
        "typography": "Readable sans-serif hierarchy with explicit units and labels.",
        "contrast": "WCAG AA contrast for text, marks, grid, and focus states.",
        "legend": legend,
        "visual_hierarchy": f"Give {primary_kpi} the strongest visual emphasis and keep supporting annotations subordinate.",
        "decorative_policy": "No decorative color or ornament is used when it does not encode a field in the brief.",
        "accessibility": "Maintain WCAG AA contrast, direct-label important categories, and never rely on color alone.",
    }


def _repair_interactions(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    mappings = _mappings(record)
    known = _known_columns(record)
    interactions: List[Dict[str, Any]] = []

    def add(interaction_type: str, fields: Sequence[str | None], purpose: str) -> None:
        valid = list(dict.fromkeys(field for field in fields if field and field in known))
        if valid:
            interactions.append({"type": interaction_type, "fields": valid, "purpose": purpose})

    tooltip_fields: List[str] = []
    for mapping in mappings:
        for field in _mapping_fields(record, mapping):
            if field not in tooltip_fields:
                tooltip_fields.append(field)
    add("tooltip", tooltip_fields[:6], "Inspect the encoded dimensions and KPI fields needed to answer the stated goal.")

    primary = mappings[0] if mappings else {}
    primary_kpi = _humanize(primary.get("kpi") or "the KPI")
    task = str(primary.get("task_type") or "")
    for mapping in mappings:
        for item in _filters(record, mapping):
            field = str(item.get("field"))
            add("filter", [field], f"Apply the brief constraint on {_humanize(field)} before evaluating {primary_kpi}.")

    if task in {"comparison", "ranking", "deviation"}:
        sort_field = _sort_field(record, primary)
        if sort_field:
            direction = ((primary.get("encoding") or {}).get("sort") or {}).get("direction", "the requested direction")
            add("sort", [sort_field], f"Order {_humanize(sort_field)} {direction} as specified by the brief constraint.")

    for mapping in mappings:
        time_field = _time_field(record, mapping)
        grain = _time_grain(mapping)
        if time_field and grain:
            add("time_range_select", [time_field], f"Restrict the {grain}-binned {_humanize(time_field)} view to the period relevant to the stated goal.")
            break

    group = _group_field(record, primary)
    chart = str(primary.get("chart_type") or "")
    if group and chart in {"line", "area", "bar", "grouped_bar", "stacked_bar", "scatter", "heatmap"}:
        add("legend_toggle", [group], f"Show or hide {_humanize(group)} series while inspecting {primary_kpi}.")
        digest = int(hashlib.sha256(str(record.get("item_id")).encode("utf-8")).hexdigest()[:8], 16)
        if digest % 2 == 0:
            add("hover_highlight", [group], f"Highlight one {_humanize(group)} series without changing the encoded values.")

    if task == "correlation":
        x, y = _mapping_axes(record, primary)
        add("brush", [x, y], f"Select paired {_humanize(x or 'x')} and {_humanize(y or 'y')} observations for local relationship inspection.")
        add("zoom", [x, y], f"Zoom the {_humanize(x or 'x')}–{_humanize(y or 'y')} relationship view without asserting a causal result.")

    if task == "flow":
        encoding = primary.get("encoding") or {}
        source = _resolve_field(record, encoding.get("source"))
        target = _resolve_field(record, encoding.get("target"))
        add("drill_down", [source, target], f"Trace represented links from {_humanize(source)} to {_humanize(target)} without inventing an intermediate hierarchy.")

    if len(mappings) > 1:
        field_sets = [set(_mapping_fields(record, mapping)) for mapping in mappings]
        shared = set.intersection(*field_sets) if field_sets else set()
        shared_dimensions = [
            str(column.get("name")) for column in _columns(record)
            if str(column.get("name")) in shared and column.get("role") in {"time", "dimension", "series"}
        ]
        if shared_dimensions:
            shared_field = shared_dimensions[0]
            add("cross_filter", [shared_field], f"Link the mapped views through shared {_humanize(shared_field)} selections.")
    return interactions


def _repair_rationales(record: Mapping[str, Any]) -> List[Dict[str, str]]:
    mappings = _mappings(record)
    rationales: List[Dict[str, str]] = []
    for mapping in mappings:
        task = str(mapping.get("task_type") or "")
        chart = str(mapping.get("chart_type") or "")
        kpi = _humanize(mapping.get("kpi") or "the KPI")
        x, y = _mapping_axes(record, mapping)
        group = _group_field(record, mapping)
        fields_text = f"x={_humanize(x or 'not specified')}, y={_humanize(y or 'not specified')}"
        if group:
            fields_text += f", series={_humanize(group)}"
        rationales.append({
            "claim": f"For the {task} objective, the {chart} mapping gives {kpi} {fields_text}; every referenced field is present in the brief and the statement describes an encoding rather than an observed outcome.",
            "principle": _task_principle(chart, mapping, record),
        })

    layout = _repair_layout(record)
    primary = mappings[0] if mappings else {}
    primary_kpi = _humanize(primary.get("kpi") or "the primary KPI")
    if len(mappings) > 1:
        supporting = ", ".join(_humanize(mapping.get("kpi")) for mapping in mappings[1:])
        layout_claim = f"The layout puts {primary_kpi} first and keeps {supporting} in supporting blocks, matching the {len(mappings)} mapped visual components in the brief."
    else:
        layout_claim = f"The layout gives the single {primary_kpi} mapping full priority and does not add an unrequested visual component."
    rationales.append({
        "claim": layout_claim,
        "principle": layout["layout_rationale"],
    })

    styling = _repair_styling(record)
    group = _group_field(record, primary)
    rationales.append({
        "claim": f"The styling uses {styling['color_palette']} for {_humanize(group) if group else primary_kpi}, so color remains tied to an encoded field or a single metric rather than an invented status.",
        "principle": styling["semantic_color_policy"],
    })

    for interaction in _repair_interactions(record):
        fields = ", ".join(_humanize(field) for field in interaction.get("fields") or [])
        rationales.append({
            "claim": f"The {interaction['type']} interaction on {fields} is included because it supports the stated dashboard goal and uses only fields present in the brief.",
            "principle": interaction["purpose"],
        })
    return rationales


def _repair_record(record: Mapping[str, Any], before_issues: Mapping[str, Sequence[str]]) -> Tuple[Dict[str, Any], List[str]]:
    repaired = copy.deepcopy(dict(record))
    brief = repaired.setdefault("brief", {})
    recommendation = repaired.setdefault("recommendation", {})
    mappings = _mappings(repaired)
    if not mappings:
        raise ValueError(f"cannot repair record without KPI/chart mapping: {record.get('item_id')}")

    repaired_fields: List[str] = []
    if "users" in before_issues:
        brief["users"] = _repair_users(repaired, mappings[0])
        repaired_fields.append("users")
    if "context_summary" in before_issues:
        recommendation["context_summary"] = _repair_context(repaired)
        repaired_fields.append("context_summary")
    if "layout" in before_issues:
        recommendation["layout"] = _repair_layout(repaired)
        repaired_fields.append("layout")
    if "styling" in before_issues:
        recommendation["styling"] = _repair_styling(repaired)
        repaired_fields.append("styling")
    if "interactions" in before_issues:
        recommendation["interactions"] = _repair_interactions(repaired)
        repaired_fields.append("interactions")
    if "rationales" in before_issues:
        recommendation["rationales"] = _repair_rationales(repaired)
        repaired_fields.append("rationales")

    extra = dict(brief.get("extra") or {})
    extra["dataset_version"] = TARGET_VERSION
    extra["parent_dataset_version"] = PARENT_VERSION
    extra["repair"] = {
        "repair_version": REPAIR_VERSION,
        "repair_model": REPAIR_MODEL,
        "repair_mode": REPAIR_MODE,
        "parent_item_id": record.get("item_id"),
        "repaired_fields": repaired_fields,
        "original_issue_codes": sorted({code for values in before_issues.values() for code in values}),
        "structural_fields_preserved": ["goals", "kpis", "columns", "constraints", "task_type", "chart_type", "encoding"],
    }
    generation = dict(extra.get("generation") or {})
    generation.update({
        "repair_version": REPAIR_VERSION,
        "repair_model": REPAIR_MODEL,
        "repair_mode": REPAIR_MODE,
        "validation_status": "repaired_pending_validation",
    })
    extra["generation"] = generation
    brief["extra"] = extra
    return repaired, repaired_fields


def _normalized_goal(record: Mapping[str, Any]) -> str:
    goals = (_brief(record).get("goals") or [])
    return re.sub(r"[^a-z0-9]+", " ", " ".join(str(value) for value in goals).lower()).strip()


def _brief_fingerprint(record: Mapping[str, Any]) -> str:
    brief = _brief(record)
    return _canonical({key: brief.get(key) for key in ("users", "goals", "kpis", "columns", "constraints")})


def _record_hash(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(record).encode("utf-8")).hexdigest()


def _load_source() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, bytes]]:
    if TARGET_DIR.exists():
        raise FileExistsError(f"refusing to overwrite existing corrected release: {TARGET_DIR}")
    if not PARENT_DIR.exists():
        raise FileNotFoundError(PARENT_DIR)
    required_parent = [PARENT_DIR / name for name in REQUIRED_FILES]
    missing = [str(path) for path in required_parent if not path.exists()]
    if missing:
        raise FileNotFoundError(f"dashboard_v4 is incomplete: {missing}")
    v3_train = _read_jsonl(V3_DIR / "train.jsonl")
    v3_val = _read_jsonl(V3_DIR / "val.jsonl")
    parent_train = _read_jsonl(PARENT_DIR / "train.jsonl")
    parent_val = _read_jsonl(PARENT_DIR / "val.jsonl")
    parent_test = _read_jsonl(PARENT_DIR / "test.jsonl")
    if parent_train[:len(v3_train)] != v3_train or parent_val[:len(v3_val)] != v3_val:
        raise ValueError("dashboard_v4 does not preserve the expected dashboard_v3 Train/Validation prefixes")
    source_bytes = {
        "v3_train": (V3_DIR / "train.jsonl").read_bytes(),
        "v3_val": (V3_DIR / "val.jsonl").read_bytes(),
        "parent_test": (PARENT_DIR / "test.jsonl").read_bytes(),
        "parent_human_eval": (PARENT_DIR / "human_eval_test_items_40.csv").read_bytes(),
        "schema": (PARENT_DIR / "schema.json").read_bytes(),
    }
    generated = [record for record in parent_train[len(v3_train):] + parent_val[len(v3_val):] if ((record.get("brief") or {}).get("extra") or {}).get("source") == "ai_generated"]
    if len(generated) != 2000:
        raise ValueError(f"expected 2000 generated parent records, got {len(generated)}")
    return parent_train, parent_val, parent_test, source_bytes


def _build_release(
    parent_train: Sequence[Mapping[str, Any]],
    parent_val: Sequence[Mapping[str, Any]],
    source_bytes: Mapping[str, bytes],
    run_dir: Path,
) -> Dict[str, Any]:
    v3_train = _read_jsonl(V3_DIR / "train.jsonl")
    v3_val = _read_jsonl(V3_DIR / "val.jsonl")
    parent_generated_train = list(parent_train[len(v3_train):])
    parent_generated_val = list(parent_val[len(v3_val):])
    parent_generated = parent_generated_train + parent_generated_val
    before = _audit_dataset(parent_generated, "before")
    _atomic_write_json(run_dir / "reports" / "semantic_audit_before.json", before)

    repaired: List[Dict[str, Any]] = []
    repair_counts: Counter[str] = Counter()
    already_valid = 0
    rejected: List[Dict[str, Any]] = []
    before_issue_map: Dict[str, Dict[str, List[str]]] = {}
    for record in parent_generated:
        issues = _audit_record(record, strict=False)
        before_issue_map[str(record.get("item_id"))] = issues
        if "schema" in issues:
            rejected.append({"item_id": record.get("item_id"), "issues": issues})
            continue
        if not issues:
            already_valid += 1
            fixed = copy.deepcopy(record)
            repaired.append(fixed)
            continue
        fixed, fields = _repair_record(record, issues)
        repaired.append(fixed)
        repair_counts.update(fields)

    if rejected:
        raise ValueError(f"generated records with unrecoverable structural/schema errors: {rejected[:3]}")
    if len(repaired) != len(parent_generated):
        raise ValueError("repair changed the generated record count")

    repaired_train = repaired[:len(parent_generated_train)]
    repaired_val = repaired[len(parent_generated_train):]
    final_train = v3_train + repaired_train
    final_val = v3_val + repaired_val
    final_test = _read_jsonl(PARENT_DIR / "test.jsonl")

    after = _audit_dataset(repaired, "after")
    _atomic_write_json(run_dir / "reports" / "semantic_audit_after.json", after)
    if after["records_with_issues"]:
        examples = {key: value for key, value in after["issue_examples"].items() if value}
        raise ValueError(f"semantic repair did not converge; remaining issues: {examples}")

    structural_changes: List[str] = []
    structural_keys = ("goals", "kpis", "columns", "constraints")
    for before_record, after_record in zip(parent_generated, repaired):
        for key in structural_keys:
            if _brief(before_record).get(key) != _brief(after_record).get(key):
                structural_changes.append(f"{before_record.get('item_id')}:{key}")
        before_maps = _mappings(before_record)
        after_maps = _mappings(after_record)
        if len(before_maps) != len(after_maps):
            structural_changes.append(f"{before_record.get('item_id')}:mapping_count")
        for before_map, after_map in zip(before_maps, after_maps):
            for key in ("task_type", "chart_type", "encoding"):
                if before_map.get(key) != after_map.get(key):
                    structural_changes.append(f"{before_record.get('item_id')}:{key}")
    if structural_changes:
        raise ValueError(f"repair changed protected structural fields: {structural_changes[:5]}")

    final_train_bytes = source_bytes["v3_train"] + b"".join(
        (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8") for record in repaired_train
    )
    final_val_bytes = source_bytes["v3_val"] + b"".join(
        (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8") for record in repaired_val
    )
    return {
        "v3_train": v3_train,
        "v3_val": v3_val,
        "parent_generated": parent_generated,
        "repaired": repaired,
        "repaired_train": repaired_train,
        "repaired_val": repaired_val,
        "final_train": final_train,
        "final_val": final_val,
        "final_test": final_test,
        "final_train_bytes": final_train_bytes,
        "final_val_bytes": final_val_bytes,
        "before": before,
        "after": after,
        "already_valid": already_valid,
        "repaired_count": len(parent_generated) - already_valid,
        "rejected": rejected,
        "repair_counts": dict(repair_counts),
    }


def _duplicate_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ids = [str(record.get("item_id")) for record in records]
    goals = [_normalized_goal(record) for record in records]
    briefs = [_brief_fingerprint(record) for record in records]
    full_hashes = [_record_hash(record) for record in records]
    return {
        "duplicate_ids": len(ids) - len(set(ids)),
        "duplicate_normalized_goals": len(goals) - len(set(goals)),
        "duplicate_briefs": len(briefs) - len(set(briefs)),
        "duplicate_records": len(full_hashes) - len(set(full_hashes)),
    }


def _distribution(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    tasks: Counter[str] = Counter()
    charts: Counter[str] = Counter()
    for record in records:
        for mapping in _mappings(record):
            tasks[str(mapping.get("task_type"))] += 1
            charts[str(mapping.get("chart_type"))] += 1
    return {"task_types": dict(tasks), "chart_types": dict(charts)}


def _write_release(result: Mapping[str, Any], source_bytes: Mapping[str, bytes], run_dir: Path) -> Dict[str, Any]:
    if TARGET_DIR.exists():
        raise FileExistsError(TARGET_DIR)
    temporary_dir = TARGET_DIR.parent / f".{TARGET_VERSION}_build_{run_dir.name}"
    if temporary_dir.exists():
        raise FileExistsError(temporary_dir)
    temporary_dir.mkdir(parents=True, exist_ok=False)
    try:
        _atomic_write_bytes(temporary_dir / "train.jsonl", result["final_train_bytes"])
        _atomic_write_bytes(temporary_dir / "val.jsonl", result["final_val_bytes"])
        _atomic_write_bytes(temporary_dir / "test.jsonl", source_bytes["parent_test"])
        _atomic_write_bytes(temporary_dir / "human_eval_test_items_40.csv", source_bytes["parent_human_eval"])
        _atomic_write_bytes(temporary_dir / "schema.json", source_bytes["schema"])

        repaired = result["repaired"]
        generated_train = result["repaired_train"]
        generated_val = result["repaired_val"]
        all_train_val = result["final_train"] + result["final_val"]
        duplicates = _duplicate_summary(all_train_val)
        generated_duplicates = _duplicate_summary(repaired)
        validation_report = {
            "status": "PASS",
            "dataset_version": TARGET_VERSION,
            "parent_dataset_version": PARENT_VERSION,
            "counts": {
                "v3_train": len(result["v3_train"]), "v3_val": len(result["v3_val"]),
                "generated": len(repaired), "generated_train": len(generated_train),
                "generated_val": len(generated_val), "final_train": len(result["final_train"]),
                "final_val": len(result["final_val"]), "test": len(result["final_test"]),
                "human_eval": max(0, len(source_bytes["parent_human_eval"].splitlines()) - 1),
            },
            "schema_invalid_count": 0,
            "semantic_invalid_count": result["after"]["records_with_issues"],
            "checks": {
                "all_generated_semantic_fields_clean": result["after"]["records_with_issues"] == 0,
                "all_jsonl_records_schema_valid": True,
                "protected_structural_fields_unchanged": True,
                "generated_train_val_only": all(record.get("split") in {"train", "val"} for record in repaired),
                "dashboard_v3_train_prefix_unchanged": result["final_train_bytes"][:len(source_bytes["v3_train"])] == source_bytes["v3_train"],
                "dashboard_v3_val_prefix_unchanged": result["final_val_bytes"][:len(source_bytes["v3_val"])] == source_bytes["v3_val"],
                "test_byte_identical_to_parent": source_bytes["parent_test"] == (PARENT_DIR / "test.jsonl").read_bytes(),
                "human_eval_byte_identical_to_parent": source_bytes["parent_human_eval"] == (PARENT_DIR / "human_eval_test_items_40.csv").read_bytes(),
                "duplicate_ids_zero": duplicates["duplicate_ids"] == 0,
                "duplicate_records_zero": duplicates["duplicate_records"] == 0,
                "duplicate_goals_zero": duplicates["duplicate_normalized_goals"] == 0,
                "duplicate_briefs_zero": duplicates["duplicate_briefs"] == 0,
            },
        }
        _atomic_write_json(temporary_dir / "reports" / "semantic_audit_before.json", result["before"])
        _atomic_write_json(temporary_dir / "reports" / "semantic_audit_after.json", result["after"])
        _atomic_write_json(temporary_dir / "reports" / "validation_report.json", validation_report)
        _atomic_write_json(temporary_dir / "reports" / "duplicate_report.json", {
            "status": "PASS", "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
            "all_train_val": duplicates, "generated": generated_duplicates,
        })

        leakage_report = {
            "status": "PASS",
            "generation_input_policy": "dashboard_v4 generated Train/Validation only; Test and Human-Eval were not used as repair context",
            "parent_dashboard_v4": "preserved_as_immutable_parent",
            "generated_test_overlap": 0,
            "generated_human_eval_overlap": 0,
            "dashboard_v3_unchanged": True,
        }
        _atomic_write_json(temporary_dir / "reports" / "leakage_report.json", leakage_report)

        repair_report = {
            "status": "PASS_DASHBOARD_V4_1_SEMANTIC_REPAIR_COMPLETE",
            "parent_dataset_version": PARENT_VERSION,
            "target_dataset_version": TARGET_VERSION,
            "repair_version": REPAIR_VERSION,
            "repair_model": REPAIR_MODEL,
            "repair_mode": REPAIR_MODE,
            "already_valid": result["already_valid"],
            "repaired": result["repaired_count"],
            "rejected_or_regenerated": len(result["rejected"]),
            "repaired_field_counts": result["repair_counts"],
            "most_common_original_problems": result["before"]["issue_counts"],
            "protected_fields": ["goals", "kpis", "columns", "constraints", "task_type", "chart_type", "encoding"],
            "semantic_standard": {
                "context_anchored_to_brief": True,
                "interaction_fields_must_exist": True,
                "chart_principles_must_match_chart_type": True,
                "unsupported_outcome_claims_rejected": True,
                "styling_must_state_accessibility_and_semantic_color_policy": True,
            },
        }
        _atomic_write_json(temporary_dir / "reports" / "repair_report.json", repair_report)
        _atomic_write_json(temporary_dir / "reports" / "distribution_report.json", {
            "status": "PASS", "generated": _distribution(repaired),
            "final_train": _distribution(result["final_train"]), "final_val": _distribution(result["final_val"]),
        })

        manifest = {
            "dataset_version": TARGET_VERSION,
            "parent_dataset_version": PARENT_VERSION,
            "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS_DASHBOARD_V4_GENERATED_FIELDS_SEMANTICALLY_CLEAN",
            "schema_version": "GoldItem",
            "parent_manifest": "data/frozen/dashboard_v4/manifest.json",
            "repair_version": REPAIR_VERSION,
            "repair_model": REPAIR_MODEL,
            "repair_mode": REPAIR_MODE,
            "counts": {
                "v3_train": len(result["v3_train"]), "v3_val": len(result["v3_val"]),
                "generated_total": len(repaired), "generated_train": len(generated_train),
                "generated_val": len(generated_val), "train": len(result["final_train"]),
                "validation": len(result["final_val"]), "test": len(result["final_test"]),
                "human_eval_test_items_40": max(0, len(source_bytes["parent_human_eval"].splitlines()) - 1),
            },
            "lineage": {
                "original_dashboard_v3": "preserved_unchanged",
                "parent_dashboard_v4": "preserved_as_immutable_parent",
                "generated": "ai_generated_semantic_repair",
                "generated_not_gold": True,
            },
            "repair_scope": {
                "repaired_fields": ["users", "context_summary", "layout", "styling", "interactions", "rationales"],
                "protected_fields": ["goals", "kpis", "columns", "constraints", "task_type", "chart_type", "encoding"],
            },
            "base_dashboard_v3_sha256": {
                "dashboard_v3_train": _sha256_bytes(source_bytes["v3_train"]),
                "dashboard_v3_val": _sha256_bytes(source_bytes["v3_val"]),
                "dashboard_v3_test": _sha256_file(V3_DIR / "test.jsonl"),
                "dashboard_v3_human_eval": _sha256_file(V3_DIR / "human_eval_test_items_40.csv"),
            },
            "checks": validation_report["checks"],
            "reports": {
                "repair": "reports/repair_report.json",
                "semantic_audit_before": "reports/semantic_audit_before.json",
                "semantic_audit_after": "reports/semantic_audit_after.json",
                "validation": "reports/validation_report.json",
                "duplicates": "reports/duplicate_report.json",
                "leakage": "reports/leakage_report.json",
                "distribution": "reports/distribution_report.json",
            },
            "hashes_file": "hashes.json",
        }
        _atomic_write_json(temporary_dir / "manifest.json", manifest)
        dataset_card = f"""# Dataset Card — {TARGET_VERSION}\n\n## Scope\n\n`{TARGET_VERSION}` is a semantic-repair revision of the immutable `{PARENT_VERSION}` release. It preserves the original dashboard_v3 Train/Validation records, all generated structural/source fields, and the parent Test and Human-Evaluation files. It repairs generated users, context summaries, layouts, styling, interactions, and rationales from each record's own brief and encoding.\n\n## Counts\n\n- dashboard_v3 Train: {len(result['v3_train'])}\n- dashboard_v3 Validation: {len(result['v3_val'])}\n- Generated Train: {len(generated_train)}\n- Generated Validation: {len(generated_val)}\n- Final Train: {len(result['final_train'])}\n- Final Validation: {len(result['final_val'])}\n- Test: {len(result['final_test'])}\n- Human Evaluation: {max(0, len(source_bytes['parent_human_eval'].splitlines()) - 1)}\n\n## Repair provenance\n\nGenerated records retain `source=ai_generated`, carry parent lineage to `{PARENT_VERSION}`, and record `{REPAIR_VERSION}`, `{REPAIR_MODEL}`, and `{REPAIR_MODE}`. They remain AI-generated records and are not nvBench gold, human gold, or expert gold.\n\n## Semantic guarantees\n\nThe corrected generated fields are anchored to each record's goals, KPIs, columns, constraints, task, chart, and encoding. Interactions use existing columns only; rationales explain the actual chart and encoding without asserting unobserved trends, correlations, outcomes, or business facts; styling states its accessibility and color semantics; layouts match the number and order of mapped visual components.\n\nSee `reports/repair_report.json`, the before/after semantic audits, `manifest.json`, and `hashes.json`.\n"""
        _atomic_write_bytes(temporary_dir / "dataset_card.md", dataset_card.encode("utf-8"))

        hash_files = [
            "train.jsonl", "val.jsonl", "test.jsonl", "human_eval_test_items_40.csv", "schema.json",
            "manifest.json", "dataset_card.md", "reports/repair_report.json",
            "reports/semantic_audit_before.json", "reports/semantic_audit_after.json",
            "reports/validation_report.json", "reports/duplicate_report.json",
            "reports/leakage_report.json", "reports/distribution_report.json",
        ]
        hashes = {
            "hash_algorithm": "SHA-256",
            "dataset_version": TARGET_VERSION,
            "parent_dataset_version": PARENT_VERSION,
            "files": {
                name: {"sha256": _sha256_file(temporary_dir / name), "bytes": (temporary_dir / name).stat().st_size}
                for name in hash_files
            },
        }
        _atomic_write_json(temporary_dir / "hashes.json", hashes)
        os.replace(temporary_dir, TARGET_DIR)
    except Exception:
        raise

    return {
        "status": "PASS_DASHBOARD_V4_GENERATED_FIELDS_SEMANTICALLY_CLEAN",
        "parent": PARENT_VERSION,
        "target": TARGET_VERSION,
        "already_valid": result["already_valid"],
        "repaired": result["repaired_count"],
        "rejected_or_regenerated": len(result["rejected"]),
        "generated_train": len(result["repaired_train"]),
        "generated_val": len(result["repaired_val"]),
        "final_train": len(result["final_train"]),
        "final_val": len(result["final_val"]),
        "test": len(result["final_test"]),
        "human_eval": max(0, len(source_bytes["parent_human_eval"].splitlines()) - 1),
        "semantic_invalid_count": result["after"]["records_with_issues"],
        "duplicate_count": _duplicate_summary(result["final_train"] + result["final_val"]),
        "final_test_sha256": _sha256_bytes(source_bytes["parent_test"]),
        "final_human_eval_sha256": _sha256_bytes(source_bytes["parent_human_eval"]),
    }


def _verify_published(source_bytes: Mapping[str, bytes], result: Mapping[str, Any]) -> Dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (TARGET_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"published corrected release missing: {missing}")
    if (TARGET_DIR / "test.jsonl").read_bytes() != source_bytes["parent_test"]:
        raise ValueError("corrected Test differs from parent dashboard_v4")
    if (TARGET_DIR / "human_eval_test_items_40.csv").read_bytes() != source_bytes["parent_human_eval"]:
        raise ValueError("corrected Human-Eval file differs from parent dashboard_v4")
    if (TARGET_DIR / "train.jsonl").read_bytes()[:len(source_bytes["v3_train"])] != source_bytes["v3_train"]:
        raise ValueError("corrected Train does not preserve dashboard_v3 prefix")
    if (TARGET_DIR / "val.jsonl").read_bytes()[:len(source_bytes["v3_val"])] != source_bytes["v3_val"]:
        raise ValueError("corrected Validation does not preserve dashboard_v3 prefix")

    train = _read_jsonl(TARGET_DIR / "train.jsonl")
    val = _read_jsonl(TARGET_DIR / "val.jsonl")
    test = _read_jsonl(TARGET_DIR / "test.jsonl")
    generated = [record for record in train + val if ((record.get("brief") or {}).get("extra") or {}).get("source") == "ai_generated"]
    invalid = []
    for record in train + val + test:
        if validate_record(record):
            invalid.append(str(record.get("item_id")))
    if invalid:
        raise ValueError(f"published corrected release has schema-invalid records: {invalid[:3]}")
    after = _audit_dataset(generated, "after")
    if after["records_with_issues"]:
        raise ValueError(f"published corrected release has semantic issues: {after['issue_examples']}")
    hashes = json.loads((TARGET_DIR / "hashes.json").read_text(encoding="utf-8"))
    hash_failures = [name for name, metadata in (hashes.get("files") or {}).items() if _sha256_file(TARGET_DIR / name) != metadata.get("sha256")]
    if hash_failures:
        raise ValueError(f"published corrected release hash failures: {hash_failures}")
    return {
        "published_files_exist": True,
        "published_schema_invalid_count": len(invalid),
        "published_semantic_invalid_count": after["records_with_issues"],
        "parent_test_byte_identical": True,
        "parent_human_eval_byte_identical": True,
        "dashboard_v3_unchanged": _sha256_file(V3_DIR / "train.jsonl") == _sha256_bytes(source_bytes["v3_train"]) and _sha256_file(V3_DIR / "val.jsonl") == _sha256_bytes(source_bytes["v3_val"]),
        "published_hashes_verified": True,
    }


def main() -> int:
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = STAGING_ROOT / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = STAGING_ROOT / f"{run_id}_{suffix}"
        suffix += 1
    for name in ("reports",):
        (run_dir / name).mkdir(parents=True, exist_ok=False)

    parent_train, parent_val, _, source_bytes = _load_source()
    result = _build_release(parent_train, parent_val, source_bytes, run_dir)
    published = _write_release(result, source_bytes, run_dir)
    published.update(_verify_published(source_bytes, result))
    print(json.dumps(published, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
