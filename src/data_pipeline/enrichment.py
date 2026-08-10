"""Phase-3 LLM enrichment: presentation fields only, source facts immutable.

The LLM may write six presentation fields and nothing else::

    users  context_summary  layout  styling  interactions  rationales

Everything analytical or source-backed (item_id, provenance, KPI, chart type,
task type, alternatives, encoding, filters/sort/limit/grouping/time grain, raw
columns and dtypes, quality score/tier, split) is immutable: it is fingerprinted
before the call and re-checked after the merge, and any change rejects the
candidate with ``immutable_source_field_changed``.

The model never merges anything. It returns a strict JSON object validated
against :class:`EnrichmentPayload` (``extra="forbid"``); trusted code in
:func:`merge_enrichment` writes the accepted values into a deep copy of the
record. Rejected candidates keep their reason codes and never enter any dataset.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.core.schemas import ChartType, Rationale

# Fields the LLM is allowed to produce. Anything else in the reply is a rejection.
ENRICHABLE_FIELDS = ("users", "context_summary", "layout", "styling", "interactions", "rationales")

# Lineage keys that describe enrichable fields; every other lineage key is immutable.
# The source lineage dict carries none of these except layout/styling/interactions/
# rationales, so excluding them from the immutable hash removes no source evidence.
ENRICHED_LINEAGE_KEYS = ENRICHABLE_FIELDS

# These six fields are LLM-generated design annotations -- never source, human or
# expert gold. The marker is the vocabulary used by the full-run lineage report.
LINEAGE_ENRICHED_VALUE = "llm_generated"

ENRICHMENT_SPEC_VERSION = "phase3-enrichment-v1"

# Interaction verbs an analytical dashboard can offer without inventing data.
ALLOWED_INTERACTION_TYPES = (
    "tooltip", "filter", "cross_filter", "drilldown", "legend_toggle",
    "sort", "zoom", "brush", "hover_highlight", "time_range_select",
)

# Task-agreement vocabulary: a rationale must speak about the record's task.
TASK_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "comparison": ("comparison", "compare", "comparing", "across categories", "versus"),
    "part_to_whole": ("part-to-whole", "part to whole", "share", "proportion", "percentage", "composition"),
    "composition": ("composition", "composed", "share", "proportion", "stacked", "breakdown"),
    "trend": ("trend", "over time", "temporal", "time series", "development"),
    "correlation": ("correlation", "relationship", "association", "scatter of", "covariation"),
    "distribution": ("distribution", "spread", "frequency"),
    "ranking": ("ranking", "rank", "ordered", "top"),
    "deviation": ("deviation", "variance", "difference from"),
    "flow": ("flow", "transition"),
}

CHART_VOCABULARY = tuple(c.value for c in ChartType)

# Chart names that are also ordinary English words ("the products table", "the
# area of interest"). They only count as a foreign chart mention next to a chart
# noun; the unambiguous ones (pie, scatter, sankey, ...) count on their own.
AMBIGUOUS_CHART_WORDS = frozenset({"table", "map", "area", "box", "line", "gauge", "bar"})
_CHART_CONTEXT = r"(?:chart|graph|plot|visuali[sz]ation|diagram|view)"

# A prose mention of another chart is only a violation when the text actually
# proposes it. Naming or comparing against the selected chart is allowed.
_REPLACEMENT_CUES = (
    r"instead", r"replac", r"switch", r"recommend", r"suggest", r"prefer",
    r"better", r"rather than", r"should (?:be )?(?:use|show|render)", r"use a\b",
    r"use an\b", r"use the\b", r"convert", r"change to", r"upgrade to",
    r"\badd\b", r"\binclude\b", r"\bsuppl(?:ement|y)\b",
)

# JSON keys whose numbers describe the interface, not the business data.
DESIGN_NUMERIC_KEYS = frozenset({
    "width", "height", "columns", "column", "rows", "row", "span", "size",
    "font_size", "spacing", "padding", "margin", "gap", "radius", "order",
    "position", "contrast_ratio", "sections", "n_sections", "grid", "weight",
    "opacity", "level", "index",
})

# Citation years and hex colours are not business facts.
_CITATION_YEAR = re.compile(r"[A-Za-z][A-Za-z.&'\s]{2,40},?\s*(?:19|20)\d{2}")
_HEX_COLOR = re.compile(r"#[0-9a-fA-F]{3,8}\b")

# Neutral ways to point at the given chart without naming a type. The prompt asks
# for exactly this wording, so it must be a valid chart reference everywhere.
_SELF_REFERENCE = re.compile(
    r"^(?:the\s+)?(?:selected|given|supplied|source|chosen|current)\s*"
    r"(?:chart|visuali[sz]ation|view)?$|^chart$|^main\s+chart$", re.IGNORECASE)

# A replacement cue inside a negated clause is not a proposal:
# "Do not replace the selected chart with a pie chart."
_NEGATION = re.compile(r"\b(?:not|never|avoid|without|no)\b", re.IGNORECASE)

REASON_CODES = (
    "response_not_json",
    "schema_invalid",
    "extra_fields_returned",
    "empty_enrichment_field",
    "immutable_source_field_changed",
    "interaction_field_not_in_source",
    "unsupported_interaction_type",
    "context_summary_conflicts_source",
    "invented_kpi",
    "invented_chart_type",
    "invented_numeric_value",
    "rationale_disagrees_with_task_chart_or_encoding",
)


# --------------------------------------------------------------- schema


class EnrichmentPayload(BaseModel):
    """Strict contract for the model reply: only the six enrichable fields."""

    model_config = ConfigDict(extra="forbid")

    users: str
    context_summary: Dict[str, Any]
    layout: Dict[str, Any]
    styling: Dict[str, Any]
    interactions: List[Dict[str, Any]] = Field(default_factory=list)
    rationales: List[Rationale] = Field(default_factory=list)


def normalize_structure(obj: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Fix harmless container-shape variation only; never touch content.

    Observed variation: ``layout`` returned as a list of blocks instead of an
    object. Nothing is invented, dropped or rewritten -- the same values are moved
    into the declared container. Content-level problems are never repaired here.
    """
    normalized = dict(obj)
    notes: List[str] = []
    layout = normalized.get("layout")
    if isinstance(layout, list):
        normalized["layout"] = {"blocks": layout}
        notes.append("normalized layout: list -> {'blocks': [...]}")
    for field_name in ("interactions", "rationales"):
        value = normalized.get(field_name)
        if isinstance(value, Mapping):
            normalized[field_name] = [dict(value)]
            notes.append(f"normalized {field_name}: object -> [object]")
    users = normalized.get("users")
    if isinstance(users, list) and all(isinstance(u, str) for u in users):
        normalized["users"] = "; ".join(users)
        notes.append("normalized users: list -> joined string")
    return normalized, notes


def parse_payload(obj: Mapping[str, Any]) -> Tuple[Optional[EnrichmentPayload], List[str], List[str]]:
    """Validate a decoded JSON object. Returns (payload, reason_codes, details).

    Extra top-level fields are rejected outright; only container shapes are
    normalized first (see :func:`normalize_structure`).
    """
    unknown = [k for k in obj if k not in ENRICHABLE_FIELDS]
    if unknown:
        return None, ["extra_fields_returned"], [f"unexpected fields: {sorted(unknown)}"]
    obj, notes = normalize_structure(obj)
    try:
        payload = EnrichmentPayload(**dict(obj))
    except ValidationError as exc:
        details = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:6]]
        return None, ["schema_invalid"], details + notes
    return payload, [], notes


# ---------------------------------------------------- immutable fingerprint


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def immutable_projection(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Every analytical / source-backed part of a record, enrichable parts removed."""
    brief = dict(record.get("brief") or {})
    extra = dict(brief.get("extra") or {})
    lineage = {k: v for k, v in (extra.get("lineage") or {}).items() if k not in ENRICHED_LINEAGE_KEYS}
    recommendation = dict(record.get("recommendation") or {})

    projection: Dict[str, Any] = {
        "item_id": record.get("item_id"),
        "split": record.get("split"),
        "brief": {
            "item_id": brief.get("item_id"),
            "goals": brief.get("goals"),
            "kpis": brief.get("kpis"),
            "columns": brief.get("columns"),
            "constraints": brief.get("constraints"),
        },
        "extra": {
            "source": extra.get("source"),
            "usage_tier": extra.get("usage_tier"),
            "provenance": extra.get("provenance"),
            "task_inference": extra.get("task_inference"),
            "lineage_source_backed": lineage,
        },
        "kpi_chart_mapping": recommendation.get("kpi_chart_mapping"),
    }
    # Quality fields live at record level in some builds; include whatever exists.
    for key in ("quality_score", "quality_tier", "quality", "source_record_id", "source_group_id"):
        if key in record:
            projection[key] = record[key]
    return projection


def immutable_fingerprint(record: Mapping[str, Any]) -> str:
    """SHA-256 over the immutable projection (stable key order)."""
    return hashlib.sha256(_canonical(immutable_projection(record)).encode("utf-8")).hexdigest()


def immutable_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> List[str]:
    """Names of immutable top-level projection keys that differ."""
    a, b = immutable_projection(before), immutable_projection(after)
    return sorted(k for k in set(a) | set(b) if _canonical(a.get(k)) != _canonical(b.get(k)))


# ------------------------------------------------------- source universe


def _mapping_of(record: Mapping[str, Any]) -> Dict[str, Any]:
    mappings = (record.get("recommendation") or {}).get("kpi_chart_mapping") or [{}]
    return dict(mappings[0]) if mappings else {}


def source_columns(record: Mapping[str, Any]) -> Set[str]:
    """Field names an interaction or rationale may legitimately reference."""
    names: Set[str] = set()
    brief = record.get("brief") or {}
    for col in brief.get("columns") or []:
        if isinstance(col, Mapping) and col.get("name"):
            names.add(str(col["name"]))
    for kpi in brief.get("kpis") or []:
        names.add(str(kpi))

    mapping = _mapping_of(record)
    names.add(str(mapping.get("kpi", "")))
    encoding = mapping.get("encoding") or {}
    for key in ("x", "y", "source_x", "source_y", "group_field", "series_field"):
        value = encoding.get(key)
        if isinstance(value, str) and value:
            names.add(value)
    for value in encoding.get("classify") or []:
        if isinstance(value, str) and value:
            names.add(value)
    sort = encoding.get("sort") or {}
    for key in ("field", "expression"):
        if isinstance(sort.get(key), str) and sort[key]:
            names.add(sort[key])
    for flt in encoding.get("filters") or []:
        if isinstance(flt, Mapping) and isinstance(flt.get("field"), str):
            names.add(flt["field"])
    grouping = encoding.get("visual_grouping") or {}
    for value in grouping.get("fields") or []:
        if isinstance(value, str) and value:
            names.add(value)

    # Bare column names inside aggregate expressions, e.g. COUNT(product_name).
    for name in list(names):
        for inner in re.findall(r"\(([^()]*)\)", name):
            for token in re.split(r"[,\s]+", inner):
                token = token.strip()
                if token and token != "*":
                    names.add(token)
    return {n for n in names if n}


def _normalized(name: str) -> str:
    return re.sub(r"[\s_]+", "", str(name).strip().lower())


def source_text(record: Mapping[str, Any]) -> str:
    """Full source record as text -- used to check nothing numeric was invented."""
    return _canonical(record)


# ------------------------------------------------------------- validation


def _numbers(text: str) -> Set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def _payload_text(payload: EnrichmentPayload) -> str:
    return _canonical(payload.model_dump())


def _chart_pattern(chart_word: str) -> str:
    return re.escape(chart_word).replace(r"\_", r"[ _-]")


def mask_source_chart(text: str, chart_type: str) -> str:
    """Blank out mentions of the selected chart so they cannot match another name.

    Without this, a ``stacked_bar`` record naming "stacked bar chart" would look
    like a mention of the foreign chart type ``bar``.
    """
    if not chart_type:
        return text
    return re.sub(_chart_pattern(chart_type), " ", text, flags=re.IGNORECASE)


def mentions_chart_type(text: str, chart_word: str) -> bool:
    """True when ``text`` uses ``chart_word`` as a chart name, not as a plain noun."""
    word = _chart_pattern(chart_word)
    if chart_word in AMBIGUOUS_CHART_WORDS:
        # "area chart" counts; "plot area", "the products table", "line of business" do not.
        pattern = rf"(?<![a-z]){word}\s+{_CHART_CONTEXT}|{_CHART_CONTEXT}\s+type\W{{0,3}}{word}(?![a-z])"
    else:
        pattern = rf"(?<![a-z_]){word}(?![a-z_])"
    return re.search(pattern, text, re.IGNORECASE) is not None


def is_source_chart_reference(value: Any, chart_type: str) -> bool:
    """True when ``value`` points at the given chart rather than another type.

    Accepts the chart type itself and its normalized aliases (``stacked_bar``,
    ``stacked bar``, ``stacked bar chart``) plus the neutral self-references the
    prompt asks for (``the selected chart``).
    """
    if not isinstance(value, str):
        return False
    text = value.strip()
    stripped = re.sub(rf"\s*{_CHART_CONTEXT}\s*$", "", text, flags=re.IGNORECASE).strip()
    if chart_type and _normalized(stripped) == _normalized(chart_type):
        return True
    return _SELF_REFERENCE.match(text) is not None or _SELF_REFERENCE.match(stripped) is not None


def named_chart_types(value: Any) -> List[str]:
    """Chart types a short field value actually names (normalized comparison)."""
    if not isinstance(value, str):
        return []
    stripped = re.sub(rf"\s*{_CHART_CONTEXT}\s*$", "", value.strip(), flags=re.IGNORECASE)
    return [c for c in CHART_VOCABULARY if _normalized(c) == _normalized(stripped)]


def proposes_chart_replacement(text: str, chart_word: str) -> bool:
    """True when a foreign chart is not merely named but put forward as the chart.

    Mentioning or arguing against another chart type stays allowed; only a nearby
    recommendation cue makes it a violation ("use a pie chart instead", "add a
    scatter plot"), and a negated clause ("do not replace ... with a pie chart")
    is not a proposal.
    """
    word = _chart_pattern(chart_word)
    pattern = (rf"(?<![a-z]){word}\s+{_CHART_CONTEXT}" if chart_word in AMBIGUOUS_CHART_WORDS
               else rf"(?<![a-z_]){word}(?![a-z_])")
    for match in re.finditer(pattern, text, re.IGNORECASE):
        window = text[max(0, match.start() - 70):match.end() + 70]
        for cue in _REPLACEMENT_CUES:
            cue_match = re.search(cue, window, re.IGNORECASE)
            if cue_match is None:
                continue
            before = window[max(0, cue_match.start() - 30):cue_match.start()]
            if _NEGATION.search(before):
                continue
            return True
    return False


# Aggregate functions are read aloud in prose ("avg(age)" -> "average age").
AGGREGATE_ALIASES = {
    "avg": ("avg", "average", "mean"),
    "count": ("count", "number", "how many", "frequency"),
    "sum": ("sum", "total"),
    "min": ("min", "minimum", "lowest", "smallest"),
    "max": ("max", "maximum", "highest", "largest"),
}

# Tokens that carry no identifying meaning inside a column name.
_FIELD_STOPWORDS = frozenset({"the", "and", "for", "per", "with", "code", "id", "of"})

# Share of a field's meaningful tokens a rationale must cover to count as
# referencing that field.
FIELD_TOKEN_COVERAGE = 0.6


def _field_tokens(field: str) -> List[str]:
    """Split ``AVG(Amount_Payment)`` into meaningful lowercase tokens."""
    text = re.sub(r"[^A-Za-z0-9_]+", " ", str(field))
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    tokens = [t.lower() for t in re.split(r"[\s_]+", text) if t]
    return [t for t in tokens if len(t) >= 3 and t not in _FIELD_STOPWORDS]


def _token_present(token: str, text: str) -> bool:
    """Match a token tolerantly: plural forms and prose aliases of aggregates."""
    for alias in AGGREGATE_ALIASES.get(token, (token,)):
        if alias in text:
            return True
    if len(token) > 4 and token[: max(4, len(token) - 3)] in text:  # nationality/nationalities
        return True
    return False


def references_encoded_field(text: str, field: str) -> bool:
    """True when ``text`` refers to ``field`` literally or in plain prose.

    The literal column name is accepted first. Otherwise the field's meaningful
    tokens must be covered: ``avg(age)`` is referenced by "average age",
    ``DEPT_CODE`` by "departments". A rationale that names none of the field's
    tokens still fails -- this recognises paraphrase, it does not accept vagueness.
    """
    text = str(text).lower()
    if _normalized(field) in _normalized(text) or str(field).lower() in text:
        return True
    tokens = _field_tokens(field)
    if not tokens:
        return False
    hits = sum(1 for token in tokens if _token_present(token, text))
    return hits / len(tokens) >= FIELD_TOKEN_COVERAGE


def _strip_non_business_numbers(text: str) -> str:
    """Remove hex colours and citation years before looking for invented numbers."""
    return _CITATION_YEAR.sub(" ", _HEX_COLOR.sub(" ", text))


def invented_numbers(payload_obj: Any, allowed: Set[str], key: str = "",
                     design_context: bool = False) -> Set[str]:
    """Numbers in the payload that are neither in the source nor pure design values.

    Numeric values under interface keys (grid width, font size, column count) are
    design specifications, not analytical claims, and are ignored -- including
    nested ones, so ``position: {"x": 0, "y": 1}`` stays a layout coordinate.
    Every number inside free text -- thresholds, targets, percentages, top-N,
    dates -- must already exist in the source record.
    """
    found: Set[str] = set()
    if isinstance(payload_obj, Mapping):
        for sub_key, value in payload_obj.items():
            nested_design = design_context or str(sub_key).lower() in DESIGN_NUMERIC_KEYS
            found |= invented_numbers(value, allowed, str(sub_key), nested_design)
    elif isinstance(payload_obj, (list, tuple)):
        for value in payload_obj:
            found |= invented_numbers(value, allowed, key, design_context)
    elif isinstance(payload_obj, bool):
        return found
    elif isinstance(payload_obj, (int, float)):
        if not design_context and key.lower() not in DESIGN_NUMERIC_KEYS:
            token = str(payload_obj)
            if token not in allowed:
                found.add(token)
    elif isinstance(payload_obj, str):
        if design_context or key.lower() in DESIGN_NUMERIC_KEYS:
            return found
        found |= {n for n in _numbers(_strip_non_business_numbers(payload_obj)) if n not in allowed}
    return found


def _string_values(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, Mapping):
        for value in obj.values():
            yield from _string_values(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _string_values(value)


def validate_payload(record: Mapping[str, Any], payload: EnrichmentPayload) -> Tuple[List[str], List[str]]:
    """Content validation of an already schema-valid payload.

    Returns (reason_codes, human-readable details). Empty codes means accepted.
    """
    codes: List[str] = []
    details: List[str] = []

    def fail(code: str, detail: str) -> None:
        if code not in codes:
            codes.append(code)
        details.append(detail)

    mapping = _mapping_of(record)
    chart_type = str(mapping.get("chart_type", ""))
    task_type = str(mapping.get("task_type", ""))
    kpis = [str(k) for k in (record.get("brief") or {}).get("kpis") or []]
    columns = source_columns(record)
    normalized_columns = {_normalized(c) for c in columns}
    payload_text = _payload_text(payload)
    payload_lower = payload_text.lower()

    # 1. non-empty enrichment
    if not payload.users.strip():
        fail("empty_enrichment_field", "users is blank")
    for name in ("context_summary", "layout", "styling"):
        if not getattr(payload, name):
            fail("empty_enrichment_field", f"{name} is empty")
    if not payload.interactions:
        fail("empty_enrichment_field", "interactions is empty")
    if not payload.rationales:
        fail("empty_enrichment_field", "rationales is empty")

    # 2. interactions reference source fields only, with a supported verb
    for idx, interaction in enumerate(payload.interactions):
        itype = str(interaction.get("type", "")).strip().lower()
        if itype not in ALLOWED_INTERACTION_TYPES:
            fail("unsupported_interaction_type", f"interactions[{idx}].type={itype!r}")
        fields = interaction.get("fields")
        if isinstance(fields, str):
            fields = [fields]
        if not fields:
            fail("interaction_field_not_in_source", f"interactions[{idx}] names no field")
            continue
        for field_name in fields:
            if _normalized(field_name) not in normalized_columns:
                fail("interaction_field_not_in_source",
                     f"interactions[{idx}] field {field_name!r} not in source columns")

    # 3. context_summary must not contradict the source record
    source_summary = (record.get("recommendation") or {}).get("context_summary") or {}
    for key, expected in source_summary.items():
        if key in payload.context_summary and payload.context_summary[key] != expected:
            fail("context_summary_conflicts_source",
                 f"context_summary.{key}={payload.context_summary[key]!r} != source {expected!r}")

    # 4. no invented KPI or chart type anywhere in the payload
    known_kpi_tokens = {_normalized(k) for k in kpis} | {_normalized(mapping.get("kpi", ""))}
    for block in payload.layout.get("blocks") or []:
        if not isinstance(block, Mapping):
            continue
        if "kpi" in block and _normalized(block["kpi"]) not in known_kpi_tokens:
            fail("invented_kpi", f"layout block kpi {block['kpi']!r} not in source KPIs")
        # The payload has no writable chart_type field -- chart correctness is held by
        # the immutable fingerprint. A block may therefore point at the given chart by
        # name, by alias, or by the neutral phrase the prompt asks for; only naming a
        # *different* chart type is a violation.
        if "chart" in block and not is_source_chart_reference(block["chart"], chart_type):
            foreign = [c for c in named_chart_types(block["chart"]) if c != chart_type]
            if foreign:
                fail("invented_chart_type",
                     f"layout block chart {block['chart']!r} names another chart type {foreign[0]!r}")
    # Short field values that name a chart type (e.g. {"chart": "pie"}) replace the
    # chart outright. Long prose is judged by intent only, with mentions of the given
    # chart masked so aliases like "stacked bar chart" cannot look foreign.
    # A source column may itself be called "Area", "Line" or "Table" (nvBench has
    # such columns). Referencing it is data, not a chart proposal.
    source_field_tokens = {_normalized(c) for c in columns}
    short_values = [v for v in _string_values(payload.model_dump())
                    if len(v.split()) <= 4 and _normalized(v) not in source_field_tokens]
    prose = mask_source_chart(payload_lower, chart_type)
    for chart_word in CHART_VOCABULARY:
        if chart_word == chart_type:
            continue
        if any(chart_word in named_chart_types(value) and not is_source_chart_reference(value, chart_type)
               for value in short_values):
            fail("invented_chart_type", f"payload sets a field to foreign chart type {chart_word!r}")
        elif proposes_chart_replacement(prose, chart_word):
            fail("invented_chart_type", f"payload proposes the foreign chart type {chart_word!r}")

    # 5. no invented numeric target or business fact (design numbers are allowed)
    invented = invented_numbers(payload.model_dump(), _numbers(source_text(record)))
    if invented:
        fail("invented_numeric_value", f"numbers absent from source: {sorted(invented)[:5]}")

    # 6. rationales must agree with task, chart and encoding
    claims = " ".join(f"{r.claim} {r.principle}" for r in payload.rationales).lower()
    names_chart = (
        chart_type in claims
        or chart_type.replace("_", " ") in claims
        # The prompt asks for the neutral phrase, which is the safest reference.
        or re.search(r"selected (?:chart|visuali[sz]ation)", claims) is not None
    )
    if chart_type and not names_chart:
        fail("rationale_disagrees_with_task_chart_or_encoding",
             f"no rationale mentions the chart type {chart_type!r} or 'the selected chart'")
    synonyms = TASK_SYNONYMS.get(task_type, (task_type,))
    if task_type and not any(s in claims for s in synonyms):
        fail("rationale_disagrees_with_task_chart_or_encoding",
             f"no rationale mentions the task {task_type!r}")
    encoding = mapping.get("encoding") or {}
    encoded_fields = [str(encoding.get(k)) for k in ("x", "y", "source_x", "source_y") if encoding.get(k)]
    if encoded_fields and not any(references_encoded_field(claims, f) for f in encoded_fields):
        fail("rationale_disagrees_with_task_chart_or_encoding",
             "no rationale mentions an encoded field")

    return codes, details


# ------------------------------------------------------------------ merge


def merge_enrichment(record: Mapping[str, Any], payload: EnrichmentPayload,
                     provenance: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Trusted merge: write the six presentation fields into a deep copy.

    Immutable parts are copied untouched; the caller re-checks the fingerprint.
    """
    merged = copy.deepcopy(dict(record))
    brief = merged.setdefault("brief", {})
    recommendation = merged.setdefault("recommendation", {})

    brief["users"] = payload.users.strip()
    recommendation["context_summary"] = copy.deepcopy(payload.context_summary)
    recommendation["layout"] = copy.deepcopy(payload.layout)
    recommendation["styling"] = copy.deepcopy(payload.styling)
    recommendation["interactions"] = copy.deepcopy(payload.interactions)
    recommendation["rationales"] = [r.model_dump() for r in payload.rationales]

    extra = brief.setdefault("extra", {})
    lineage = extra.setdefault("lineage", {})
    for key in ENRICHED_LINEAGE_KEYS:
        lineage[key] = LINEAGE_ENRICHED_VALUE
    if provenance:
        extra["enrichment"] = dict(provenance)
    return merged


def enrichment_provenance(model: str, applied_reasoning_mode: str, requested_effort: Optional[str],
                          temperature: float, prompt_sha256: str, run_id: str,
                          fingerprint_before: str) -> Dict[str, Any]:
    """Secret-free provenance block recorded on every enriched record."""
    return {
        "spec_version": ENRICHMENT_SPEC_VERSION,
        "enriched_fields": list(ENRICHABLE_FIELDS),
        "model": model,
        "requested_reasoning_effort": requested_effort,
        "applied_reasoning_mode": applied_reasoning_mode,
        "temperature": temperature,
        "prompt_sha256": prompt_sha256,
        "run_id": run_id,
        "immutable_fingerprint": fingerprint_before,
    }


# ----------------------------------------------------------------- prompt


JSON_EXAMPLE = """{
  "users": "Sales analyst comparing order counts per region.",
  "context_summary": {"db_id": "orders_db", "source": "nvbench", "n_kpis": 1,
                      "data_scope": "orders grouped by region"},
  "layout": {"type": "single", "blocks": [{"kpi": "COUNT(order_id)", "chart": "the selected chart"}]},
  "styling": {"theme": "minimal", "emphasis": "ordered categories"},
  "interactions": [{"type": "tooltip", "fields": ["region", "COUNT(order_id)"]},
                   {"type": "sort", "fields": ["COUNT(order_id)"]}],
  "rationales": [{"claim": "The selected chart supports comparison of COUNT(order_id) across region.",
                  "principle": "position on a common scale (Cleveland & McGill)"}]
}"""

SYSTEM_PROMPT = (
    "You are a dashboard-design expert enriching an existing, verified dashboard "
    "specification. The analytical content is already fixed and correct: KPI, chart "
    "type, task type, encoding, filters, sorting, grouping and time grain must be "
    "treated as given facts you may describe but never change, extend or contradict.\n"
    "Return ONLY one JSON object with exactly these six top-level keys and these types:\n"
    "  users            string  (one short persona sentence)\n"
    "  context_summary  object  (string/number values)\n"
    "  layout           object  (NOT a list; put blocks under a \"blocks\" key)\n"
    "  styling          object  (NOT a list)\n"
    "  interactions     list of objects {\"type\": string, \"fields\": [string]}\n"
    "  rationales       list of objects {\"claim\": string, \"principle\": string}\n"
    "No other top-level key, no prose outside the JSON, no code fence.\n"
    "Hard rules:\n"
    "1. Invent no KPI, no metric, no threshold, no target, no percentage, no date, no "
    "top-N value, no company, product, currency or industry fact. Avoid explicit numeric "
    "values in layout, styling, interactions and rationales unless the value already "
    "exists in the supplied source evidence.\n"
    "2. Refer to the visualization only as \"the selected chart\". Do not name, introduce, "
    "compare or recommend alternative chart types.\n"
    "3. interactions: allowed verbs are " + ", ".join(ALLOWED_INTERACTION_TYPES) + ". Every "
    "field must be one of the listed source fields, spelled exactly as given.\n"
    "4. context_summary: keep every key/value of the given context summary unchanged; you "
    "may add a few short descriptive string keys.\n"
    "5. layout blocks may reference only the given KPI; for a block's \"chart\" value write "
    "\"the selected chart\" or copy the given chart_type verbatim -- never another type.\n"
    "6. rationales: one short rationale per analytical mapping (at most two). Together they "
    "must name the selected chart, the given analytical task and at least one encoded "
    "field, and cite an established visualization principle.\n"
    "7. Be concise: short persona, compact context_summary, compact layout and styling, at "
    "most four relevant interactions, no essays, no repeated explanations.\n"
    "Exact shape to follow:\n" + JSON_EXAMPLE
)


def build_user_prompt(record: Mapping[str, Any]) -> str:
    """Deterministic, source-faithful prompt body for one record."""
    brief = record.get("brief") or {}
    mapping = _mapping_of(record)
    encoding = mapping.get("encoding") or {}
    provenance = (brief.get("extra") or {}).get("provenance") or {}
    task_inference = (brief.get("extra") or {}).get("task_inference") or {}

    facts = {
        "database": provenance.get("db_id"),
        "goal": (brief.get("goals") or [None])[0],
        "kpi": mapping.get("kpi"),
        "task_type": mapping.get("task_type"),
        "task_evidence": task_inference.get("evidence"),
        "chart_type": mapping.get("chart_type"),
        "valid_alternatives": mapping.get("alternatives"),
        "columns": brief.get("columns"),
        "encoding": {
            "x": encoding.get("x"), "y": encoding.get("y"),
            "aggregate": encoding.get("aggregate"),
            "grouped": encoding.get("grouped"), "group_field": encoding.get("group_field"),
            "filters": encoding.get("filters"), "sort": encoding.get("sort"),
            "limit": encoding.get("limit"), "time_grain": encoding.get("time_grain"),
            "visual_grouping": encoding.get("visual_grouping"),
        },
        "context_summary_to_preserve": (record.get("recommendation") or {}).get("context_summary"),
        "allowed_source_fields": sorted(source_columns(record)),
    }
    return (
        "Enrich this dashboard record. Fixed source facts:\n"
        + json.dumps(facts, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        + "\n\nReturn the JSON object with exactly the six allowed keys."
    )


def build_messages(record: Mapping[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(record)},
    ]


def prompt_sha256(messages: Sequence[Mapping[str, str]]) -> str:
    return hashlib.sha256(_canonical(list(messages)).encode("utf-8")).hexdigest()


# -------------------------------------------------------------- selection


def _chart_of(record: Mapping[str, Any]) -> str:
    return str(_mapping_of(record).get("chart_type", "unknown"))


def _group_of(record: Mapping[str, Any]) -> str:
    provenance = ((record.get("brief") or {}).get("extra") or {}).get("provenance") or {}
    return str(provenance.get("source_group_id") or record.get("item_id"))


def select_records(train: Sequence[Mapping[str, Any]], val: Sequence[Mapping[str, Any]],
                   n: int, seed: int = 42,
                   exclude_item_ids: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """Deterministic, chart-stratified, source-group-disjoint selection.

    Charts are filled round-robin in descending corpus frequency, so every chart
    type present reaches the sample before any chart type is doubled. Within a
    chart the candidates are ordered by a seeded hash of the item_id, making the
    result reproducible and independent of file order. Rounds alternate the
    preferred split (train, val, train, ...) with fallback to the other, so both
    splits are represented instead of one dominating a small sample. ``test`` is
    never an input.
    """
    excluded = {str(i) for i in (exclude_item_ids or ())}
    pool: List[Dict[str, Any]] = []
    for split_name, records in (("val", val), ("train", train)):
        for record in records:
            if str(record.get("split", "")).lower() == "test":
                continue
            if str(record.get("item_id")) in excluded:
                continue
            item = dict(record)
            item.setdefault("split", split_name)
            pool.append(item)

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for record in pool:
        buckets.setdefault(_chart_of(record), []).append(record)

    def order_key(record: Mapping[str, Any]) -> Tuple[str, str]:
        digest = hashlib.sha256(f"{seed}:{record.get('item_id')}".encode("utf-8")).hexdigest()
        return (digest, str(record.get("item_id")))

    for chart in buckets:
        buckets[chart].sort(key=order_key)

    chart_order = sorted(buckets, key=lambda c: (-len(buckets[c]), c))
    selected: List[Dict[str, Any]] = []
    used_groups: Set[str] = set()
    round_index = 0
    exhausted = False
    while len(selected) < n and not exhausted:
        exhausted = True
        preferred = "train" if round_index % 2 == 0 else "val"
        for chart in chart_order:
            if len(selected) >= n:
                break
            candidates = buckets[chart]
            picked = None
            for wanted in (preferred, None):  # None = any remaining split
                for position, candidate in enumerate(candidates):
                    if wanted is not None and str(candidate.get("split")) != wanted:
                        continue
                    if _group_of(candidate) in used_groups:
                        continue
                    picked = candidates.pop(position)
                    break
                if picked is not None:
                    break
            if picked is None:
                continue
            used_groups.add(_group_of(picked))
            selected.append(picked)
            exhausted = False
        round_index += 1
    # Drop candidates whose group was consumed by a later round's pick order.
    return selected[:n]


def selection_summary(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    records = list(records)
    charts: Dict[str, int] = {}
    splits: Dict[str, int] = {}
    for record in records:
        charts[_chart_of(record)] = charts.get(_chart_of(record), 0) + 1
        split = str(record.get("split", "unknown"))
        splits[split] = splits.get(split, 0) + 1
    return {
        "n": len(records),
        "chart_type": dict(sorted(charts.items())),
        "split": dict(sorted(splits.items())),
        "unique_source_groups": len({_group_of(r) for r in records}),
    }
