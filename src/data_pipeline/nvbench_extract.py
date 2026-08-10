"""Deterministic source-clause extractors for nvBench SQL/VQL/NL text.

Every function here is pure (no I/O beyond an injected ``DbMetadataResolver``) and
corpus-verified against ``NVBench.json`` (see the pilot-v3 plan). These extractors
back the builder's source-constraint preservation and rejection policy: filters,
sort, time-bin, nested aggregates, and two explicitly narrow conflict rules.

The conflict rules are intentionally narrow. A naive "keyword vs aggregate function"
scan false-positives at 7-10% of aggregate/query pairs in the real corpus, because
nvBench's generated NL uses "total" as a generic intensifier (e.g. "the total
minimal stu_gpa" means MIN, not SUM) and "number of" commonly co-occurs with
unrelated aggregates. The rules below fire only in the narrow, verified-safe cases.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_AGG_FUNCS = ("COUNT", "SUM", "AVG", "MIN", "MAX")
_AGG_CALL_RE = re.compile(r"^\s*(\w+)\s*\(\s*(.*?)\s*\)\s*$")
_NESTED_RE = re.compile(
    r"^\s*(\w+)\s*\(\s*(\w+)\s*\(\s*(.*?)\s*\)\s*\)\s*$", re.IGNORECASE
)
_ID_SHAPED_RE = re.compile(r"(^|_)id$", re.IGNORECASE)

_CHART_KEYWORDS = {
    "bar": "Bar",
    "pie": "Pie",
    "line": "Line",
    "scatter": "Scatter",
    "stacked bar": "Stacked Bar",
}

_WHERE_RE = re.compile(
    r"\bWHERE\b(.*?)(?:\bGROUP\s+BY\b|\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b|\bBIN\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_GROUP_BY_RE = re.compile(
    r"\bGROUP\s+BY\b(.*?)(?:\bORDER\s+BY\b|\bBIN\b|\bLIMIT\b|\bHAVING\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_ORDER_BY_RE = re.compile(
    r"\bORDER\s+BY\b(.*?)(?:\bLIMIT\b|\bOFFSET\b|\bFETCH\b|\bBIN\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_BIN_RE = re.compile(r"\bBIN\s+([\w.]+)\s+BY\s+(\w+)", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)
_TOP_RE = re.compile(
    r"\bSELECT\s+(?:DISTINCT\s+)?TOP\s*(?:\(\s*)?(\d+)(?:\s*\))?(?=\s|[A-Za-z_`\"\[])",
    re.IGNORECASE,
)
_HAVING_RE = re.compile(
    r"\bHAVING\b(.*?)(?:\bORDER\s+BY\b|\bLIMIT\b|\bOFFSET\b|\bFETCH\b|\bBIN\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_COND_RE = re.compile(
    r"^\s*([\w.]+)\s*(=|!=|<>|>=|<=|>|<|LIKE|IN|NOT\s+IN)\s*(.+?)\s*$", re.IGNORECASE
)
_AGG_COND_RE = re.compile(
    r"^\s*((?:COUNT|SUM|AVG|MIN|MAX)\s*\(\s*(?:\*|[\w.]+)\s*\))\s*"
    r"(=|!=|<>|>=|<=|>|<)\s*(.+?)\s*$",
    re.IGNORECASE,
)

_SUPPORTED_TIME_GRAINS = ("YEAR", "QUARTER", "MONTH", "WEEK", "WEEKDAY", "DAY", "HOUR", "DATE")


def _strip_alias(name: str) -> str:
    """Strip a table qualifier from a simple identifier, never from an expression."""
    if not name:
        return name
    stripped = name.strip()
    if re.fullmatch(r"[\w.`\"\[\]]+", stripped):
        return stripped.split(".")[-1].strip().strip('`"[]')
    return stripped


# --------------------------------------------------------------------------- #
# aggregate expression parsing
# --------------------------------------------------------------------------- #
def extract_base_field(expr: str) -> Optional[str]:
    """Raw column referenced by an aggregate expression.

    ``SUM(order_quantity)`` -> ``order_quantity``; ``COUNT(*)`` -> ``None`` (never
    invents a physical ``*`` column). Not an aggregate expression -> ``None``.
    """
    if not expr:
        return None
    m = _AGG_CALL_RE.match(expr)
    if not m or m.group(1).upper() not in _AGG_FUNCS:
        return None
    arg = re.sub(r"^DISTINCT\s+", "", m.group(2).strip(), flags=re.IGNORECASE)
    if arg == "*" or not arg:
        return None
    return _strip_alias(arg)


def extract_nested(expr: str) -> Optional[Dict[str, str]]:
    """Detect ``outer(inner(arg))``; returns ``{outer, inner, arg}`` or ``None``."""
    if not expr:
        return None
    m = _NESTED_RE.match(expr)
    if not m:
        return None
    outer, inner, arg = m.group(1).upper(), m.group(2).upper(), m.group(3).strip()
    if outer not in _AGG_FUNCS or inner not in _AGG_FUNCS:
        return None
    return {"outer": outer, "inner": inner, "arg": arg}


def resolve_nested(nested: Dict[str, str], nl_queries: List[str]) -> Dict[str, Any]:
    """Corpus-verified nested-aggregate resolution.

    ``outer == inner`` and the inner argument is a real column (not ``*``) ->
    normalize by collapsing to ``outer(arg)`` (e.g. ``SUM(sum(x))`` -> ``SUM(x)``).
    Otherwise: if an unambiguous, narrow query-intent conflict is detected, reject
    with ``aggregate_intent_conflict``; else reject as ``nested_aggregate_ambiguous``
    (cannot reduce to a single raw base field without inventing meaning).
    """
    outer, inner, arg = nested["outer"], nested["inner"], nested["arg"]
    if outer == inner and arg and arg != "*":
        return {"action": "normalize", "collapsed": f"{outer}({arg})"}

    intent = detect_query_intent(nl_queries, allow_count_number_of=False)
    if intent and intent != outer:
        return {"action": "reject", "reason": "aggregate_intent_conflict",
                "detail": f"nested expression outer={outer} conflicts with query intent={intent}"}
    return {"action": "reject", "reason": "nested_aggregate_ambiguous",
            "detail": f"outer={outer} inner={inner} arg={arg!r}: cannot reduce to one raw base field"}


# --------------------------------------------------------------------------- #
# narrow query-intent conflict rules
# --------------------------------------------------------------------------- #
def detect_query_intent(nl_queries: List[str], *, allow_count_number_of: bool = True) -> Optional[str]:
    """Mutually-exclusive aggregate-intent keyword scan (narrow, low-recall by design).

    ``allow_count_number_of`` gates whether "number of"/"total number"/"how many"
    map to COUNT — callers that only care about the SUM-vs-other rule (a) disable
    this so "total number of X" is not mistaken for a plain SUM signal.
    """
    for nl in nl_queries:
        n = nl.lower()
        is_count_idiom = bool(
            re.search(r"total\s+number\b", n) or "number of" in n or "how many" in n
            or "count of" in n or "amount of" in n
        )
        families = []
        if allow_count_number_of and is_count_idiom:
            families.append("COUNT")
        if re.search(r"\baverage\b|\bavg\b|\bmean\b", n):
            families.append("AVG")
        if re.search(r"\bminimal\b|\bminimum\b|\blowest\b|\bsmallest\b", n):
            families.append("MIN")
        if re.search(r"\bmaximal\b|\bmaximum\b|\bhighest\b|\blargest\b", n):
            families.append("MAX")
        # "total number of X" is a COUNT idiom, never a SUM signal, regardless of
        # whether the COUNT family itself is gated off for this call.
        if re.search(r"\btotal\b", n) and not families and not is_count_idiom:
            families.append("SUM")
        if len(families) == 1:
            return families[0]
    return None


def check_aggregate_intent_conflict(
    expr: str, nl_queries: List[str], resolver, db_id: str
) -> Optional[str]:
    """Two narrow, corpus-verified conflict rules; returns a detail string or ``None``.

    Rule (a): the query says "total <x>" with no other aggregate keyword present
    (so "total" unambiguously signals SUM), but the encoded function is not SUM.
    Rule (b): the encoded function is SUM over an id-shaped raw column (e.g.
    ``Shop_ID``) while the query says "number of"/"total number" (a COUNT idiom) —
    summing an identifier's values is very unlikely to be the intended measure.
    """
    m = _AGG_CALL_RE.match(expr or "")
    if not m or m.group(1).upper() not in _AGG_FUNCS:
        return None
    outer = m.group(1).upper()
    arg = m.group(2).strip()

    # Rule (a): unambiguous "total" (no co-occurring keyword) vs non-SUM function.
    intent = detect_query_intent(nl_queries, allow_count_number_of=False)
    if intent == "SUM" and outer != "SUM":
        return f"query intent=SUM ('total ...') conflicts with encoded function={outer}"

    # Rule (b): SUM over an id-shaped field + a COUNT idiom in the query.
    if outer == "SUM" and _ID_SHAPED_RE.search(_strip_alias(arg)):
        for nl in nl_queries:
            n = nl.lower()
            if re.search(r"total\s+number\b", n) or "number of" in n:
                return f"SUM over id-shaped field {arg!r} conflicts with COUNT-like phrasing in query"
    return None


def check_chart_query_conflict(source_chart: str, nl_queries: List[str]) -> Optional[str]:
    """Explicit '<other chart> chart/plot' phrase with the source's own keyword absent."""
    src_variants = {source_chart}
    if source_chart.startswith("Grouping "):
        src_variants.add(source_chart.replace("Grouping ", ""))
    for nl in nl_queries:
        n = nl.lower()
        mentioned = [name for kw, name in _CHART_KEYWORDS.items()
                     if f"{kw} chart" in n or f"{kw} plot" in n]
        conflicting = [m for m in mentioned if m not in src_variants
                       and not (source_chart == "Stacked Bar" and m == "Bar")]
        if conflicting:
            return f"query explicitly requests {conflicting[0]!r} chart; source chart is {source_chart!r}"
    return None


# --------------------------------------------------------------------------- #
# WHERE / ORDER BY / BIN extraction
# --------------------------------------------------------------------------- #
def extract_where_filters(sql: str) -> Dict[str, Any]:
    """Extract simple ``field OP value`` filters from a ``WHERE`` clause.

    A subquery or top-level ``OR`` cannot be flattened into an unordered filter
    list without changing meaning -> ``status: "unrepresentable"``. Any individual
    AND-joined condition that fails to parse also makes the whole clause
    unrepresentable (never drop one condition silently).
    """
    m = _WHERE_RE.search(sql or "")
    if not m:
        return {"filters": [], "status": "none"}
    seg = m.group(1).strip()
    if not seg:
        return {"filters": [], "status": "none"}
    if re.search(r"\(\s*SELECT\b", seg, re.IGNORECASE):
        return {"filters": [], "status": "unrepresentable", "detail": "subquery in WHERE clause"}
    if re.search(r"\bOR\b", seg, re.IGNORECASE):
        return {"filters": [], "status": "unrepresentable", "detail": "OR in WHERE clause"}

    filters: List[Dict[str, str]] = []
    for cond in re.split(r"\bAND\b", seg, flags=re.IGNORECASE):
        cond = cond.strip()
        if not cond:
            continue
        cm = _COND_RE.match(cond)
        if not cm:
            return {"filters": [], "status": "unrepresentable",
                     "detail": f"unparseable condition: {cond!r}"}
        field, op, value = cm.group(1), cm.group(2).upper(), cm.group(3).strip().strip("'\"")
        filters.append({"field": _strip_alias(field), "operator": op, "value": value})
    return {"filters": filters, "status": "ok"}


def extract_group_by_fields(sql: str) -> List[str]:
    """Raw column names in a ``GROUP BY`` clause, order preserved, deduplicated."""
    m = _GROUP_BY_RE.search(sql or "")
    if not m:
        return []
    seen: set = set()
    out: List[str] = []
    for part in m.group(1).split(","):
        col = _strip_alias(part.strip())
        low = col.lower()
        if col and low not in seen:
            seen.add(low)
            out.append(col)
    return out


def classify_group_by_terms(sql: str) -> List[Dict[str, Any]]:
    """Classify each ``GROUP BY`` term without treating expressions as fields.

    Aggregate calls are invalid grouping keys for the supported SQLite source
    dialect. Other computed expressions remain explicit ``expression`` terms;
    callers may preserve them as source evidence but must not claim that they
    are physical grouping columns.
    """
    terms: List[Dict[str, Any]] = []
    for expression in extract_group_by_fields(sql):
        aggregate_match = _AGG_CALL_RE.match(expression)
        if aggregate_match and aggregate_match.group(1).upper() in _AGG_FUNCS:
            terms.append({
                "expression": expression,
                "kind": "invalid_aggregate_expression",
                "aggregate": aggregate_match.group(1).upper(),
                "field": extract_base_field(expression),
            })
        elif re.fullmatch(r"[\w.`\"\[\]]+", expression):
            terms.append({
                "expression": expression,
                "kind": "field",
                "aggregate": None,
                "field": _strip_alias(expression),
            })
        else:
            terms.append({
                "expression": expression,
                "kind": "expression",
                "aggregate": None,
                "field": None,
            })
    return terms


def extract_order_by(sql_or_vql: str) -> Dict[str, Any]:
    """Extract a single sort key + direction from ``ORDER BY``.

    Multiple comma-separated sort keys are ambiguous to represent as one sort spec
    -> ``status: "unrepresentable"``. Direction defaults to ``"asc"`` (the SQL
    default) only when no explicit ``ASC``/``DESC`` token is present.
    """
    m = _ORDER_BY_RE.search(sql_or_vql or "")
    if not m:
        return {"field": None, "direction": None, "status": "none"}
    seg = m.group(1).strip()
    if not seg:
        return {"field": None, "direction": None, "status": "none"}
    if "," in seg:
        return {"field": None, "direction": None, "status": "unrepresentable",
                "detail": "multiple ORDER BY keys"}

    direction = "asc"
    if re.search(r"\bDESC\b", seg, re.IGNORECASE):
        direction = "desc"
        seg = re.sub(r"\bDESC\b", "", seg, flags=re.IGNORECASE)
    elif re.search(r"\bASC\b", seg, re.IGNORECASE):
        seg = re.sub(r"\bASC\b", "", seg, flags=re.IGNORECASE)
    field = seg.strip()
    if not field:
        return {"field": None, "direction": None, "status": "unrepresentable",
                "detail": "empty ORDER BY expression"}
    return {"field": _strip_alias(field), "expression": field, "direction": direction, "status": "ok"}


def extract_limit(sql_or_vql: str) -> Dict[str, Any]:
    """Extract a row limit from either ``LIMIT N`` or ``SELECT TOP N``.

    The two syntaxes are normalized into one integer ``value`` and retain their
    source syntax for provenance. If both appear with different values, the
    constraint is internally inconsistent and therefore unrepresentable.
    """
    text = sql_or_vql or ""
    limit_match = _LIMIT_RE.search(text)
    top_match = _TOP_RE.search(text)
    values = []
    if limit_match:
        values.append((int(limit_match.group(1)), "limit"))
    if top_match:
        values.append((int(top_match.group(1)), "top"))
    if not values:
        return {"value": None, "syntax": None, "status": "none"}
    unique_values = {value for value, _syntax in values}
    if len(unique_values) != 1:
        return {
            "value": None,
            "syntax": None,
            "status": "unrepresentable",
            "detail": f"conflicting row limits: {values}",
        }
    value = values[0][0]
    if value <= 0:
        return {
            "value": None,
            "syntax": values[0][1],
            "status": "unrepresentable",
            "detail": "row limit must be positive",
        }
    return {"value": value, "syntax": values[0][1], "status": "ok"}


def extract_having_conditions(sql_or_vql: str) -> Dict[str, Any]:
    """Normalize simple AND-joined aggregate predicates from ``HAVING``.

    Conditions that cannot be represented without changing Boolean semantics
    fail closed. No condition is inferred from natural language here.
    """
    match = _HAVING_RE.search(sql_or_vql or "")
    if not match:
        return {"conditions": [], "status": "none"}
    segment = match.group(1).strip()
    if not segment:
        return {"conditions": [], "status": "unrepresentable", "detail": "empty HAVING clause"}
    if re.search(r"\bOR\b", segment, re.IGNORECASE):
        return {"conditions": [], "status": "unrepresentable", "detail": "OR in HAVING clause"}

    conditions: List[Dict[str, Any]] = []
    for raw_condition in re.split(r"\bAND\b", segment, flags=re.IGNORECASE):
        condition = raw_condition.strip()
        parsed = _AGG_COND_RE.match(condition)
        if not parsed:
            return {
                "conditions": [],
                "status": "unrepresentable",
                "detail": f"unparseable aggregate condition: {condition!r}",
            }
        expression, operator, value = parsed.group(1), parsed.group(2), parsed.group(3).strip().strip("'\"")
        aggregate = _AGG_CALL_RE.match(expression)
        assert aggregate is not None
        conditions.append({
            "expression": re.sub(r"\s+", "", expression).upper(),
            "aggregate": aggregate.group(1).upper(),
            "field": None if aggregate.group(2).strip() == "*" else _strip_alias(aggregate.group(2)),
            "operator": operator,
            "value": value,
        })
    return {"conditions": conditions, "status": "ok"}


def extract_time_grain(binning: str) -> Optional[Dict[str, str]]:
    """Parse ``BIN <field> BY <GRAIN>``; returns ``{field, grain}`` or ``None``."""
    m = _BIN_RE.search(binning or "")
    if not m:
        return None
    return {"field": _strip_alias(m.group(1)), "grain": m.group(2).upper()}


def extract_nl_time_grains(nl_query: str) -> List[str]:
    """Return only explicit natural-language binning/grain instructions.

    The patterns intentionally require words such as ``bin`` or ``interval``;
    incidental mentions of dates or years are not treated as constraints.
    """
    text = " ".join((nl_query or "").split())
    grain_alt = "|".join(g.lower() for g in _SUPPORTED_TIME_GRAINS)
    patterns = (
        rf"\bbin(?:ning)?\b.*?\bby\s+({grain_alt})\b",
        rf"\bbin(?:ning)?\b.*?\binto\s+(?:the\s+)?({grain_alt})\s+interval\b",
        rf"\binto\s+(?:the\s+)?({grain_alt})\s+interval\b",
    )
    found: List[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            grain = match.group(1).upper()
            if grain not in found:
                found.append(grain)
    return found


def extract_nl_aggregate_conditions(nl_query: str) -> List[Dict[str, str]]:
    """Extract exact count thresholds stated in natural language.

    This is diagnostic evidence only. It does not create SQL or VQL clauses.
    The narrow comparative patterns avoid confusing ``top N`` requests with
    aggregate filters.
    """
    text = " ".join((nl_query or "").lower().split())
    operators = {
        "more than": ">",
        "greater than": ">",
        "fewer than": "<",
        "less than": "<",
        "at least": ">=",
        "at most": "<=",
        "exactly": "=",
    }
    alt = "|".join(re.escape(phrase) for phrase in operators)
    pattern = re.compile(rf"\b({alt})\s+(\d+)\s+([a-z][a-z0-9_-]*)\b", re.IGNORECASE)
    out: List[Dict[str, str]] = []
    for match in pattern.finditer(text):
        subject = match.group(3).rstrip("s")
        condition = {
            "aggregate": "COUNT",
            "operator": operators[match.group(1).lower()],
            "value": match.group(2),
            "subject": subject,
            "origin": "natural_language",
        }
        if condition not in out:
            out.append(condition)
    return out


def extract_nl_required_dimensions(nl_query: str) -> List[Dict[str, str]]:
    """Extract narrow, explicit analytical-dimension requests from NL text.

    Returned phrases must still be checked against selected/grouped/binned
    source fields; this function never guesses a missing database column.
    """
    text = " ".join((nl_query or "").lower().split())
    out: List[Dict[str, str]] = []

    def add(name: str, origin: str) -> None:
        normalized = re.sub(r"[^a-z0-9_]+", "_", name).strip("_")
        if normalized.endswith("ies") and not normalized.endswith("_series") and normalized != "series":
            normalized = normalized[:-3] + "y"
        elif normalized.endswith("s") and not normalized.endswith(("ss", "series")):
            normalized = normalized[:-1]
        if normalized in {"bin", "axis", "value", "record", "item", "one"}:
            return
        item = {"dimension": normalized, "origin": origin}
        if normalized and not any(existing["dimension"] == normalized for existing in out):
            out.append(item)

    # Keep short noun phrases rather than only their first word: ``each ship
    # type`` requests ``ship_type``, not ``ship``.  Boundaries are explicit
    # discourse/function words so this remains a high-precision extractor.
    phrase_stop = {
        "and", "or", "that", "who", "whose", "which", "where", "with",
        "without", "using", "show", "return", "plot", "listed", "list",
        "sort", "order", "into", "as", "on", "from", "of", "to",
        "in", "the", "this", "these", "those", "named", "please", "what",
        "is", "are", "was", "were", "has", "have",
    }
    for match in re.finditer(
        r"\b(for\s+each|each)\s+"
        r"([a-z][a-z0-9_-]*(?:'s)?(?:\s+[a-z][a-z0-9_-]*(?:'s)?){0,4})",
        text,
    ):
        raw_tokens = match.group(2).split()
        possessive = bool(raw_tokens and raw_tokens[0].endswith("'s"))
        tokens: List[str] = []
        for token in raw_tokens:
            clean = token[:-2] if token.endswith("'s") else token
            # ``move in date`` is a field phrase after a possessive; elsewhere
            # ``in`` and ``of`` delimit the requested dimension.
            if clean in phrase_stop and not (possessive and clean in {"in", "of"}):
                break
            tokens.append(clean)
        if tokens:
            add("_".join(tokens), "for_each" if match.group(1).startswith("for") else "each")
    for match in re.finditer(r"\b(?:a|the)\s+series\s+of\s+([a-z][a-z0-9_-]*)", text):
        noun = match.group(1)
        if noun.endswith("s"):
            noun = noun[:-1]
        add(f"{noun}_series", "series_of")
    for match in re.finditer(r"\b([a-z][a-z0-9_-]*)\s+series\b(?!\s+table\b)", text):
        add(f"{match.group(1)}_series", "entity_series")
    return out


# --------------------------------------------------------------------------- #
# SELECT-list aggregate extraction (axis-aware KPI/SQL agreement)
# --------------------------------------------------------------------------- #
_SELECT_RE = re.compile(r"\bSELECT\b(.*?)\bFROM\b", re.IGNORECASE | re.DOTALL)


def _split_top_level(text: str) -> List[str]:
    """Split on commas that are not nested inside parentheses."""
    parts: List[str] = []
    depth = 0
    buf = []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts


def extract_select_aggregates(sql: str) -> List[Dict[str, Any]]:
    """Aggregate calls in a ``SELECT`` list, in order.

    Returns ``[{func, base_field, position}]`` for each ``FUNC(arg)`` term
    (FUNC in COUNT/SUM/AVG/MIN/MAX). ``base_field`` is the alias/quote-stripped
    argument, or ``None`` for ``COUNT(*)``. Non-aggregate select terms are
    skipped but still advance ``position`` so an aggregate keeps its column
    index (used to line an encoded axis up with its source expression).
    """
    m = _SELECT_RE.search(sql or "")
    if not m:
        return []
    out: List[Dict[str, Any]] = []
    for position, term in enumerate(_split_top_level(m.group(1))):
        alias_match = re.search(
            r"\s+AS\s+([\w.`\"\[\]]+)\s*$", term.strip(), flags=re.IGNORECASE
        )
        # Drop a trailing ``AS alias`` so ``SUM(x) AS total`` still parses.
        clean = re.sub(r"\s+AS\s+[\w.`\"\[\]]+\s*$", "", term.strip(), flags=re.IGNORECASE)
        cm = _AGG_CALL_RE.match(clean)
        if not cm or cm.group(1).upper() not in _AGG_FUNCS:
            continue
        func = cm.group(1).upper()
        arg = re.sub(r"^DISTINCT\s+", "", cm.group(2).strip(), flags=re.IGNORECASE)
        if arg == "*" or not arg:
            base = None
        else:
            base = _strip_alias(arg).strip().strip('`"[]')
        item = {"func": func, "base_field": base, "position": position}
        if alias_match:
            item["alias"] = _strip_alias(alias_match.group(1)).strip().strip('`"[]')
        out.append(item)
    return out


def extract_select_projection_fields(sql: str) -> List[str]:
    """Physical fields referenced directly by the top-level ``SELECT`` list.

    Simple projections and aggregate arguments are returned in source order,
    deduplicated case-insensitively. ``COUNT(*)`` and unsupported computed
    expressions do not invent physical columns.
    """
    match = _SELECT_RE.search(sql or "")
    if not match:
        return []
    fields: List[str] = []
    seen: set = set()
    for term in _split_top_level(match.group(1)):
        clean = re.sub(r"\s+AS\s+[\w.`\"\[\]]+\s*$", "", term.strip(), flags=re.IGNORECASE)
        aggregate = _AGG_CALL_RE.match(clean)
        if aggregate and aggregate.group(1).upper() in _AGG_FUNCS:
            field = extract_base_field(clean)
        elif re.fullmatch(r"[\w.`\"\[\]]+", clean):
            field = _strip_alias(clean)
        else:
            field = None
        key = str(field or "").lower()
        if field and key not in seen:
            seen.add(key)
            fields.append(field)
    return fields


# --------------------------------------------------------------------------- #
# time-grain signals in SQL / VQL (strftime / date / EXTRACT / YEAR(...) ...)
# --------------------------------------------------------------------------- #
# SQLite strftime format codes are case-sensitive; map each to a normalized grain.
_STRFTIME_GRAIN = {
    "%Y": "YEAR", "%y": "YEAR", "%m": "MONTH", "%d": "DAY", "%w": "WEEKDAY",
    "%W": "WEEK", "%j": "DAY", "%H": "HOUR",
}
_STRFTIME_RE = re.compile(r"strftime\s*\(\s*'([^']+)'\s*,\s*([\w.]+)\s*\)", re.IGNORECASE)
_DATEFUNC_RE = re.compile(r"\b(date|datetime)\s*\(\s*([\w.]+)\s*\)", re.IGNORECASE)
_EXTRACT_RE = re.compile(r"\bEXTRACT\s*\(\s*(\w+)\s+FROM\s+([\w.]+)\s*\)", re.IGNORECASE)
_NAMED_GRAIN_RE = re.compile(
    r"\b(YEAR|MONTH|DAY|WEEKDAY|WEEK|QUARTER|HOUR)\s*\(\s*([\w.]+)\s*\)", re.IGNORECASE
)


def extract_time_grain_signals(sql: str, vql: str = "") -> List[Dict[str, str]]:
    """Deterministically detect explicit time-grain expressions in SQL/VQL.

    Recognizes ``strftime('%Y', f)``, ``date(f)``/``datetime(f)``,
    ``EXTRACT(YEAR FROM f)`` and ``YEAR(f)``/``MONTH(f)``/... . Returns a
    deduplicated list of ``{field, grain, source}``; never infers an
    unsupported grain. Grouping/x-axis over a plain datetime *column* (no
    function) is handled by the quality layer with database evidence, not here.
    """
    text = " ".join(t for t in (sql, vql) if t)
    out: List[Dict[str, str]] = []
    seen: set = set()

    def add(field: str, grain: str, source: str) -> None:
        field = _strip_alias(field).strip().strip('`"[]')
        key = (field.lower(), grain)
        if field and key not in seen:
            seen.add(key)
            out.append({"field": field, "grain": grain, "source": source})

    for m in _STRFTIME_RE.finditer(text):
        grain = _STRFTIME_GRAIN.get(m.group(1).strip())
        if grain:
            add(m.group(2), grain, "strftime")
    for m in _DATEFUNC_RE.finditer(text):
        add(m.group(2), "DATE", m.group(1).lower())
    for m in _EXTRACT_RE.finditer(text):
        grain = m.group(1).upper()
        if grain in ("YEAR", "MONTH", "DAY", "WEEK", "WEEKDAY", "QUARTER", "HOUR"):
            add(m.group(2), grain, "extract")
    for m in _NAMED_GRAIN_RE.finditer(text):
        add(m.group(2), m.group(1).upper(), "named_function")
    for m in _BIN_RE.finditer(vql or ""):
        grain = m.group(2).upper()
        if grain in _SUPPORTED_TIME_GRAINS:
            add(m.group(1), grain, "vql_bin")
    return out
