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
    r"\bWHERE\b(.*?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bBIN\b|$)", re.IGNORECASE | re.DOTALL
)
_GROUP_BY_RE = re.compile(
    r"\bGROUP\s+BY\b(.*?)(?:\bORDER\s+BY\b|\bBIN\b|\bLIMIT\b|\bHAVING\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_ORDER_BY_RE = re.compile(
    r"\bORDER\s+BY\b(.*?)(?:\bBIN\b|$)", re.IGNORECASE | re.DOTALL
)
_BIN_RE = re.compile(r"\bBIN\s+([\w.]+)\s+BY\s+(\w+)", re.IGNORECASE)
_COND_RE = re.compile(
    r"^\s*([\w.]+)\s*(=|!=|<>|>=|<=|>|<|LIKE|IN|NOT\s+IN)\s*(.+?)\s*$", re.IGNORECASE
)


def _strip_alias(name: str) -> str:
    return name.split(".")[-1].strip() if name else name


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
    arg = m.group(2).strip()
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


def extract_time_grain(binning: str) -> Optional[Dict[str, str]]:
    """Parse ``BIN <field> BY <GRAIN>``; returns ``{field, grain}`` or ``None``."""
    m = _BIN_RE.search(binning or "")
    if not m:
        return None
    return {"field": _strip_alias(m.group(1)), "grain": m.group(2).upper()}
