"""Pure identifier-likelihood classification for physical database fields.

No I/O: takes a field profile (see ``nvbench_profile.DbProfiler``) and a field
name, returns an evidence-based identifier decision. Kept separate from
``nvbench_profile`` so it is trivially unit-testable against constructed
profile dicts, and separate from ``nvbench_quality`` so the "is this an
identifier" question is independent of "is this usage of it allowed".

A field name alone is never sufficient evidence (per project policy): a name
match without supporting cardinality/uniqueness evidence is only ever
``confidence="ambiguous"``, never ``"strong"``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

RULE_VERSION = "nvbench_identifier_v1"

_DEFAULT_NAME_PATTERNS = (
    r"(^|_)id$",
    r"^id(_|$)",
    r"identifier",
    r"(^|_)key$",
    r"(^|_)code$",
)


def _name_pattern_re(cfg: Dict[str, Any]) -> re.Pattern:
    patterns = (cfg.get("identifier", {}) or {}).get("name_patterns") or _DEFAULT_NAME_PATTERNS
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


def detect_identifier(profile: Dict[str, Any], field_name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Evidence-based identifier decision for one physical field.

    ``profile`` is a ``FieldProfile`` dict from ``nvbench_profile.DbProfiler``.
    A ``resolution == "ambiguous_table"`` profile (field name exists in more
    than one table with no source signal to disambiguate) can never confirm or
    deny identifier status from statistics -- it is reported as ambiguous with
    an explicit note, and callers must treat table ambiguity as its own
    mandatory Tier-A blocker (see ``nvbench_quality.py``), independent of this
    identifier decision.
    """
    icfg = cfg.get("identifier", {}) or {}
    strong_unique_ratio = float(icfg.get("strong_unique_ratio", 0.98))
    strong_min_distinct = int(icfg.get("strong_min_distinct", 20))
    ambiguous_unique_ratio = float(icfg.get("ambiguous_unique_ratio", 0.5))
    name_re = _name_pattern_re(cfg)

    evidence: List[str] = []
    strong = False

    if profile.get("resolution") == "ambiguous_table":
        evidence.append("field_table_ambiguous")

    if profile.get("is_primary_key"):
        strong = True
        evidence.append("primary_key")
    if profile.get("is_unique_index"):
        strong = True
        evidence.append("unique_index")

    stats_available = bool(profile.get("stats_available"))
    unique_ratio = profile.get("unique_ratio")
    distinct_count = profile.get("distinct_count")
    if (
        stats_available
        and unique_ratio is not None
        and distinct_count is not None
        and unique_ratio >= strong_unique_ratio
        and distinct_count >= strong_min_distinct
    ):
        strong = True
        evidence.append(f"unique_ratio={unique_ratio:.3f}(n_distinct={distinct_count})")

    name_match = bool(name_re.search(field_name or ""))
    if name_match:
        evidence.append("name_pattern")

    if strong:
        return {
            "is_identifier": True,
            "confidence": "strong",
            "evidence": evidence,
            "rule_version": RULE_VERSION,
        }

    if name_match:
        if stats_available and unique_ratio is not None and unique_ratio >= ambiguous_unique_ratio:
            return {
                "is_identifier": True,
                "confidence": "ambiguous",
                "evidence": evidence,
                "rule_version": RULE_VERSION,
            }
        if not stats_available:
            return {
                "is_identifier": True,
                "confidence": "ambiguous",
                "evidence": evidence + ["stats_unavailable"],
                "rule_version": RULE_VERSION,
            }

    return {
        "is_identifier": False,
        "confidence": "none",
        "evidence": evidence,
        "rule_version": RULE_VERSION,
    }
