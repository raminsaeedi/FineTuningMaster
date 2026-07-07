"""Source-conditioned synthetic generator for frozen dataset v2.

Separate from the legacy v1 generator (``synth_generator.py``). Every candidate
item is *conditioned on a documented source template* (a domain seed pool plus a
citation for the chart-task mapping it relies on) and carries that provenance in
``brief.extra`` so the frozen dataset card can trace where each item came from.

The chart-selection labels reuse the same principled, literature-backed
``task_type -> chart_type`` mapping as v1 (Cleveland & McGill 1984; Few 2006;
Munzner 2014) via :data:`TASK_CHART`, so labels stay valid and comparable.

Modes:
  ``sample``  — fully offline, deterministic generation. No network, no paid API.
                This is the only implemented mode and the default.
  ``api``     — reserved for future LLM-backed generation; **not implemented** and
                intentionally raises so no paid API is ever called by accident.

Everything in ``sample`` mode is deterministic per (seed, index): the same
arguments always produce byte-identical records, and growing ``n`` never changes
earlier items (mirrors the v1 determinism contract).
"""

from __future__ import annotations

import random
from typing import Dict, List

from src.data_pipeline.dataset import compute_item_id
from src.data_pipeline.synth_generator import (
    EXPERTISE,
    INDUSTRIES,
    KEYWORD_TASK,
    PRINCIPLE,
    TASK_CHART,
    UPDATE_FREQ,
)

# Bumped whenever the generation logic changes; recorded in every item's
# provenance and in generation_spec.yaml so frozen data is traceable.
GENERATOR_VERSION = "v2.0-sample"

VALID_MODES = ("sample", "api")

# Documented source template per domain. ``source_ref`` cites the basis for the
# chart-task mapping used to label the item; the domain seed pools come from the
# shared INDUSTRIES catalogue. These are *generic domain templates*, deliberately
# distinct instances from the independent evaluation briefs, so training data and
# eval gold never share lineage (enforced again by the leakage check at freeze).
SOURCE_REF = "Cleveland&McGill1984; Few2006; Munzner2014 (chart-task effectiveness)"


def _source_id(domain: str) -> str:
    slug = domain.lower().replace(" & ", "_").replace(" / ", "_").replace(" ", "_")
    return f"tmpl_{slug}"


def _task_for_kpi(kpi: str, rng: random.Random) -> str:
    low = kpi.lower()
    for keywords, task in KEYWORD_TASK:
        if any(k in low for k in keywords):
            return task
    return rng.choice(list(TASK_CHART))


def _columns_for(kpis: List[str]) -> List[Dict[str, str]]:
    cols: List[Dict[str, str]] = [
        {"name": "date", "dtype": "datetime"},
        {"name": "segment", "dtype": "categorical"},
        {"name": "region", "dtype": "categorical"},
    ]
    for k in kpis:
        cols.append({"name": k.lower().replace(" ", "_"), "dtype": "numeric"})
    return cols


def _mapping_for_kpi(kpi: str, rng: random.Random) -> dict:
    task = _task_for_kpi(kpi, rng)
    primary, alts = TASK_CHART[task]
    return {
        "kpi": kpi,
        "task_type": task,
        "chart_type": primary,
        "alternatives": list(alts),
        "encoding": {
            "x": "date" if task == "trend" else "segment",
            "y": kpi.lower().replace(" ", "_"),
        },
    }


def _build_brief(domain: str, rng: random.Random) -> dict:
    spec = INDUSTRIES[domain]
    audience = rng.choice(spec["audiences"])
    goals = rng.sample(spec["goals"], k=min(rng.randint(1, 2), len(spec["goals"])))
    kpis = rng.sample(spec["kpis"], k=min(rng.randint(3, 5), len(spec["kpis"])))
    expertise = rng.choice(EXPERTISE)
    freq = rng.choice(UPDATE_FREQ)
    return {
        "users": f"{audience} in the {domain} sector ({expertise} data literacy)",
        "goals": goals,
        "kpis": kpis,
        "columns": _columns_for(kpis),
        "constraints": f"Data refreshes {freq}; respect WCAG AA accessibility.",
        "extra": {
            "source_id": _source_id(domain),
            "source_ref": SOURCE_REF,
            "generator_version": GENERATOR_VERSION,
            "domain": domain,
            "data_literacy": expertise,
            "update_frequency": freq,
        },
    }


def _build_recommendation(brief: dict, rng: random.Random) -> dict:
    kpis = brief["kpis"]
    extra = brief["extra"]
    domain = extra["domain"]
    expertise = extra["data_literacy"]
    freq = extra["update_frequency"]
    # Same rng instance as the brief build → fully deterministic per (seed, index).
    mappings = [_mapping_for_kpi(k, rng) for k in kpis]

    rationales = [
        {"claim": f"Use a {m['chart_type']} chart for {m['kpi']}.", "principle": PRINCIPLE[m["task_type"]]}
        for m in mappings[:4]
    ]
    rationales.append({
        "claim": "Limit each view to 5-7 elements and group related KPIs.",
        "principle": "Managing cognitive load keeps the dashboard scannable (Few 2006).",
    })
    interactions = (
        ["cross-filtering across charts", "drill-down to detail", "export to PDF/Excel"]
        if expertise != "beginner"
        else ["date-range filter", "export to PDF"]
    )
    return {
        "context_summary": {
            "audience": brief["users"],
            "domain": domain,
            "primary_goal": brief["goals"][0] if brief["goals"] else "",
            "data_literacy": expertise,
            "update_frequency": freq,
        },
        "kpi_chart_mapping": mappings,
        "layout": {
            "pattern": "Headline KPI strip on top; supporting charts ordered by importance.",
            "primary_section": f"Headline metrics for {kpis[0]}"
            + (f" and {kpis[1]}" if len(kpis) > 1 else ""),
            "secondary_section": "Trend and comparison charts for the remaining KPIs",
            "responsive": expertise != "advanced",
        },
        "styling": {
            "color_palette": INDUSTRIES[domain]["palette"],
            "number_format": "percentages to 1 decimal; large values with K/M suffix",
            "accessibility": "maintain >= 4.5:1 contrast; never rely on color alone",
        },
        "interactions": interactions,
        "rationales": rationales,
    }


def _v2_item_id(brief: dict) -> str:
    """Stable ``v2_<md5-8>`` id derived from the brief content (reuses v1 hashing)."""
    return "v2_" + compute_item_id(brief).split("_", 1)[1]


def generate_candidates(n: int = 24, seed: int = 42, mode: str = "sample") -> List[dict]:
    """Generate ``n`` candidate ``{item_id, brief, recommendation}`` records.

    Deterministic per (seed, index). ``mode="sample"`` is fully offline. Any other
    supported mode (currently only ``"api"``) is not implemented and raises, so no
    paid API call can happen here yet.
    """
    if mode == "api":
        raise NotImplementedError(
            "API generation mode is not enabled. Use mode='sample' (offline). "
            "Paid API generation must be wired up explicitly before use."
        )
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {VALID_MODES}")

    domains = list(INDUSTRIES)
    items: List[dict] = []
    for i in range(n):
        rng = random.Random(seed * 100_003 + i)
        domain = domains[i % len(domains)]
        brief = _build_brief(domain, rng)
        recommendation = _build_recommendation(brief, rng)
        item_id = _v2_item_id(brief)
        brief_out = dict(brief)
        brief_out["item_id"] = item_id
        items.append({"item_id": item_id, "brief": brief_out, "recommendation": recommendation})
    return items
