"""Render a brief and a system output as Markdown for the rating UI.

If the output parsed into the schema, it is shown as readable sections/tables;
otherwise the raw model text is shown verbatim (raters still judge whatever the
system produced). The method identity is never included here.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def render_brief(brief: Dict[str, Any]) -> str:
    goals = ", ".join(brief.get("goals", []) or []) or "N/A"
    kpis = ", ".join(brief.get("kpis", []) or []) or "N/A"
    cols = ", ".join(
        f"{c.get('name')} ({c.get('dtype')})" for c in brief.get("columns", []) or []
    ) or "N/A"
    return (
        f"**Users:** {brief.get('users', 'N/A')}\n\n"
        f"**Goals:** {goals}\n\n"
        f"**KPIs:** {kpis}\n\n"
        f"**Data columns:** {cols}\n\n"
        f"**Constraints:** {brief.get('constraints') or 'None'}"
    )


def render_output(output: Dict[str, Any]) -> str:
    parsed: Optional[Dict[str, Any]] = output.get("parsed")
    if not parsed:
        raw = output.get("raw_text", "") or "(no output produced)"
        return "_Could not parse a structured recommendation; raw model output:_\n\n```\n" + raw[:4000] + "\n```"

    lines = []
    ctx = parsed.get("context_summary")
    if ctx:
        lines.append("**Context summary**")
        if isinstance(ctx, dict):
            for k, v in ctx.items():
                lines.append(f"- {k}: {v}")
        else:
            lines.append(str(ctx))
        lines.append("")

    mapping = parsed.get("kpi_chart_mapping") or []
    if mapping:
        lines.append("**KPI → chart mapping**\n")
        lines.append("| KPI | Task | Chart | Alternatives |")
        lines.append("|---|---|---|---|")
        for m in mapping:
            alts = ", ".join(m.get("alternatives", []) or [])
            lines.append(f"| {m.get('kpi','')} | {m.get('task_type','')} | {m.get('chart_type','')} | {alts} |")
        lines.append("")

    for section in ("layout", "styling"):
        sec = parsed.get(section)
        if sec:
            lines.append(f"**{section.capitalize()}**")
            if isinstance(sec, dict):
                for k, v in sec.items():
                    lines.append(f"- {k}: {v}")
            else:
                lines.append(str(sec))
            lines.append("")

    interactions = parsed.get("interactions") or []
    if interactions:
        lines.append("**Interactions**")
        for it in interactions:
            lines.append(f"- {it}")
        lines.append("")

    rationales = parsed.get("rationales") or []
    if rationales:
        lines.append("**Rationales**")
        for r in rationales:
            if isinstance(r, dict):
                lines.append(f"- {r.get('claim','')} — _{r.get('principle','')}_")
            else:
                lines.append(f"- {r}")
        lines.append("")

    return "\n".join(lines)
