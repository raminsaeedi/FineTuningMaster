"""Render a brief and a system output for the rating UI.

Markdown is used by the local Streamlit app; the HTML variants are used by the
static web app that is deployed for remote raters. Both read the same fields so
the two front-ends show the same content.

If the output parsed into the schema, it is shown as readable sections/tables;
otherwise the raw model text is shown verbatim (raters still judge whatever the
system produced). The method identity is never included here.
"""

from __future__ import annotations

from html import escape
from typing import Any, Dict, List, Optional


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


def _esc(value: Any) -> str:
    return escape(str(value), quote=True)


def _scalar_html(value: Any) -> str:
    if value is None:
        return "<em>none</em>"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return _esc(value)


def _value_html(value: Any, depth: int = 0) -> str:
    """Render an arbitrary parsed value, escaping every model-produced string."""
    if depth >= 4:
        return _scalar_html(value)
    if isinstance(value, dict):
        if not value:
            return "<em>empty</em>"
        rows = "".join(
            f"<li><span class='k'>{_esc(key)}</span>: {_value_html(inner, depth + 1)}</li>"
            for key, inner in value.items()
        )
        return f"<ul class='kv'>{rows}</ul>"
    if isinstance(value, (list, tuple)):
        if not value:
            return "<em>empty</em>"
        rows = "".join(f"<li>{_value_html(inner, depth + 1)}</li>" for inner in value)
        return f"<ul class='items'>{rows}</ul>"
    return _scalar_html(value)


def render_brief_html(brief: Dict[str, Any]) -> str:
    """Return the dashboard brief as escaped HTML."""
    goals = [str(goal) for goal in (brief.get("goals") or [])]
    kpis = [str(kpi) for kpi in (brief.get("kpis") or [])]
    columns = brief.get("columns") or []
    parts: List[str] = ["<dl class='brief'>"]
    parts.append(f"<dt>Users</dt><dd>{_scalar_html(brief.get('users') or 'N/A')}</dd>")
    parts.append(
        "<dt>Goals</dt><dd>"
        + ("<ul>" + "".join(f"<li>{_esc(goal)}</li>" for goal in goals) + "</ul>" if goals else "N/A")
        + "</dd>"
    )
    parts.append(
        "<dt>KPIs</dt><dd>"
        + ("<ul>" + "".join(f"<li>{_esc(kpi)}</li>" for kpi in kpis) + "</ul>" if kpis else "N/A")
        + "</dd>"
    )
    if columns:
        cells = "".join(
            f"<tr><td>{_esc(column.get('name', ''))}</td><td>{_esc(column.get('dtype', ''))}</td>"
            f"<td>{_esc(column.get('role', '') or '')}</td></tr>"
            for column in columns
        )
        table = (
            "<table class='cols'><thead><tr><th>Column</th><th>Type</th><th>Role</th></tr></thead>"
            f"<tbody>{cells}</tbody></table>"
        )
    else:
        table = "N/A"
    parts.append(f"<dt>Data columns</dt><dd>{table}</dd>")
    parts.append(f"<dt>Constraints</dt><dd>{_scalar_html(brief.get('constraints') or 'None')}</dd>")
    parts.append("</dl>")
    return "".join(parts)


def render_output_html(output: Dict[str, Any]) -> str:
    """Return one system output as escaped HTML, without any method identity."""
    parsed: Optional[Dict[str, Any]] = output.get("parsed")
    if not parsed:
        raw = output.get("raw_text", "") or "(no output produced)"
        return (
            "<p class='note'>This output could not be parsed into the expected structure; "
            "the raw text produced by the system is shown.</p>"
            f"<pre class='raw'>{_esc(raw[:8000])}</pre>"
        )

    parts: List[str] = []
    context = parsed.get("context_summary")
    if context:
        parts.append("<h4>Context summary</h4>")
        parts.append(_value_html(context))

    mapping = parsed.get("kpi_chart_mapping") or []
    if mapping:
        parts.append("<h4>KPI to chart mapping</h4>")
        rows = []
        for entry in mapping:
            if not isinstance(entry, dict):
                rows.append(f"<tr><td colspan='4'>{_scalar_html(entry)}</td></tr>")
                continue
            alternatives = entry.get("alternatives") or []
            if isinstance(alternatives, (list, tuple)):
                alternatives_text = ", ".join(str(alternative) for alternative in alternatives)
            else:
                alternatives_text = str(alternatives)
            rows.append(
                "<tr>"
                f"<td>{_esc(entry.get('kpi', ''))}</td>"
                f"<td>{_esc(entry.get('task_type', ''))}</td>"
                f"<td><strong>{_esc(entry.get('chart_type', ''))}</strong></td>"
                f"<td>{_esc(alternatives_text)}</td>"
                "</tr>"
            )
        parts.append(
            "<table class='mapping'><thead><tr><th>KPI</th><th>Task</th><th>Chart</th>"
            f"<th>Alternatives</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )

    for section, title in (("layout", "Layout"), ("styling", "Styling")):
        value = parsed.get(section)
        if value:
            parts.append(f"<h4>{title}</h4>")
            parts.append(_value_html(value))

    interactions = parsed.get("interactions") or []
    if interactions:
        parts.append("<h4>Interactions</h4>")
        parts.append(_value_html(interactions))

    rationales = parsed.get("rationales") or []
    if rationales:
        parts.append("<h4>Rationales</h4>")
        entries = []
        for rationale in rationales:
            if isinstance(rationale, dict):
                claim = _esc(rationale.get("claim", ""))
                principle = _esc(rationale.get("principle", ""))
                entries.append(f"<li>{claim} <span class='principle'>{principle}</span></li>")
            else:
                entries.append(f"<li>{_scalar_html(rationale)}</li>")
        parts.append(f"<ul class='items'>{''.join(entries)}</ul>")

    if not parts:
        raw = output.get("raw_text", "") or "(no output produced)"
        return f"<pre class='raw'>{_esc(raw[:8000])}</pre>"
    return "".join(parts)
