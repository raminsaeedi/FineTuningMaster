"""Before/after comparison of the nvBench staging pilots v2 vs v3.

Pilot v3 uses an entirely different, database-capped/near-duplicate-aware
sampler (:func:`select_pilot_v3`), so its item set largely does not overlap with
v2's chart-stratified sample. This report therefore compares both whole-corpus
correction metrics (raw-columns-vs-aggregates, grouping recovery, constraint
preservation, duplicate reduction, mandatory validation) AND any item_ids the two
pilots happen to share. Read-only w.r.t. both pilots; writes only under the v3
reports directory.

Usage:
    python experiments/scripts/compare_nvbench_pilot_v2_v3.py \
        --v2 data/staging/dashboard_v3/nvbench_pilot_v2 \
        --v3 data/staging/dashboard_v3/nvbench_pilot_v3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.frozen_validation import read_jsonl_strict  # noqa: E402
from src.data_pipeline.nvbench_extract import extract_nested  # noqa: E402
from src.utils.io import write_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Compare nvBench pilot v2 vs v3.")
    p.add_argument("--v2", default="data/staging/dashboard_v3/nvbench_pilot_v2")
    p.add_argument("--v3", default="data/staging/dashboard_v3/nvbench_pilot_v3")
    return p.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _index(pilot: Path):
    recs, _ = read_jsonl_strict(pilot / "accepted.jsonl")
    return {r["item_id"]: r for r in recs}


def _report_json(pilot: Path):
    p = pilot / "reports" / "validation_report.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _manifest(pilot: Path):
    p = pilot / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _dup_rows(pilot: Path) -> int:
    p = pilot / "reports" / "duplicate_report.jsonl"
    if not p.exists():
        return 0
    return sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())


def _dist(pilot: Path) -> dict:
    p = pilot / "distribution_report.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    out: dict = {}
    for row in data.get("distributions", []):
        out.setdefault(row["dimension"], {})[row["value"]] = row["count"]
    return out


def _axis_view(rec: dict):
    prov = (rec.get("brief") or {}).get("extra", {}).get("provenance", {})
    at = prov.get("axis_typing")
    if at:
        x = (at["x"].get("dtype"), at["x"].get("role"), at["x"].get("name"))
        y = (at["y"].get("dtype"), at["y"].get("role"), at["y"].get("name"))
    else:
        cols = (rec.get("brief") or {}).get("columns", [])
        x = (cols[0].get("dtype"), cols[0].get("role", "?"), cols[0].get("name")) if cols else (None, None, None)
        y = (cols[1].get("dtype"), cols[1].get("role", "?"), cols[1].get("name")) if len(cols) > 1 else (None, None, None)
    maps = (rec.get("recommendation") or {}).get("kpi_chart_mapping") or [{}]
    return x, y, maps[0].get("kpi")


def _corpus_metrics(index: dict) -> dict:
    """Whole-pilot correction metrics, computed the same way for v2 and v3."""
    n = len(index)
    agg_in_columns = 0
    stacked_bar_total = 0
    stacked_bar_grouped = 0
    categorical_scatter = 0
    nested_remaining = 0
    with_filters = with_sort = with_time_grain = with_grouping = 0

    for rec in index.values():
        brief = rec.get("brief") or {}
        m = ((rec.get("recommendation") or {}).get("kpi_chart_mapping") or [{}])[0]
        enc = m.get("encoding") or {}
        prov = (brief.get("extra") or {}).get("provenance") or {}
        col_names = [str(c.get("name")) for c in (brief.get("columns") or [])]

        if any("(" in c and ")" in c for c in col_names):
            agg_in_columns += 1
        if m.get("chart_type") == "stacked_bar":
            stacked_bar_total += 1
            if enc.get("grouped") and enc.get("group_field"):
                stacked_bar_grouped += 1
        if m.get("chart_type") == "scatter":
            at = prov.get("axis_typing") or {}
            if any((at.get(a) or {}).get("dtype") == "categorical" for a in ("x", "y")):
                categorical_scatter += 1
        if any(extract_nested(str(enc.get(a) or "")) for a in ("x", "y")):
            nested_remaining += 1

        constraints = prov.get("constraints") or {}
        if constraints.get("filters"):
            with_filters += 1
        if constraints.get("sort"):
            with_sort += 1
        if constraints.get("time_grain"):
            with_time_grain += 1
        if (prov.get("grouping") or {}).get("is_grouped"):
            with_grouping += 1

    return {
        "n_accepted": n,
        "records_with_aggregate_in_columns": agg_in_columns,
        "stacked_bar_total": stacked_bar_total,
        "stacked_bar_with_valid_group_field": stacked_bar_grouped,
        "categorical_scatter_records": categorical_scatter,
        "records_with_nested_aggregate_remaining": nested_remaining,
        "records_with_filters": with_filters,
        "records_with_sort": with_sort,
        "records_with_time_grain": with_time_grain,
        "records_grouped": with_grouping,
    }


def main():
    args = parse_args()
    v2, v3 = _resolve(args.v2), _resolve(args.v3)
    a, b = _index(v2), _index(v3)
    common = sorted(set(a) & set(b))

    changed = []
    for iid in common:
        xa, ya, ka = _axis_view(a[iid])
        xb, yb, kb = _axis_view(b[iid])
        if (xa, ya, ka) != (xb, yb, kb):
            changed.append({
                "item_id": iid,
                "old_x": {"dtype": xa[0], "role": xa[1], "name": xa[2]},
                "new_x": {"dtype": xb[0], "role": xb[1], "name": xb[2]},
                "old_y": {"dtype": ya[0], "role": ya[1], "name": ya[2]},
                "new_y": {"dtype": yb[0], "role": yb[1], "name": yb[2]},
                "old_kpi": ka, "new_kpi": kb,
            })

    r2, r3 = _report_json(v2), _report_json(v3)
    m2, m3 = _manifest(v2), _manifest(v3)

    def mandatory(rep):
        fm = rep.get("failed_mandatory", [])
        return {"passed": rep.get("passed"), "failed_mandatory": [c.get("check") for c in fm],
                "byte_identical": (rep.get("determinism") or {}).get("byte_identical")}

    report = {
        "v2_dir": v2.as_posix(), "v3_dir": v3.as_posix(),
        "n_common_items": len(common),
        "n_changed_items_among_common": len(changed),
        "changed_items": changed,
        "corpus_metrics": {"v2": _corpus_metrics(a), "v3": _corpus_metrics(b)},
        "rejection_reasons": {"v2": m2.get("rejection_reasons", {}), "v3": m3.get("rejection_reasons", {})},
        "sampling_fallback_v3": m3.get("sampling_fallback"),
        "duplicate_report_rows": {"v2_total": _dup_rows(v2), "v3_total": _dup_rows(v3)},
        "chart_distribution": {"v2": _dist(v2).get("normalized_chart_type", {}),
                               "v3": _dist(v3).get("normalized_chart_type", {})},
        "database_distribution": {"v2": _dist(v2).get("database", {}), "v3": _dist(v3).get("database", {})},
        "mandatory_validation": {"v2": mandatory(r2), "v3": mandatory(r3)},
    }

    out_dir = v3 / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(report, out_dir / "before_after_v2_v3.json")

    cm2, cm3 = report["corpus_metrics"]["v2"], report["corpus_metrics"]["v3"]
    L = ["# nvBench Pilot v2 -> v3 — Before/After Report", ""]
    L.append(f"- v2: `{report['v2_dir']}`  (immutable diagnostic)")
    L.append(f"- v3: `{report['v3_dir']}`")
    L.append(f"- common item_ids: {report['n_common_items']} (v3 uses a different sampler; "
             "overlap is expected to be small)")
    L.append("")
    L.append("## Corpus-wide correction metrics (v2 -> v3)")
    for key in ("n_accepted", "records_with_aggregate_in_columns", "stacked_bar_total",
                "stacked_bar_with_valid_group_field", "categorical_scatter_records",
                "records_with_nested_aggregate_remaining", "records_with_filters",
                "records_with_sort", "records_with_time_grain", "records_grouped"):
        L.append(f"- `{key}`: {cm2[key]} -> {cm3[key]}")
    L.append("")
    L.append("## Rejection reasons")
    L.append(f"- v2: {report['rejection_reasons']['v2']}")
    L.append(f"- v3: {report['rejection_reasons']['v3']}")
    fb = report["sampling_fallback_v3"]
    if fb:
        L.append("")
        L.append(f"## v3 sampling fallback: {len(fb.get('fallbacks', []))} admissions over the database cap")
    dr = report["duplicate_report_rows"]
    L.append("")
    L.append(f"## Duplicate report rows: v2={dr['v2_total']} -> v3={dr['v3_total']}")
    L.append("")
    L.append("## Chart distribution")
    L.append(f"- v2: {report['chart_distribution']['v2']}")
    L.append(f"- v3: {report['chart_distribution']['v3']}")
    mv = report["mandatory_validation"]
    L.append("")
    L.append(f"## Mandatory validation: v2 passed={mv['v2']['passed']}; v3 passed={mv['v3']['passed']}")
    if changed:
        L.append("")
        L.append("## Changed items among the common set")
        for c in changed:
            L.append(f"- `{c['item_id']}`: x {c['old_x']} -> {c['new_x']}; y {c['old_y']} -> {c['new_y']}; "
                     f"KPI `{c['old_kpi']}` -> `{c['new_kpi']}`")
    (out_dir / "before_after_v2_v3.md").write_text("\n".join(L), encoding="utf-8")

    print(f"common_items={len(common)} changed_among_common={len(changed)}")
    print(f"v2 corpus: {cm2}")
    print(f"v3 corpus: {cm3}")
    print(f"wrote {out_dir / 'before_after_v2_v3.json'}")


if __name__ == "__main__":
    main()
