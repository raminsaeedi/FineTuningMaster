"""Before/after comparison of two nvBench staging pilots (v1 vs v2).

Reads both pilots' accepted records and validation reports and writes a
before/after report to the v2 reports directory. Read-only w.r.t. both pilots.

Usage:
    python experiments/scripts/compare_nvbench_pilots.py \
        --v1 data/staging/dashboard_v3/nvbench_pilot_v1 \
        --v2 data/staging/dashboard_v3/nvbench_pilot_v2
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
from src.utils.io import write_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Compare nvBench pilot v1 vs v2.")
    p.add_argument("--v1", default="data/staging/dashboard_v3/nvbench_pilot_v1")
    p.add_argument("--v2", default="data/staging/dashboard_v3/nvbench_pilot_v2")
    return p.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _index(pilot: Path):
    recs, _ = read_jsonl_strict(pilot / "accepted.jsonl")
    return {r["item_id"]: r for r in recs}


def _axis_view(rec: dict):
    """(x_dtype, x_role, y_dtype, y_role, kpi) tolerant of old/new shapes."""
    prov = (rec.get("brief") or {}).get("extra", {}).get("provenance", {})
    at = prov.get("axis_typing")
    if at:
        x = (at["x"].get("dtype"), at["x"].get("role"))
        y = (at["y"].get("dtype"), at["y"].get("role"))
    else:
        cols = (rec.get("brief") or {}).get("columns", [])
        x = (cols[0].get("dtype"), cols[0].get("role", "?")) if len(cols) > 0 else (None, None)
        y = (cols[1].get("dtype"), cols[1].get("role", "?")) if len(cols) > 1 else (None, None)
    maps = (rec.get("recommendation") or {}).get("kpi_chart_mapping") or [{}]
    return x, y, maps[0].get("kpi")


def _report_json(pilot: Path):
    p = pilot / "reports" / "validation_report.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _dup_rows(pilot: Path) -> int:
    p = pilot / "reports" / "duplicate_report.jsonl"
    if not p.exists():
        return 0
    return sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())


def _dup_near_rows(pilot: Path) -> int:
    p = pilot / "reports" / "duplicate_report.jsonl"
    if not p.exists():
        return 0
    n = 0
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.strip() and json.loads(ln).get("type") == "near_duplicate_pair":
            n += 1
    return n


def main():
    args = parse_args()
    v1, v2 = _resolve(args.v1), _resolve(args.v2)
    a, b = _index(v1), _index(v2)
    common = sorted(set(a) & set(b))

    changed = []
    for iid in common:
        xa, ya, ka = _axis_view(a[iid])
        xb, yb, kb = _axis_view(b[iid])
        if (xa, ya, ka) != (xb, yb, kb):
            changed.append({
                "item_id": iid,
                "old_x": {"dtype": xa[0], "role": xa[1]}, "new_x": {"dtype": xb[0], "role": xb[1]},
                "old_y": {"dtype": ya[0], "role": ya[1]}, "new_y": {"dtype": yb[0], "role": yb[1]},
                "old_kpi": ka, "new_kpi": kb,
            })

    recovered_grouping, unresolved_grouping, categorical_scatter = [], [], []
    for iid, rec in b.items():
        prov = rec["brief"]["extra"]["provenance"]
        g = prov.get("grouping") or {}
        if g.get("series_field"):
            recovered_grouping.append({"item_id": iid, "series_field": g["series_field"]})
        elif g.get("is_grouped"):
            unresolved_grouping.append(iid)
        for w in prov.get("build_warnings") or []:
            if w.get("type") == "scatter_non_numeric_axis":
                categorical_scatter.append({"item_id": iid, "axis": w.get("axis"), "field": w.get("field")})

    r1, r2 = _report_json(v1), _report_json(v2)

    def mandatory(rep):
        out = {"passed": rep.get("passed")}
        fm = rep.get("failed_mandatory", [])
        out["failed_mandatory"] = [c.get("check") for c in fm]
        out["byte_identical"] = (rep.get("determinism") or {}).get("byte_identical")
        return out

    report = {
        "v1_dir": v1.as_posix(), "v2_dir": v2.as_posix(),
        "n_common_items": len(common),
        "n_changed_items": len(changed),
        "changed_items": changed,
        "recovered_grouping_fields": recovered_grouping,
        "unresolved_grouping_warnings": unresolved_grouping,
        "categorical_scatter_warnings": categorical_scatter,
        "duplicate_report_rows": {"v1_total": _dup_rows(v1), "v2_total": _dup_rows(v2),
                                  "v1_near_pairs": _dup_near_rows(v1), "v2_near_pairs": _dup_near_rows(v2)},
        "mandatory_validation": {"v1": mandatory(r1), "v2": mandatory(r2)},
    }

    out_dir = v2 / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(report, out_dir / "before_after_v1_v2.json")

    L = ["# nvBench Pilot v1 -> v2 — Before/After Report", ""]
    L.append(f"- v1: `{report['v1_dir']}`  (immutable diagnostic)")
    L.append(f"- v2: `{report['v2_dir']}`")
    L.append(f"- common items: {report['n_common_items']}  changed: {report['n_changed_items']}")
    L.append(f"- recovered grouping fields: {len(recovered_grouping)}")
    L.append(f"- unresolved grouping warnings: {len(unresolved_grouping)}")
    L.append(f"- categorical-scatter warnings: {len(categorical_scatter)}")
    dr = report["duplicate_report_rows"]
    L.append(f"- duplicate report rows: v1={dr['v1_total']} (near={dr['v1_near_pairs']}) "
             f"-> v2={dr['v2_total']} (near={dr['v2_near_pairs']})")
    mv = report["mandatory_validation"]
    L.append(f"- mandatory validation: v1 passed={mv['v1']['passed']} byte_identical={mv['v1']['byte_identical']}; "
             f"v2 passed={mv['v2']['passed']} byte_identical={mv['v2']['byte_identical']}")
    L.append("")
    L.append("## Changed items (old -> new x/y dtype,role and KPI)")
    for c in changed:
        L.append(f"- `{c['item_id']}`: "
                 f"x {c['old_x']['dtype']}/{c['old_x']['role']} -> {c['new_x']['dtype']}/{c['new_x']['role']}; "
                 f"y {c['old_y']['dtype']}/{c['old_y']['role']} -> {c['new_y']['dtype']}/{c['new_y']['role']}; "
                 f"KPI `{c['old_kpi']}` -> `{c['new_kpi']}`")
    if recovered_grouping:
        L.append("")
        L.append("## Recovered grouping fields")
        for g in recovered_grouping:
            L.append(f"- `{g['item_id']}`: {g['series_field']}")
    (out_dir / "before_after_v1_v2.md").write_text("\n".join(L), encoding="utf-8")

    print(f"changed={len(changed)}/{len(common)} recovered_grouping={len(recovered_grouping)} "
          f"unresolved_grouping={len(unresolved_grouping)} categorical_scatter={len(categorical_scatter)}")
    print(f"dup rows v1={dr['v1_total']} -> v2={dr['v2_total']}")
    print(f"wrote {out_dir / 'before_after_v1_v2.json'}")


if __name__ == "__main__":
    main()
