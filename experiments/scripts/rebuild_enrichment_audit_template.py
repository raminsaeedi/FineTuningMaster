"""Rebuild the human-audit template from existing enrichment artifacts (offline).

Repairs a run whose audit CSV covered only the automatically accepted records:
human review needs every selected record, including automatically rejected ones.

Reads only files already on disk (``input_records.jsonl``, ``raw_responses.jsonl``,
``accepted_records.jsonl``, ``rejected_records.jsonl``), joins them by ``item_id``,
and rewrites the CSV in the original deterministic selection order. Makes **no API
request**, re-validates nothing and changes no model output. The previous CSV is
preserved next to the new one.

Usage:
    python experiments/scripts/rebuild_enrichment_audit_template.py
    python experiments/scripts/rebuild_enrichment_audit_template.py --run-dir data/staging/enrichment/pilot_30
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.scripts.run_enrichment_sample import (  # noqa: E402
    AUDIT_HUMAN_COLUMNS,
    AUDIT_VALIDATION_COLUMNS,
    build_audit_rows,
    _write_audit_template,
)
from src.utils.io import read_json, read_jsonl, write_json  # noqa: E402

DEFAULT_RUN_DIR = "data/staging/enrichment/pilot_30"
AUDIT_NAME = "manual_enrichment_audit_template_30.csv"
FINAL_STATUS = "AUDIT_TEMPLATE_FIXED_READY_FOR_HUMAN_R1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild the enrichment audit template offline")
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--audit-name", default=AUDIT_NAME)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = _PROJECT_ROOT / run_dir

    selected = read_jsonl(run_dir / "input_records.jsonl")
    raw_rows = read_jsonl(run_dir / "raw_responses.jsonl")
    accepted = read_jsonl(run_dir / "accepted_records.jsonl")
    rejected = read_jsonl(run_dir / "rejected_records.jsonl")

    audit_path = run_dir / args.audit_name
    rows = build_audit_rows(selected, accepted, rejected, raw_rows)

    print("=" * 60)
    print("ENRICHMENT AUDIT TEMPLATE - OFFLINE REBUILD (no API call)")
    print("=" * 60)
    print(f"  run dir                : {run_dir}")
    print(f"  selected pilot records : {len(selected)}")
    print(f"  automatically accepted : {len(accepted)}")
    print(f"  automatically rejected : {len(rejected)}")
    print(f"  human audit rows       : {len(rows)}")

    # Preserve the incomplete file before overwriting it.
    if audit_path.exists():
        with audit_path.open(encoding="utf-8", newline="") as f:
            previous_rows = list(csv.DictReader(f))
        if len(previous_rows) != len(rows):
            backup = audit_path.with_name(
                f"{audit_path.stem}_incomplete_{len(previous_rows)}rows{audit_path.suffix}")
            backup.write_bytes(audit_path.read_bytes())
            print(f"  previous file kept as  : {backup.name} ({len(previous_rows)} rows)")

    _write_audit_template(audit_path, rows)
    print(f"  written                : {audit_path.name}")

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        manifest.setdefault("outputs", {})["audit_template"] = audit_path.name
        manifest["audit_template"] = {
            "path": audit_path.name,
            "rows": len(rows),
            "selected_pilot_records": len(selected),
            "automatically_accepted": len(accepted),
            "automatically_rejected": len(rejected),
            "human_audit_rows": len(rows),
            "row_order": "deterministic selection order (input_records.jsonl)",
            "validation_columns": list(AUDIT_VALIDATION_COLUMNS),
            "human_columns_blank": list(AUDIT_HUMAN_COLUMNS),
            "rebuilt_offline": True,
            "api_calls_made": 0,
        }
        manifest["status"] = FINAL_STATUS
        write_json(manifest, manifest_path)
        print(f"  manifest updated       : {manifest_path.name}")

    report_path = run_dir / "report.md"
    if report_path.exists():
        report = report_path.read_text(encoding="utf-8").rstrip("\n")
        report += (
            "\n\n## Human audit template (rebuilt offline)\n\n"
            f"- selected pilot records: {len(selected)}\n"
            f"- automatically accepted: {len(accepted)}\n"
            f"- automatically rejected: {len(rejected)}\n"
            f"- human audit rows: {len(rows)}\n"
            "- rows follow the deterministic selection order, not acceptance status\n"
            "- automatically rejected records are included with their reason codes in "
            "`automatic_validation_status` / `automatic_reason_codes` / "
            "`automatic_validation_details`\n"
            "- all human-review columns are blank; no API call and no model output changed\n"
            f"- status: **{FINAL_STATUS}**\n"
        )
        report_path.write_text(report + "\n", encoding="utf-8")
        print(f"  report updated         : {report_path.name}")

    print(f"  status                 : {FINAL_STATUS}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
