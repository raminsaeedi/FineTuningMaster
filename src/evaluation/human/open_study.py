"""Build the versioned, open-enrolment 27B study (private Apps Script bundle).

Never publish Code.gs or the unblinding key to a public repository.
The four fixed packets are a balanced incomplete-block allocation, not four
experimental groups: each participant sees every method on different briefs.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

from .pipeline import HumanEvaluationError, load_study, verify_source_predictions_unchanged
from .render import render_brief_html, render_output_html
from .rubric import RUBRIC_KEYS
from .web import _check_no_leaks

TEMPLATES = Path(__file__).parent / "open_web"
SCHEMA = "human-eval-open-v1"
RUBRIC_VERSION = "dashboard-plain-bilingual-v2"
CONSENT_VERSION = "open-de-en-2026-09-v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def build_open_study(*, source_dir: Path, out_dir: Path, project_root: Path,
                     spreadsheet_id: str, sheet_prefix: str = "openv1",
                     target_per_output: int = 3) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{20,100}", spreadsheet_id):
        raise HumanEvaluationError("Invalid spreadsheet ID")
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,20}", sheet_prefix):
        raise HumanEvaluationError("Invalid sheet prefix")
    if target_per_output != 3:
        raise HumanEvaluationError("Version 1 freezes the coverage target at 3")
    source, items, _ = load_study(source_dir)
    verify_source_predictions_unchanged(source, project_root=project_root)
    if source["model"] != "qwen3_8_27b" or source["methods"] != list("ABCD") or len(items) != 8:
        raise HumanEvaluationError("Open v1 requires eight frozen briefs and methods A-D of the 27B study")
    # Source study is authoritative. Do not reselect from a mutable sampling CSV.
    source_hashes = {name: digest(source_dir / name) for name in
                     ("study_manifest.json", "items.jsonl", "assignment.json")}
    template_hashes = {p.name: digest(p) for p in sorted(TEMPLATES.iterdir()) if p.is_file()}
    identity = dict(schema_version=SCHEMA, model=source["model"], dataset=source["dataset"],
                    seed=source["seed"], source_hashes=source_hashes, template_hashes=template_hashes,
                    spreadsheet_id=spreadsheet_id, sheet_prefix=sheet_prefix,
                    target_per_output=target_per_output, rubric_version=RUBRIC_VERSION,
                    consent_version=CONSENT_VERSION)
    fingerprint = hashlib.sha256(dumps(identity).encode()).hexdigest()
    manifest_path = out_dir / "open_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["build_fingerprint"] != fingerprint:
            raise HumanEvaluationError("Frozen open study differs. Use a new version/directory; do not overwrite collected study assets.")
        for name, expected in previous["artifact_hashes"].items():
            if digest(out_dir / name) != expected:
                raise HumanEvaluationError(f"Study artifact changed: {name}")
        return previous
    if out_dir.exists() and any(out_dir.iterdir()):
        raise HumanEvaluationError("Output directory is not empty")
    salt = secrets.token_hex(32)
    token = lambda value: "u_" + hashlib.sha256((salt + value).encode()).hexdigest()[:24]
    public_id = "open_" + secrets.token_hex(8)
    units, private_units, lookup = {}, {}, {}
    for item in items:
        for method in "ABCD":
            unit_token = token(item["item_id"] + "|" + method)
            lookup[(item["item_id"], method)] = unit_token
            units[unit_token] = dict(brief_html=render_brief_html(item["brief"]),
                                     output_html=render_output_html(item["outputs"][method]))
            private_units[unit_token] = dict(item_id=item["item_id"], method=method)
    if len(units) != 32 or _check_no_leaks(dumps(units)):
        raise HumanEvaluationError("Output token collision or method identity leak")
    rng = random.Random(42)
    item_order = [item["item_id"] for item in items]
    rng.shuffle(item_order)
    methods = list("ABCD")
    rng.shuffle(methods)
    packets = [[lookup[(item_id, methods[(i + shift) % 4])]
                for i, item_id in enumerate(item_order)] for shift in range(4)]
    server = dict(study_id=public_id, spreadsheet_id=spreadsheet_id, sheet_prefix=sheet_prefix,
                  dimensions=list(RUBRIC_KEYS), session_size=8, target_per_output=3,
                  consent_version=CONSENT_VERSION, rubric_version=RUBRIC_VERSION,
                  packets=packets, units=units)
    page = (TEMPLATES / "index.html").read_text(encoding="utf-8")
    page = page.replace("/*STYLE*/", (TEMPLATES / "style.css").read_text(encoding="utf-8"))
    page = page.replace("/*APP*/", (TEMPLATES / "app.js").read_text(encoding="utf-8"))
    code = (TEMPLATES / "collector.js").read_text(encoding="utf-8")
    code += "\nvar OPEN_STUDY = " + dumps(server) + ";\n"
    code += "var PAGE_HTML = " + json.dumps(page, ensure_ascii=False) + ";\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "Code.gs").write_text(code, encoding="utf-8")
    (out_dir / "page.html").write_text(page, encoding="utf-8")
    (out_dir / "server_config.json").write_text(dumps(server), encoding="utf-8")
    key = dict(study_id=public_id, rubric_version=RUBRIC_VERSION, units=private_units, packets=packets)
    (out_dir / "unblinding_key.json").write_text(dumps(key), encoding="utf-8")
    manifest = dict(identity, study_id=public_id, build_fingerprint=fingerprint,
                    created_utc=datetime.now(timezone.utc).isoformat(), source_dir=str(source_dir.resolve()),
                    n_items=8, n_outputs=32, session_size=8, item_ids=[x["item_id"] for x in items],
                    recruitment="Open convenience sample; score-blind coverage stopping; not a power calculation",
                    analysis="Exploratory descriptive; do not pool earlier protocols",
                    minimum_complete_sessions_for_target=12, max_active_minutes=30,
                    source_prediction_hashes=source["source_prediction_hashes"],
                    artifact_hashes={name: digest(out_dir / name) for name in
                                    ("Code.gs", "page.html", "server_config.json", "unblinding_key.json")})
    manifest_path.write_text(dumps(manifest), encoding="utf-8")
    return manifest
