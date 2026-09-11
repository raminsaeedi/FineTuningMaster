"""Export one study as a static rating website and import the ratings back.

The deployed site is plain HTML/CSS/JS with the study content baked in, so it
can be served by any static host (GitHub Pages, Netlify, a university web
space) without a server-side runtime.

Two properties of the local Streamlit app are preserved:

* **Blindness.** Nothing published to the web carries a method, model, seed or
  item identifier. Every rating unit is addressed by an opaque token, and the
  token to (unit, method) mapping stays in the study directory, next to the
  assignment, and is never copied into the deployed bundle.
* **Contract.** Imported ratings are written in exactly the schema that
  ``storage.append_rating`` produces, so ``compute_irr.py`` validates and
  analyses web-collected ratings with no special case.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import secrets
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.evaluation.human.pipeline import (
    HumanEvaluationError,
    load_study,
    verify_source_predictions_unchanged,
)
from src.evaluation.human.render import render_brief_html, render_output_html
from src.evaluation.human.rubric import LIKERT_MAX, LIKERT_MIN, RUBRIC, RUBRIC_KEYS, rubric_hash

APP_VERSION = "human-eval-web-v1"
EXPORT_SCHEMA_VERSION = "human-eval-web-export-v1"
TEMPLATE_DIR = Path(__file__).resolve().parent / "webapp_template"
SITE_FILES = ("index.html", "styles.css", "app.js")
KEY_FILENAME = "unblinding_key.json"

# Strings that would tell a rater which system produced an output. The deployed
# bundle is searched for them before it is considered publishable.
LEAK_PATTERNS = (
    r"prompt[_\- ]only",
    r"ft[_\- ]rag",
    r"qlora",
    r"fine[_\- ]?tun",
    r"method_name",
    r"unblind",
)


class WebExportError(HumanEvaluationError):
    """Raised when a study cannot be turned into a publishable bundle."""


def _token(salt: str, kind: str, value: str, length: int = 12) -> str:
    digest = hashlib.sha256(f"{salt}|{kind}|{value}".encode("utf-8")).hexdigest()
    return digest[:length]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _presentation_order(unit_ids: Sequence[str], unit_to_item: Mapping[str, str], seed: int) -> list[str]:
    """Shuffle a rater's units, avoiding two outputs for the same brief in a row.

    Consecutive repetitions of a brief invite direct comparison between two
    systems, which the blind design is meant to prevent.
    """
    rng = random.Random(seed)
    shuffled = list(unit_ids)
    rng.shuffle(shuffled)
    by_item: dict[str, list[str]] = defaultdict(list)
    for unit_id in shuffled:
        by_item[unit_to_item[unit_id]].append(unit_id)

    ordered: list[str] = []
    previous_item: str | None = None
    while any(by_item.values()):
        candidates = [item for item, units in by_item.items() if units and item != previous_item]
        if not candidates:
            # Only possible when one brief holds more than half of what is left.
            candidates = [item for item, units in by_item.items() if units]
        chosen = max(candidates, key=lambda item: (len(by_item[item]), item))
        ordered.append(by_item[chosen].pop())
        previous_item = chosen
    return ordered


def _check_no_leaks(payload_text: str) -> list[str]:
    found = []
    for pattern in LEAK_PATTERNS:
        if re.search(pattern, payload_text, flags=re.IGNORECASE):
            found.append(pattern)
    return found


def build_web_bundle(
    *,
    study_dir: Path,
    project_root: Path,
    out_dir: Path | None = None,
    endpoint: str = "",
    contact_email: str = "",
    base_url: str = "",
    salt: str | None = None,
    presentation_seed: int = 20250911,
    verify_sources: bool = True,
) -> dict[str, Any]:
    """Write a deployable static site plus the local unblinding key."""
    manifest, items, assignment = load_study(study_dir)
    if verify_sources:
        verify_source_predictions_unchanged(manifest, project_root=project_root)

    web_dir = out_dir or (study_dir / "web")
    site_dir = web_dir / "site"
    salt = salt or secrets.token_hex(16)
    study_id = f"{manifest['dataset']}__{manifest['model']}__seed_{manifest['seed']}"
    export_id = _token(salt, "export", study_id, length=10)
    # Chapter 6.8 hides model identity and seed from raters, so the published
    # payload carries an opaque study handle and the real one stays in the key.
    public_study_id = "study_" + export_id

    items_by_id = {item["item_id"]: item for item in items}
    item_tokens = {item_id: "i_" + _token(salt, "item", item_id) for item_id in items_by_id}

    unit_records: dict[str, dict[str, Any]] = {}
    for rater, tasks in (assignment.get("raters") or {}).items():
        for task in tasks:
            unit_id = str(task["unit_id"])
            unit_records[unit_id] = {
                "item_id": str(task["item_id"]),
                "method": str(task["method"]),
            }

    unit_tokens = {unit_id: "u_" + _token(salt, "unit", unit_id) for unit_id in unit_records}
    if len(set(unit_tokens.values())) != len(unit_tokens):
        raise WebExportError("Token collision while blinding rating units; rerun the export.")
    if len(set(item_tokens.values())) != len(item_tokens):
        raise WebExportError("Token collision while blinding items; rerun the export.")

    site_items: dict[str, Any] = {}
    for item_id, item in items_by_id.items():
        site_items[item_tokens[item_id]] = {"brief_html": render_brief_html(item["brief"])}

    site_units: dict[str, Any] = {}
    for unit_id, record in unit_records.items():
        item = items_by_id.get(record["item_id"])
        if item is None:
            raise WebExportError(f"Assignment references unknown item {record['item_id']!r}.")
        output = (item.get("outputs") or {}).get(record["method"])
        if output is None:
            raise WebExportError(
                f"Item {record['item_id']!r} has no output for one of its assigned systems."
            )
        site_units[unit_tokens[unit_id]] = {
            "item": item_tokens[record["item_id"]],
            "output_html": render_output_html(output),
        }

    site_raters: dict[str, Any] = {}
    key_raters: dict[str, Any] = {}
    for index, (rater, tasks) in enumerate(sorted((assignment.get("raters") or {}).items())):
        unit_ids = [str(task["unit_id"]) for task in tasks]
        unit_to_item = {unit_id: unit_records[unit_id]["item_id"] for unit_id in unit_ids}
        ordered = _presentation_order(unit_ids, unit_to_item, presentation_seed + index)
        rater_token = _token(salt, "rater", rater, length=8)
        site_raters[rater] = {
            "token": rater_token,
            "units": [unit_tokens[unit_id] for unit_id in ordered],
        }
        key_raters[rater] = {"token": rater_token, "n_units": len(ordered)}

    study_payload = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "app_version": APP_VERSION,
        "study_id": public_study_id,
        "export_id": export_id,
        "rubric_version": manifest.get("rubric_version"),
        "likert": {"min": LIKERT_MIN, "max": LIKERT_MAX},
        "rubric": [
            {
                "key": dimension["key"],
                "label": dimension["label"],
                "description": dimension["description"],
                "anchors": {str(level): text for level, text in dimension["anchors"].items()},
            }
            for dimension in RUBRIC
        ],
        "items": site_items,
        "units": site_units,
        "raters": site_raters,
    }

    study_js = "window.HEVAL_STUDY = " + json.dumps(study_payload, ensure_ascii=False) + ";\n"
    config_js = (
        "window.HEVAL_CONFIG = "
        + json.dumps(
            {"endpoint": endpoint, "contact_email": contact_email, "app_version": APP_VERSION},
            ensure_ascii=False,
        )
        + ";\n"
    )

    leaks = _check_no_leaks(study_js)
    if leaks:
        raise WebExportError(
            "Refusing to write the bundle: blinding check matched "
            + ", ".join(leaks)
            + ". Inspect the rendered outputs before publishing."
        )

    site_dir.mkdir(parents=True, exist_ok=True)
    for filename in SITE_FILES:
        source = TEMPLATE_DIR / filename
        if not source.exists():
            raise WebExportError(f"Web template file is missing: {source}")
        shutil.copyfile(source, site_dir / filename)
    (site_dir / "study-data.js").write_text(study_js, encoding="utf-8")
    (site_dir / "config.js").write_text(config_js, encoding="utf-8")
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")
    shutil.copyfile(TEMPLATE_DIR / "apps_script" / "Code.gs", web_dir / "Code.gs")

    key_payload = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "study_id": study_id,
        "public_study_id": public_study_id,
        "export_id": export_id,
        "study_dir": str(study_dir.resolve()),
        "dataset": manifest.get("dataset"),
        "model": manifest.get("model"),
        "seed": manifest.get("seed"),
        "study_type": manifest.get("study_type"),
        "rubric_hash": rubric_hash(),
        "salt": salt,
        "presentation_seed": presentation_seed,
        "endpoint": endpoint,
        "raters": key_raters,
        "items": {token: item_id for item_id, token in item_tokens.items()},
        "units": {
            unit_tokens[unit_id]: {
                "unit_id": unit_id,
                "item_id": record["item_id"],
                "method": record["method"],
            }
            for unit_id, record in unit_records.items()
        },
    }
    (web_dir / KEY_FILENAME).write_text(
        json.dumps(key_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    links = _rater_links(base_url, site_raters)
    (web_dir / "rater_links.md").write_text(_links_markdown(study_id, export_id, links), encoding="utf-8")

    return {
        "web_dir": web_dir,
        "site_dir": site_dir,
        "key_path": web_dir / KEY_FILENAME,
        "links_path": web_dir / "rater_links.md",
        "study_id": study_id,
        "public_study_id": public_study_id,
        "export_id": export_id,
        "n_units": len(site_units),
        "n_items": len(site_items),
        "n_raters": len(site_raters),
        "endpoint": endpoint,
        "links": links,
        "bundle_bytes": sum(path.stat().st_size for path in site_dir.glob("*")),
    }


def _rater_links(base_url: str, raters: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    base = (base_url or "<YOUR-SITE-URL>").rstrip("/")
    return {
        rater: f"{base}/?r={rater}&t={payload['token']}"
        for rater, payload in sorted(raters.items())
    }


def _links_markdown(study_id: str, export_id: str, links: Mapping[str, str]) -> str:
    lines = [
        "# Personal rating links",
        "",
        f"Study: `{study_id}`  ",
        f"Export: `{export_id}`",
        "",
        "Send each rater exactly one link. The link carries their rater ID, so nobody has to",
        "choose an ID and two people cannot rate under the same one by accident.",
        "",
        "| Rater | Link |",
        "|---|---|",
    ]
    for rater, url in links.items():
        lines.append(f"| {rater} | {url} |")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------- import


def _coerce_score(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
    return value


def _scores_from_mapping(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("scores_json") or row.get("scores")
    scores: dict[str, Any] = {}
    if isinstance(raw, Mapping):
        scores = {key: _coerce_score(value) for key, value in raw.items()}
    elif isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, Mapping):
            scores = {key: _coerce_score(value) for key, value in parsed.items()}
    if not scores:
        scores = {
            dimension: _coerce_score(row[dimension])
            for dimension in RUBRIC_KEYS
            if row.get(dimension) not in (None, "")
        }
    return scores


def _records_from_csv(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    records = []
    for row in reader:
        cleaned = {key.strip(): value for key, value in row.items() if key}
        records.append(
            {
                "rater_id": (cleaned.get("rater_id") or "").strip(),
                "unit_token": (cleaned.get("unit_token") or "").strip(),
                "rating_id": (cleaned.get("rating_id") or "").strip(),
                "scores": _scores_from_mapping(cleaned),
                "comment": cleaned.get("comment") or "",
                "client_timestamp": (cleaned.get("client_timestamp") or "").strip(),
                "received_at": (cleaned.get("received_at") or "").strip(),
                "duration_ms": cleaned.get("duration_ms") or "",
                "source": str(path),
            }
        )
    return records


def _records_from_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        rater_id = str(payload.get("rater_id", ""))
        raw_ratings = payload.get("ratings") or []
    elif isinstance(payload, list):
        rater_id = ""
        raw_ratings = payload
    else:
        raise WebExportError(f"Unsupported rating file: {path}")
    records = []
    for rating in raw_ratings:
        if not isinstance(rating, Mapping):
            continue
        records.append(
            {
                "rater_id": str(rating.get("rater_id") or rater_id).strip(),
                "unit_token": str(rating.get("unit_token") or "").strip(),
                "rating_id": str(rating.get("rating_id") or "").strip(),
                "scores": _scores_from_mapping(rating),
                "comment": rating.get("comment") or "",
                "client_timestamp": str(rating.get("client_timestamp") or "").strip(),
                "received_at": str(rating.get("received_at") or "").strip(),
                "duration_ms": rating.get("duration_ms", ""),
                "source": str(path),
            }
        )
    return records


def _collect_inputs(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if path.is_dir():
            children = sorted(
                child for child in path.iterdir() if child.suffix.lower() in {".csv", ".json"}
            )
            records.extend(_collect_inputs(children))
            continue
        if not path.exists():
            raise WebExportError(f"Rating input not found: {path}")
        if path.suffix.lower() == ".csv":
            records.extend(_records_from_csv(path))
        elif path.suffix.lower() == ".json":
            records.extend(_records_from_json(path))
        else:
            raise WebExportError(f"Unsupported rating input (expected .csv or .json): {path}")
    return records


def _sort_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return (str(record.get("client_timestamp") or ""), str(record.get("received_at") or ""))


def _existing_rows(ratings_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if not ratings_dir.exists():
        return existing
    for path in sorted(ratings_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, Mapping) and row.get("rater_id") and row.get("unit_id"):
                existing[(str(row["rater_id"]), str(row["unit_id"]))] = dict(row)
    return existing


def import_web_ratings(
    *,
    study_dir: Path,
    inputs: Sequence[Path],
    key_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Map web ratings back onto the study and write rater JSONL files."""
    manifest, _items, assignment = load_study(study_dir)
    key_file = key_path or (study_dir / "web" / KEY_FILENAME)
    if not key_file.exists():
        raise WebExportError(
            f"Unblinding key not found: {key_file}. Export the web bundle before importing."
        )
    key = json.loads(key_file.read_text(encoding="utf-8"))
    if key.get("rubric_hash") != rubric_hash():
        raise WebExportError("The rubric changed after this web bundle was exported; ratings are not comparable.")

    units_by_token = key.get("units") or {}
    assigned: dict[str, set[str]] = defaultdict(set)
    for rater, tasks in (assignment.get("raters") or {}).items():
        for task in tasks:
            assigned[str(rater)].add(str(task["unit_id"]))

    records = _collect_inputs([Path(path) for path in inputs])
    accepted: dict[tuple[str, str], dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    duplicates = 0
    superseded = 0

    for record in sorted(records, key=_sort_key):
        rater_id = record["rater_id"]
        token = record["unit_token"]
        problem = None
        unit = units_by_token.get(token)
        if not rater_id:
            problem = "missing rater_id"
        elif rater_id not in assigned:
            problem = f"unknown rater_id {rater_id!r}"
        elif not token:
            problem = "missing unit_token"
        elif unit is None:
            problem = f"unit_token {token!r} does not belong to this export"
        elif unit["unit_id"] not in assigned[rater_id]:
            problem = f"unit {unit['unit_id']!r} was not assigned to {rater_id}"
        else:
            scores = record["scores"]
            missing = sorted(set(RUBRIC_KEYS) - set(scores))
            unknown = sorted(set(scores) - set(RUBRIC_KEYS))
            invalid = {
                key_name: value
                for key_name, value in scores.items()
                if key_name in RUBRIC_KEYS
                and (isinstance(value, bool) or not isinstance(value, int) or not LIKERT_MIN <= value <= LIKERT_MAX)
            }
            if missing:
                problem = f"missing dimensions {missing}"
            elif unknown:
                problem = f"unknown dimensions {unknown}"
            elif invalid:
                problem = f"scores outside {LIKERT_MIN}-{LIKERT_MAX}: {invalid}"
        if problem:
            rejected.append(
                {
                    "rater_id": rater_id,
                    "unit_token": token,
                    "rating_id": record.get("rating_id", ""),
                    "source": record.get("source", ""),
                    "problem": problem,
                }
            )
            continue

        identity = (rater_id, unit["unit_id"])
        if identity in accepted:
            duplicates += 1
        row = {
            "rater_id": rater_id,
            "unit_id": unit["unit_id"],
            "item_id": unit["item_id"],
            "method": unit["method"],
            "scores": {dimension: int(record["scores"][dimension]) for dimension in RUBRIC_KEYS},
            "comment": str(record.get("comment") or ""),
            "timestamp": record.get("client_timestamp") or record.get("received_at") or _utc_now(),
            "source": "web",
            "rating_id": record.get("rating_id", ""),
            "duration_ms": record.get("duration_ms", ""),
        }
        accepted[identity] = row

    ratings_dir = study_dir / "ratings"
    existing = _existing_rows(ratings_dir)
    merged = dict(existing)
    for identity, row in accepted.items():
        previous = merged.get(identity)
        if previous is not None:
            superseded += 1
        merged[identity] = row

    by_rater: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (rater_id, _unit_id), row in merged.items():
        by_rater[rater_id].append(row)

    expected_per_rater = {rater: len(units) for rater, units in assigned.items()}
    written: dict[str, int] = {}
    if not dry_run:
        ratings_dir.mkdir(parents=True, exist_ok=True)
        for rater_id, rows in by_rater.items():
            rows.sort(key=lambda row: (str(row.get("timestamp") or ""), str(row["unit_id"])))
            path = ratings_dir / f"{rater_id}.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
            )
            written[rater_id] = len(rows)
    else:
        written = {rater_id: len(rows) for rater_id, rows in by_rater.items()}

    expected_total = int(manifest.get("total_expected_ratings", 0))
    report = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "imported_utc": _utc_now(),
        "study_dir": str(study_dir.resolve()),
        "study_id": key.get("study_id"),
        "export_id": key.get("export_id"),
        "inputs": [str(path) for path in inputs],
        "records_read": len(records),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "duplicate_submissions": duplicates,
        "superseded_existing": superseded,
        "rows_per_rater": dict(sorted(written.items())),
        "expected_per_rater": dict(sorted(expected_per_rater.items())),
        "total_rows": sum(written.values()),
        "expected_total": expected_total,
        "completion_percentage": round(100.0 * sum(written.values()) / expected_total, 2)
        if expected_total
        else 0.0,
        "rejected_rows": rejected[:200],
        "rejected_reasons": dict(Counter(entry["problem"].split(":")[0] for entry in rejected)),
        "dry_run": dry_run,
    }
    if not dry_run:
        (study_dir / "web").mkdir(parents=True, exist_ok=True)
        (study_dir / "web" / "import_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return report
