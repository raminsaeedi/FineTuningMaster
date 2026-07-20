"""Registration and integrity tooling for manually downloaded external sources.

This module audits raw external corpora that were provisioned by hand under
``data/raw_external/`` (nvBench v1, nvBench 2.0, Quda). It records a per-file
SHA-256 inventory, sizes, source metadata and a license status established
*only* from the actual downloaded files.

Nothing here ingests data, extracts archives, or builds datasets. It is
read-only with respect to the raw files: it never opens a raw file for writing.

The registration outputs themselves (``source_manifest.json``,
``SHA256SUMS.txt``, ``source_inspection_report.json``) are excluded from the
hash inventory so they never appear in their own checksum listing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from src.data_pipeline.frozen_validation import sha256_of_file

# Registration artefacts must never be hashed into their own inventory.
REGISTRATION_OUTPUT_NAMES = frozenset(
    {"source_manifest.json", "SHA256SUMS.txt", "source_inspection_report.json"}
)

# License keyword -> canonical SPDX-ish label. Order matters (most specific first).
_LICENSE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"creative commons attribution[- ]sharealike\s*4\.0|cc[ -]by[ -]sa[ -]?4\.0", "CC-BY-SA-4.0"),
    (r"creative commons attribution\s*4\.0|cc[ -]by[ -]?4\.0", "CC-BY-4.0"),
    (r"\bmit license\b|opensource\.org/licenses/mit", "MIT"),
    (r"\bapache license\b|apache-2\.0", "Apache-2.0"),
    (r"\bbsd\b.{0,20}license", "BSD"),
    (r"\bgnu general public license\b|\bgpl-", "GPL"),
)

_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_SHA256SUMS_LINE_RE = re.compile(r"^([0-9a-f]{64})  (.+)$")


# --------------------------------------------------------------------------- #
# File inventory
# --------------------------------------------------------------------------- #
def walk_files(
    root: str | Path,
    exclude_names: frozenset = REGISTRATION_OUTPUT_NAMES,
) -> List[Dict[str, object]]:
    """Recursively inventory every file under ``root`` (read-only).

    Returns a list of ``{relative_path, size_bytes, sha256}`` records sorted by
    POSIX relative path. Files whose basename is in ``exclude_names`` are
    skipped so registration outputs never enter their own inventory.
    """
    root = Path(root)
    records: List[Dict[str, object]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name in exclude_names:
            continue
        rel = path.relative_to(root).as_posix()
        records.append(
            {
                "relative_path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_of_file(path),
            }
        )
    records.sort(key=lambda r: r["relative_path"])
    return records


def write_sha256sums(files: List[Dict[str, object]], out_path: str | Path) -> None:
    """Write a standard ``SHA256SUMS.txt`` (``<hex>  <relative_path>``), sorted."""
    out_path = Path(out_path)
    lines = [f"{f['sha256']}  {f['relative_path']}" for f in files]
    lines.sort(key=lambda ln: ln.split("  ", 1)[1])
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_sha256sums(path: str | Path) -> Tuple[Dict[str, str], List[str]]:
    """Parse a ``SHA256SUMS.txt`` into ``{relative_path: hex}`` and malformed lines."""
    path = Path(path)
    recorded: Dict[str, str] = {}
    malformed: List[str] = []
    if not path.exists():
        return recorded, malformed
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        m = _SHA256SUMS_LINE_RE.match(raw_line)
        if not m:
            malformed.append(raw_line)
            continue
        recorded[m.group(2)] = m.group(1)
    return recorded, malformed


# --------------------------------------------------------------------------- #
# Metadata detection (from on-disk content only)
# --------------------------------------------------------------------------- #
def _readme_texts(source_dir: Path) -> List[Tuple[Path, str]]:
    """Return (path, text) for every README/readme markdown under source_dir."""
    out: List[Tuple[Path, str]] = []
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and path.name.lower() in {"readme.md", "readme.txt", "readme"}:
            try:
                out.append((path, path.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                continue
    return out


def _match_license(text: str) -> Optional[str]:
    lowered = text.lower()
    for pattern, label in _LICENSE_PATTERNS:
        if re.search(pattern, lowered):
            return label
    return None


def detect_references(source_dir: str | Path) -> List[str]:
    """Collect external reference URLs scraped from README files (sorted, unique)."""
    source_dir = Path(source_dir)
    urls = set()
    for _path, text in _readme_texts(source_dir):
        for url in _URL_RE.findall(text):
            urls.add(url.rstrip(".,);"))
    return sorted(urls)


def detect_license(source_dir: str | Path) -> Dict[str, Optional[str]]:
    """Establish license status from on-disk files only.

    Status is one of ``confirmed`` | ``pending`` | ``unknown``:
    - A real LICENSE/COPYING file, or an *active* README license statement,
      yields ``confirmed`` with the detected label.
    - A license mention that survives *only* inside an HTML comment
      (``<!-- ... -->``) yields ``pending`` (not an active grant).
    - No license signal anywhere yields ``unknown``.
    """
    source_dir = Path(source_dir)

    # 1) Dedicated license file wins.
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and path.name.lower().split(".")[0] in {"license", "copying"}:
            label = _match_license(path.read_text(encoding="utf-8", errors="replace"))
            return {
                "status": "confirmed",
                "license": label or "present-unclassified",
                "evidence": f"License file: {path.relative_to(source_dir).as_posix()}",
            }

    # 2) README active vs. commented-out text.
    for path, text in _readme_texts(source_dir):
        active = _HTML_COMMENT_RE.sub("", text)
        active_label = _match_license(active)
        if active_label:
            return {
                "status": "confirmed",
                "license": active_label,
                "evidence": f"Active license statement in {path.relative_to(source_dir).as_posix()}",
            }
        raw_label = _match_license(text)
        if raw_label:
            return {
                "status": "pending",
                "license": None,
                "evidence": (
                    f"License ({raw_label}) mentioned only inside a commented-out "
                    f"section of {path.relative_to(source_dir).as_posix()}; not an active grant."
                ),
            }

    return {
        "status": "unknown",
        "license": None,
        "evidence": "No LICENSE file and no license statement found in downloaded files.",
    }


def detect_revision(
    source_dir: str | Path,
    files: List[Dict[str, object]],
) -> Dict[str, Optional[str]]:
    """Best-effort source revision from on-disk content.

    A GitHub branch archive folder (e.g. ``nvBench-main``) identifies the
    downloaded *ref*, never a pinned commit. The archive SHA-256 is recorded as
    the immutable local source identifier.
    """
    source_dir = Path(source_dir)
    archive_rel: Optional[str] = None
    archive_sha: Optional[str] = None
    for f in files:
        rel = str(f["relative_path"])
        if rel.startswith("archive/") and rel.endswith(".zip"):
            archive_rel = rel
            archive_sha = str(f["sha256"])
            break

    downloaded_ref: Optional[str] = None
    stem = Path(archive_rel).stem if archive_rel else ""
    for ref in ("main", "master"):
        if stem.endswith(f"-{ref}"):
            downloaded_ref = ref
            break

    return {
        "downloaded_ref": downloaded_ref,
        "pinned_commit": None,
        "archive_name": Path(archive_rel).name if archive_rel else None,
        "archive_sha256": archive_sha,
        "note": (
            "Downloaded as a GitHub branch archive; the '-<ref>' folder suffix "
            "identifies the branch, not a pinned commit. The archive SHA-256 is "
            "the immutable local source identifier."
        ),
    }


def nvbench_v1_stats(source_dir: str | Path) -> Optional[Dict[str, object]]:
    """Count nvBench v1 records, distinguishing the three different totals.

    - ``vis_object_count``: number of top-level visualization objects.
    - ``nl_queries_total``: sum of ``nl_queries`` across all objects.
    - ``published_nl_vis_pairs``: the figure stated in the README (25,750).

    Each visualization object holds multiple NL queries, so the top-level object
    count is expected to be smaller than the published NL-VIS pair count; they do
    not conflict.
    """
    source_dir = Path(source_dir)
    main_json: Optional[Path] = None
    for path in sorted(source_dir.rglob("*.json")):
        if path.name.lower() == "nvbench.json":
            main_json = path
            break
    if main_json is None:
        return None

    with main_json.open("r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    vis_object_count = len(data)
    nl_queries_total = 0
    for entry in data.values():
        if isinstance(entry, dict):
            nl_queries_total += len(entry.get("nl_queries", []) or [])

    return {
        "main_file": main_json.relative_to(source_dir).as_posix(),
        "vis_object_count": vis_object_count,
        "nl_queries_total": nl_queries_total,
        "published_nl_vis_pairs": 25750,
        "note": (
            "Each visualization object contains multiple nl_queries; the "
            f"{vis_object_count} top-level objects are not in conflict with the "
            "published 25,750 NL-VIS pairs."
        ),
    }


# --------------------------------------------------------------------------- #
# Manifest assembly
# --------------------------------------------------------------------------- #
def build_source_manifest(
    source_dir: str | Path,
    *,
    source_name: str,
    repository_url: Optional[str],
    timestamp: str,
    extra_references: Optional[List[str]] = None,
    license_override_to_pending: bool = False,
    stats_fn: Optional[Callable[[Path], Optional[Dict[str, object]]]] = None,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Assemble a source manifest plus the file inventory used to build it.

    Returns ``(manifest, files)`` so the caller can write ``SHA256SUMS.txt`` from
    the exact same inventory. ``license_override_to_pending`` downgrades a
    non-``confirmed`` license to ``pending`` (used for sources with no active
    grant, e.g. nvBench 2.0).
    """
    source_dir = Path(source_dir)
    files = walk_files(source_dir)
    license_info = detect_license(source_dir)
    if license_override_to_pending and license_info["status"] != "confirmed":
        license_info = {**license_info, "status": "pending"}

    references = set(detect_references(source_dir))
    if extra_references:
        references.update(extra_references)

    revision = detect_revision(source_dir, files)
    stats = stats_fn(source_dir) if stats_fn else None

    manifest: Dict[str, object] = {
        "source": source_name,
        "repository_url": repository_url,
        "detected_references": sorted(references),
        "source_revision": revision,
        "license": license_info,
        "dataset_stats": stats,
        "n_files": len(files),
        "total_bytes": sum(int(f["size_bytes"]) for f in files),
        "files": files,
        "registered_at": timestamp,
    }
    return manifest, files


def inspect_quda(
    quda_dir: str | Path,
    *,
    repository_url: Optional[str],
    timestamp: str,
) -> Dict[str, object]:
    """Inspect the Quda directory without assuming a corpus is present."""
    quda_dir = Path(quda_dir)
    files = walk_files(quda_dir)
    corpus_present = len(files) > 0

    return {
        "source": "quda",
        "repository_url": repository_url,
        "corpus_present": corpus_present,
        "n_files": len(files),
        "files": files,
        "query_count": None,
        "task_labels_present": False,
        "ids_present": False,
        "license": {
            "status": "pending",
            "license": None,
            "evidence": (
                "No corpus or license files present locally."
                if not corpus_present
                else "Files present but corpus/license not yet verified; kept pending."
            ),
        },
        "note": (
            "quda_pending is empty; the Quda corpus has not been provisioned. "
            "Registered as pending/absent. Ingestion is intentionally not implemented."
            if not corpus_present
            else "Files detected under quda_pending; verify corpus and license before ingestion."
        ),
        "registered_at": timestamp,
    }


# --------------------------------------------------------------------------- #
# Verification (strictly read-only)
# --------------------------------------------------------------------------- #
def verify_source(source_dir: str | Path) -> Dict[str, object]:
    """Re-hash a source and diff against its recorded manifest and checksums.

    Detects (without writing anything): missing files, added files, changed
    hashes, changed byte sizes, and malformed checksum entries.
    """
    source_dir = Path(source_dir)
    current = {str(f["relative_path"]): f for f in walk_files(source_dir)}

    recorded_hashes, malformed = parse_sha256sums(source_dir / "SHA256SUMS.txt")

    recorded_sizes: Dict[str, int] = {}
    manifest_path = source_dir / "source_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for f in manifest.get("files", []):
                recorded_sizes[str(f["relative_path"])] = int(f["size_bytes"])
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    current_paths = set(current)
    recorded_paths = set(recorded_hashes)

    missing = sorted(recorded_paths - current_paths)
    added = sorted(current_paths - recorded_paths)

    changed_hash: List[str] = []
    changed_size: List[str] = []
    for rel in sorted(current_paths & recorded_paths):
        if str(current[rel]["sha256"]) != recorded_hashes[rel]:
            changed_hash.append(rel)
        if rel in recorded_sizes and int(current[rel]["size_bytes"]) != recorded_sizes[rel]:
            changed_size.append(rel)

    ok = not (missing or added or changed_hash or changed_size or malformed)
    return {
        "source_dir": source_dir.as_posix(),
        "ok": ok,
        "checksums_found": (source_dir / "SHA256SUMS.txt").exists(),
        "missing_files": missing,
        "added_files": added,
        "changed_hashes": changed_hash,
        "changed_sizes": changed_size,
        "malformed_checksum_entries": malformed,
    }
