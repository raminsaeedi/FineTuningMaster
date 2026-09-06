"""Crash-safe resume primitives shared by inference and training.

A thesis run is long, expensive and routinely interrupted — a preempted GPU job,
a full disk, a killed terminal. Interruption must cost the remaining work, never
the finished work. Three mechanisms here, all of them additive to what the
pipeline already does:

Atomic status writes
    ``atomic_write_json`` writes to a temporary file in the same directory and
    then ``os.replace``s it into place. A reader therefore sees either the old
    file or the new one, never a half-written one. Status is written at job
    boundaries, not per item: ``predictions*.jsonl`` is the authority for what is
    finished, and one fsync per generated item would be real overhead for a file
    that is only provenance.

Partial-line repair
    Appending a JSONL record is one ``write`` plus a flush. A process killed in
    the middle leaves a truncated final line. ``read_jsonl`` already skips it,
    but the *next* append concatenates onto that fragment and silently destroys
    the following record too. ``repair_trailing_partial_line`` removes exactly
    that trailing fragment, keeps every earlier byte, and preserves the removed
    text in a ``.partial-<timestamp>.bak`` sibling for audit.

Identity gating
    Resuming is only safe when the artifact was produced by the same run: same
    model and revision, dataset, knowledge base, seed, method, configuration —
    and the same code. ``source_code_hash`` covers the last one, which no other
    hash in the repository did.

Nothing here changes generation, training, or any recorded metric.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

STATUS_RUNNING = "running"
STATUS_INTERRUPTED = "interrupted"
STATUS_FAILED = "failed"
STATUS_COMPLETED = "completed"

STATUS_SCHEMA_VERSION = 1

#: Filename of the inference status artifact inside a run directory.
INFERENCE_STATUS_FILENAME = "inference_status.json"
#: Filename of the training status artifact inside a run directory.
TRAINING_STATUS_FILENAME = "training_status.json"
#: Written into a Trainer checkpoint once it is completely on disk.
CHECKPOINT_COMPLETE_FILENAME = "checkpoint_complete.json"

_ITEM_ID_RE = re.compile(r'"item_id"\s*:\s*"((?:[^"\\]|\\.)*)"')


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------

def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write ``text`` to ``path`` so a reader never sees a partial file.

    The temporary file is created in the destination directory, so the final
    ``os.replace`` is a same-filesystem rename and therefore atomic.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(handle.name, path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise
    return path


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=2, default=str) + "\n")


def read_json_or_none(path: str | Path) -> Optional[dict]:
    """Read a JSON object, returning ``None`` for absent or unreadable files."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


# ---------------------------------------------------------------------------
# Code identity
# ---------------------------------------------------------------------------

_CODE_HASH_CACHE: dict[str, str] = {}


def source_code_hash(project_root: str | Path, *, use_cache: bool = True) -> str:
    """Stable hash of the Python sources that produce predictions.

    Covers ``src/`` only, excluding ``src/tests`` and ``__pycache__``: those are
    the modules that build prompts, retrieve, generate and parse. Editing them
    mid-run means a resumed run would mix two implementations, which is exactly
    what the resume gate must catch. Test files and notebooks cannot change a
    prediction, so they are excluded to avoid pointless refusals.
    """
    root = Path(project_root).resolve()
    key = str(root)
    if use_cache and key in _CODE_HASH_CACHE:
        return _CODE_HASH_CACHE[key]

    source_root = root / "src"
    digest = hashlib.sha256()
    if source_root.is_dir():
        files = sorted(
            path for path in source_root.rglob("*.py")
            if "__pycache__" not in path.parts
            and "tests" not in path.relative_to(source_root).parts[:1]
        )
        for path in files:
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            try:
                digest.update(hashlib.sha256(path.read_bytes()).digest())
            except OSError:
                digest.update(b"<unreadable>")
    value = digest.hexdigest()[:12]
    if use_cache:
        _CODE_HASH_CACHE[key] = value
    return value


# ---------------------------------------------------------------------------
# Partial trailing line repair
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PartialLineRepair:
    """What was removed from the end of a JSONL file, and where it went."""

    path: Path
    removed_bytes: int
    removed_text: str
    recovered_item_id: Optional[str]
    backup_path: Optional[Path]
    kept_records: int

    def as_metadata(self) -> dict:
        return {
            "path": str(self.path),
            "removed_bytes": self.removed_bytes,
            "recovered_item_id": self.recovered_item_id,
            "backup_path": str(self.backup_path) if self.backup_path else None,
            "kept_records": self.kept_records,
            "utc": utc_now(),
        }


def _valid_json_line(raw: bytes) -> bool:
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return True
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def repair_trailing_partial_line(path: str | Path) -> Optional[PartialLineRepair]:
    """Drop an incomplete final line, keeping every complete record before it.

    Two interruption shapes are handled: a file that does not end with a
    newline, and a file whose final line is not parseable JSON. Both mean the
    process died mid-append. Earlier lines are never inspected or rewritten —
    the file is truncated in place at the last known-good byte offset, so no
    completed prediction can be lost by this operation.

    Returns ``None`` when there is nothing to repair.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None

    data = path.read_bytes()
    newline = data.rfind(b"\n")

    if newline == len(data) - 1:
        # Ends with a newline: the last record is only suspect if it does not
        # parse, which a torn write can produce when the fragment happened to
        # end on a newline byte.
        previous = data.rfind(b"\n", 0, newline)
        last_line = data[previous + 1 : newline]
        if _valid_json_line(last_line):
            return None
        keep_to = previous + 1
    elif newline == -1:
        # A single incomplete line and nothing else.
        keep_to = 0
    else:
        keep_to = newline + 1

    removed = data[keep_to:]
    if not removed.strip():
        return None

    removed_text = removed.decode("utf-8", errors="replace")
    backup_path = path.with_name(
        f"{path.name}.partial-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.bak"
    )
    try:
        backup_path.write_bytes(removed)
    except OSError:
        backup_path = None

    with path.open("r+b") as handle:
        handle.truncate(keep_to)
        handle.flush()
        os.fsync(handle.fileno())

    match = _ITEM_ID_RE.search(removed_text)
    kept = data[:keep_to].count(b"\n")
    return PartialLineRepair(
        path=path,
        removed_bytes=len(removed),
        removed_text=removed_text,
        recovered_item_id=match.group(1) if match else None,
        backup_path=backup_path,
        kept_records=kept,
    )


# ---------------------------------------------------------------------------
# Run status
# ---------------------------------------------------------------------------

class IncompatibleResumeError(RuntimeError):
    """Raised when existing artifacts were produced by a different run."""


@dataclass
class RunStatus:
    """Explicit status artifact for one run stage, written atomically.

    ``identity`` is what a resume is checked against. It is stored here rather
    than folded into the existing ``cache_identity`` on purpose: adding a field
    there would change ``cache_identity_hash`` for every completed run in the
    repository and invalidate caches that must stay readable.
    """

    path: Path
    kind: str
    identity: dict
    status: str = STATUS_RUNNING
    started_utc: str = field(default_factory=utc_now)
    payload: dict = field(default_factory=dict)

    # -- construction -------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> Optional[dict]:
        return read_json_or_none(path)

    def to_dict(self) -> dict:
        record = {
            "schema_version": STATUS_SCHEMA_VERSION,
            "kind": self.kind,
            "status": self.status,
            "started_utc": self.started_utc,
            "updated_utc": utc_now(),
            "identity": dict(self.identity),
        }
        record.update(self.payload)
        return record

    def write(self, status: Optional[str] = None, **updates: Any) -> dict:
        if status is not None:
            self.status = status
        self.payload.update(updates)
        record = self.to_dict()
        atomic_write_json(self.path, record)
        return record

    # -- gating -------------------------------------------------------
    def check_compatible(self, *, resume: bool) -> Optional[dict]:
        """Validate a previous status file against this run's identity.

        With ``resume=False`` nothing is checked: the caller is starting fresh
        and is responsible for setting existing artifacts aside first.
        """
        previous = self.load(self.path)
        if previous is None or not resume:
            return previous
        stored = previous.get("identity") or {}
        problems = [
            f"{key}: previous run has {stored.get(key)!r}, this run has {value!r}"
            for key, value in self.identity.items()
            if key in stored and stored.get(key) != value
        ]
        if problems:
            raise IncompatibleResumeError(
                f"{self.path} belongs to a different run and cannot be resumed:\n"
                + "\n".join(f"- {problem}" for problem in problems)
                + "\nRe-run with --no-resume to start this stage fresh (existing "
                "files are preserved under _stale_cache/), or use a distinct "
                "experiment_id."
            )
        return previous


def quarantine_files(paths: Iterable[Path], target_dir: Path) -> list[Path]:
    """Move files aside instead of deleting them; return the new locations.

    Used by ``--no-resume``: a fresh start must not destroy a previous run's
    predictions, only stop them from being reused.
    """
    import shutil

    moved: list[Path] = []
    target_dir = Path(target_dir)
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        destination = target_dir / path.name
        counter = 1
        while destination.exists():
            destination = target_dir / f"{path.name}.{counter}"
            counter += 1
        shutil.move(str(path), str(destination))
        moved.append(destination)
    return moved
