"""Explicit, crash-safe status artifact for one inference run.

``predictions*.jsonl`` says what exists; it does not say whether the process that
wrote it finished, died, or was killed. ``inference_status.json`` records that,
next to the identity a resume has to match:

    running | interrupted | failed | completed

The file is written atomically at job boundaries — never per generated item,
because one fsync per generation would be measurable overhead on a file that is
provenance rather than data. ``predictions*.jsonl`` therefore remains the single
source of truth for which items are done; the status file always reports the ids
it read back from that file at write time, so the two cannot drift apart in a
way that matters.

The identity stored here is deliberately separate from ``cache_identity``:
adding a field to the latter would change ``cache_identity_hash`` for every run
already completed in this repository and invalidate caches that must stay
readable. A run that predates this file has no status, and is therefore never
gated by it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from src.utils.io import read_jsonl
from src.utils.resume import (
    INFERENCE_STATUS_FILENAME,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_RUNNING,
    PartialLineRepair,
    RunStatus,
    source_code_hash,
    utc_now,
)

logger = logging.getLogger(__name__)


def inference_identity(cfg: Any, project_root: Path, cache_identity: dict) -> dict:
    """What a resumed inference run must match to reuse existing predictions.

    Model identity and revision, dataset hashes, knowledge-base hash, seed and
    method all arrive through ``cache_identity``; the configuration hash and the
    source-code hash are added here, because nothing in the repository hashed
    the code that produced a prediction before.
    """
    return {
        "config_hash": str(cfg.get("config_hash", "") or ""),
        "code_hash": source_code_hash(project_root),
        "cache_identity": dict(cache_identity or {}),
    }


class InferenceStatusRecorder:
    """Owns ``inference_status.json`` for one run directory."""

    def __init__(
        self,
        run_dir: str | Path,
        identity: dict,
        *,
        resume: bool = True,
        experiment_id: str = "",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.status = RunStatus(
            path=self.run_dir / INFERENCE_STATUS_FILENAME,
            kind="inference",
            identity=identity,
        )
        self.resume = resume
        self.experiment_id = experiment_id
        self._variants: dict[str, dict] = {}
        self._repairs: list[dict] = []
        self._retries: list[dict] = []

    @property
    def path(self) -> Path:
        return self.status.path

    # -- lifecycle -----------------------------------------------------
    def start(self) -> None:
        """Validate any previous status, then mark this attempt as running.

        Raises :class:`src.utils.resume.IncompatibleResumeError` when the
        existing artifacts belong to a different run.
        """
        previous = self.status.check_compatible(resume=self.resume)
        if not self.resume:
            # A fresh run starts a fresh history: the previous attempt's files
            # have been set aside, so carrying its counters forward would
            # describe a run that no longer exists here.
            previous = None
        if previous:
            # Keep the original start time and the accumulated history so the
            # artifact describes the run, not only the newest attempt.
            self.status.started_utc = str(previous.get("started_utc") or utc_now())
            self._variants = dict(previous.get("variants") or {})
            self._repairs = list(previous.get("repairs") or [])
            self._retries = list(previous.get("retries") or [])
            attempts = int(previous.get("attempts", 1) or 1) + 1
            previous_status = str(previous.get("status") or "")
            if previous_status in {STATUS_RUNNING, STATUS_INTERRUPTED, STATUS_FAILED}:
                logger.info(
                    "Resuming an inference run previously left in state '%s'.",
                    previous_status,
                )
        else:
            attempts = 1
        self._write(STATUS_RUNNING, attempts=attempts, experiment_id=self.experiment_id)

    def finish(self) -> None:
        self._write(STATUS_COMPLETED, finished_utc=utc_now())

    def fail(self, exc: BaseException) -> None:
        self._write(
            STATUS_FAILED,
            finished_utc=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )

    def interrupt(self) -> None:
        self._write(STATUS_INTERRUPTED, finished_utc=utc_now())

    # -- progress ------------------------------------------------------
    def record_progress(self, variant: str, predictions_path: str | Path) -> None:
        """Store the ids currently on disk for one variant."""
        path = Path(predictions_path)
        item_ids: list[str] = []
        if path.exists():
            item_ids = [str(record.get("item_id", "")) for record in read_jsonl(path)]
        unique = sorted(set(item_ids))
        self._variants[variant] = {
            "predictions_file": path.name,
            "n_completed": len(unique),
            "n_records": len(item_ids),
            "has_duplicate_ids": len(unique) != len(item_ids),
            "completed_item_ids": unique,
            "updated_utc": utc_now(),
        }
        self._write(STATUS_RUNNING)

    def record_repair(self, variant: str, repair: PartialLineRepair) -> None:
        entry = repair.as_metadata()
        entry["variant"] = variant
        self._repairs.append(entry)

    def record_retry(self, variant: str, item_id: str, reason: str) -> None:
        """Note that an item is generated again after an interruption.

        With ``do_sample: true`` a regenerated item is a fresh sample. Nothing
        here asserts that it equals whatever the lost attempt produced.
        """
        self._retries.append({
            "variant": variant,
            "item_id": item_id,
            "reason": reason,
            "resampled": True,
            "bitwise_equality_verified": False,
            "utc": utc_now(),
        })

    # -- internals -----------------------------------------------------
    def _write(self, status: str, **updates: Any) -> None:
        self.status.write(
            status,
            variants=self._variants,
            repairs=self._repairs,
            retries=self._retries,
            **updates,
        )


class NullStatusRecorder:
    """No-op recorder, so callers never need a conditional."""

    path: Optional[Path] = None

    def start(self) -> None: ...
    def finish(self) -> None: ...
    def fail(self, exc: BaseException) -> None: ...
    def interrupt(self) -> None: ...
    def record_progress(self, variant: str, predictions_path: str | Path) -> None: ...
    def record_repair(self, variant: str, repair: PartialLineRepair) -> None: ...
    def record_retry(self, variant: str, item_id: str, reason: str) -> None: ...
