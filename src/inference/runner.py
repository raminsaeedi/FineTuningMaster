"""Batch inference with cache + resume.

Inference over the test set is the expensive step, so it must never be repeated
unnecessarily. Predictions are appended to a JSONL file keyed by ``item_id``:
- if every requested item is already present, the run is a cache hit and returns
  immediately;
- if a run crashed halfway, re-running skips finished items and continues.

The runner depends only on the method interface, so it works unchanged for all
four methods.

Generation is sequential by default — one item per ``generate`` call, which is
what every existing result file was produced with. A method may declare
``inference_batch_size > 1`` (opt-in only; see :mod:`src.inference.batching`),
in which case items are handed over in ordered chunks instead. Item order, item
ids, the JSONL format, the resume set and the error file are identical either
way.

Crash safety: an interrupted append leaves a truncated final line. Before the
resume set is computed, that fragment — and only that fragment — is removed by
:func:`src.utils.resume.repair_trailing_partial_line`, because the next append
would otherwise concatenate onto it and destroy the following record as well.
Every complete record before it is kept byte-for-byte. An item whose record was
lost that way is regenerated and recorded as a retry: with ``do_sample: true``
the regenerated text is a fresh sample, not a reproduction of the lost one.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any, List, Optional, Set

from src.core.interfaces import BaseMethod
from src.core.schemas import GenerationResult
from src.utils.io import read_jsonl
from src.utils.resume import repair_trailing_partial_line

logger = logging.getLogger(__name__)

_FATAL_CUDA_ERRORS = (
    "device-side assert triggered",
    "an illegal memory access was encountered",
    "unspecified launch failure",
)


def _is_fatal_cuda_error(exc: Exception) -> bool:
    """Return whether CUDA context cannot safely process another item."""
    message = str(exc).lower()
    return any(marker in message for marker in _FATAL_CUDA_ERRORS)


class InferenceRunner:
    """Run a method over a list of briefs, caching results to ``out_path``."""

    def __init__(
        self,
        method: BaseMethod,
        out_path: str | Path,
        cache_identity: dict[str, Any] | None = None,
        recorder: Any | None = None,
    ) -> None:
        self.method = method
        self.out_path = Path(out_path)
        self.cache_identity = cache_identity
        # Optional observer for the run status artifact. ``None`` keeps the
        # runner usable standalone (and keeps every existing caller working).
        self.recorder = recorder

    def _expected_config_hash(self) -> str:
        return str(getattr(self.method, "config_hash", "") or "")

    def _stale_config_hashes(self) -> Set[str]:
        """Config hashes present in the cache that differ from the current run.

        The cache is keyed by ``item_id`` alone, so an override that changes
        generation settings or the adapter — without changing the experiment
        name or seed, which is what determines the directory — would otherwise be
        served from predictions produced under different settings.
        """
        expected = self._expected_config_hash()
        if not expected or not self.out_path.exists():
            return set()
        seen = {str(r.get("config_hash", "") or "") for r in read_jsonl(self.out_path)}
        return {h for h in seen if h and h != expected}

    def _load_done(self) -> Set[str]:
        if not self.out_path.exists():
            return set()
        return {r.get("item_id", "") for r in read_jsonl(self.out_path)}

    def _load_existing(self) -> List[GenerationResult]:
        if not self.out_path.exists():
            return []
        return [GenerationResult(**r) for r in read_jsonl(self.out_path)]

    @property
    def cache_identity_path(self) -> Path:
        return self.out_path.parent / "cache_identity.json"

    def _check_cache_identity(self) -> None:
        """Reject a cache from another dataset/model/method/seed identity."""
        if self.cache_identity is None or not self.cache_identity_path.exists():
            return
        try:
            stored = json.loads(self.cache_identity_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid cache identity next to {self.out_path}: {self.cache_identity_path}"
            ) from exc
        if stored != self.cache_identity:
            raise RuntimeError(
                f"{self.out_path} holds predictions for a different cache identity. "
                "Use a distinct run directory or remove only the incompatible cache."
            )

    @property
    def errors_path(self) -> Path:
        """Sibling of ``predictions*.jsonl`` recording items that raised."""
        return self.out_path.parent / self.out_path.name.replace("predictions", "errors")

    def _record_error(self, brief, variant: str, exc: BaseException) -> None:
        """Append one failed item to the errors file so it is never silently lost."""
        # Formatted from the exception object, not from sys.exc_info(): a batched
        # item's failure is reported after its except block has already exited.
        rec = {
            "item_id": brief.item_id or "",
            "variant": variant,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        }
        self.errors_path.parent.mkdir(parents=True, exist_ok=True)
        with self.errors_path.open("a", encoding="utf-8") as ef:
            ef.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _repair_partial_tail(self, variant: str) -> Optional[str]:
        """Remove an interrupted trailing line; return the item id it held."""
        repair = repair_trailing_partial_line(self.out_path)
        if repair is None:
            return None
        logger.warning(
            "Removed an incomplete trailing line from %s (%d bytes, %d complete "
            "records kept); backup: %s",
            self.out_path,
            repair.removed_bytes,
            repair.kept_records,
            repair.backup_path,
        )
        print(
            f"[RESUME] {self.out_path.name}: dropped an incomplete final line "
            f"({repair.removed_bytes} bytes); {repair.kept_records} complete "
            "records preserved."
        )
        if self.recorder is not None:
            self.recorder.record_repair(variant, repair)
            if repair.recovered_item_id:
                self.recorder.record_retry(
                    variant,
                    repair.recovered_item_id,
                    "record lost in an interrupted append; regenerated",
                )
        return repair.recovered_item_id

    def _prepare(self, briefs, variant: str = "original") -> List[Any]:
        """Validate cache and return only briefs that still need generation."""
        self._repair_partial_tail(variant)
        self._check_cache_identity()
        stale = self._stale_config_hashes()
        if stale:
            raise RuntimeError(
                f"{self.out_path} holds predictions generated under a different "
                f"configuration (config_hash {sorted(stale)} != "
                f"{self._expected_config_hash()}).\n"
                f"Reusing them would mix settings within one result file. Delete "
                f"the file to regenerate, or run under a distinct experiment_name."
            )

        briefs = list(briefs)
        done = self._load_done()
        # Order follows the requested items, not the file: a resumed run appends
        # the missing ids in dataset order, exactly where an uninterrupted run
        # would have written them.
        remaining = [b for b in briefs if (b.item_id or "") not in done]
        self._retry_previously_failed(remaining, variant)

        if not remaining:
            logger.info("[CACHE HIT] %s already complete (%d items).", self.out_path, len(done))
            print(f"[CACHE HIT] {self.out_path.name}: {len(done)} items already done.")
        return remaining

    def _retry_previously_failed(self, remaining: List[Any], variant: str) -> None:
        """Record items that failed in an earlier attempt and run again now."""
        if self.recorder is None or not remaining or not self.errors_path.exists():
            return
        failed = {
            str(record.get("item_id", ""))
            for record in read_jsonl(self.errors_path)
            if str(record.get("variant", "")) == variant
        }
        for brief in remaining:
            if (brief.item_id or "") in failed:
                self.recorder.record_retry(
                    variant, brief.item_id or "", "failed in an earlier attempt"
                )

    def _batch_size(self) -> int:
        """Items per generate call. 1 (the default) keeps the sequential path."""
        try:
            size = int(getattr(self.method, "inference_batch_size", 1) or 1)
        except (TypeError, ValueError):
            return 1
        return max(1, size)

    def _run_remaining(self, remaining: List[Any], variant: str) -> List[GenerationResult]:
        """Generate prepared items; caller owns method setup/teardown lifecycle."""
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        batch_size = self._batch_size()
        n = len(remaining)
        # Last line of defence against a duplicated item_id: the resume set is
        # computed once per job, so anything already on disk (or written earlier
        # in this job) must never be appended a second time.
        self._written_ids: Set[str] = self._load_done()
        try:
            with self.out_path.open("a", encoding="utf-8") as f:
                if batch_size <= 1:
                    n_errors = self._generate_sequential(f, remaining, variant, n)
                else:
                    n_errors = self._generate_batched(f, remaining, variant, n, batch_size)
        finally:
            if self.recorder is not None:
                self.recorder.record_progress(variant, self.out_path)

        if n_errors:
            print(f"  {n_errors} item(s) failed — logged to {self.errors_path.name}")

        return self._load_existing()

    def _generate_sequential(self, f, remaining: List[Any], variant: str, n: int) -> int:
        """The default path: one item per generate call, unchanged."""
        n_errors = 0
        for i, brief in enumerate(remaining, start=1):
            try:
                result = self.method.generate(brief)
                self._write(f, result, variant, brief, i, n)
            except Exception as exc:  # recoverable item errors do not abort the run
                # Record it instead of dropping it: otherwise n_predictions
                # silently differs across methods and a crashing method looks
                # artificially better. See errors*.jsonl next to predictions.
                n_errors += self._handle_item_error(brief, variant, exc, i, n)
        return n_errors

    def _generate_batched(
        self, f, remaining: List[Any], variant: str, n: int, batch_size: int
    ) -> int:
        """Opt-in throughput path: several items per generate call.

        Items are consumed in their original order, and each batch's results are
        written in that same order before the next batch starts, so the output
        file, the resume set and the error file are indistinguishable in shape
        from the sequential path. Only the generated text and the latency
        semantics differ — see :mod:`src.inference.batching`.
        """
        n_errors = 0
        for start in range(0, n, batch_size):
            chunk = remaining[start : start + batch_size]
            outcomes = self.method.generate_batch(chunk)
            if len(outcomes) != len(chunk):
                raise RuntimeError(
                    f"{type(self.method).__name__}.generate_batch returned "
                    f"{len(outcomes)} outcomes for {len(chunk)} items; item "
                    "identity could not be preserved."
                )
            for offset, (brief, outcome) in enumerate(zip(chunk, outcomes)):
                i = start + offset + 1
                if isinstance(outcome, BaseException):
                    n_errors += self._handle_item_error(brief, variant, outcome, i, n)
                else:
                    self._write(f, outcome, variant, brief, i, n)
        return n_errors

    def _write(self, f, result: GenerationResult, variant: str, brief, i: int, n: int) -> None:
        item_id = result.item_id or ""
        written = getattr(self, "_written_ids", None)
        if written is not None:
            if item_id in written:
                # A completed prediction is never overwritten and never doubled.
                logger.warning(
                    "Refusing to append a duplicate prediction for %s to %s",
                    item_id, self.out_path,
                )
                print(f"  [{i:>3}/{n}] {item_id} SKIPPED (already present)")
                return
            written.add(item_id)
        result.variant = variant
        f.write(result.model_dump_json() + "\n")
        f.flush()
        status = "ok" if result.parse_error is None else result.parse_error
        print(f"  [{i:>3}/{n}] {brief.item_id} {status} ({result.latency_ms:.0f} ms)")

    def _handle_item_error(self, brief, variant: str, exc: BaseException, i: int, n: int) -> int:
        logger.error("Generation failed for %s: %s", brief.item_id, exc, exc_info=exc)
        self._record_error(brief, variant, exc)
        print(f"  [{i:>3}/{n}] {brief.item_id} ERROR: {exc}")
        if _is_fatal_cuda_error(exc):
            raise exc
        return 1

    def run(self, briefs, variant: str = "original") -> List[GenerationResult]:
        """Run one file, loading and releasing method resources around it."""
        remaining = self._prepare(briefs, variant)
        if not remaining:
            if self.recorder is not None:
                self.recorder.record_progress(variant, self.out_path)
            return self._load_existing()

        logger.info("Setting up method '%s'…", self.method.name)
        self.method.setup()
        try:
            return self._run_remaining(remaining, variant)
        finally:
            self.method.teardown()

    def run_many(
        self,
        jobs: list[tuple[Any, str | Path, str]],
    ) -> List[List[GenerationResult]]:
        """Run several cache files while loading one method instance once.

        Jobs execute in supplied order. Cache checks still happen before model
        setup, and fatal CUDA errors still stop the batch immediately.
        """
        runners = [
            InferenceRunner(
                self.method,
                out_path,
                cache_identity=self.cache_identity,
                recorder=self.recorder,
            )
            for _briefs, out_path, _variant in jobs
        ]
        results: List[Optional[List[GenerationResult]]] = [None] * len(jobs)
        pending: list[tuple[int, InferenceRunner, List[Any], str]] = []

        for index, (briefs, _out_path, variant) in enumerate(jobs):
            remaining = runners[index]._prepare(briefs, variant)
            if remaining:
                pending.append((index, runners[index], remaining, variant))
            else:
                # A finished variant still reports its completed ids, so the
                # status file describes the whole run, not only what this
                # invocation had to generate.
                if self.recorder is not None:
                    self.recorder.record_progress(variant, runners[index].out_path)
                results[index] = runners[index]._load_existing()

        if not pending:
            return [value or [] for value in results]

        logger.info("Setting up method '%s' once for %d inference files…", self.method.name, len(pending))
        self.method.setup()
        try:
            for index, runner, remaining, variant in pending:
                results[index] = runner._run_remaining(remaining, variant)
        finally:
            self.method.teardown()

        return [value or [] for value in results]
