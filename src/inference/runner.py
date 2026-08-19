"""Batch inference with cache + resume.

Inference over the test set is the expensive step, so it must never be repeated
unnecessarily. Predictions are appended to a JSONL file keyed by ``item_id``:
- if every requested item is already present, the run is a cache hit and returns
  immediately;
- if a run crashed halfway, re-running skips finished items and continues.

The runner depends only on the method interface, so it works unchanged for all
four methods.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any, List, Set

from src.core.interfaces import BaseMethod
from src.core.schemas import GenerationResult
from src.utils.io import read_jsonl

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
    ) -> None:
        self.method = method
        self.out_path = Path(out_path)
        self.cache_identity = cache_identity

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

    def _record_error(self, brief, variant: str, exc: Exception) -> None:
        """Append one failed item to the errors file so it is never silently lost."""
        rec = {
            "item_id": brief.item_id or "",
            "variant": variant,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        self.errors_path.parent.mkdir(parents=True, exist_ok=True)
        with self.errors_path.open("a", encoding="utf-8") as ef:
            ef.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def run(self, briefs, variant: str = "original") -> List[GenerationResult]:
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

        done = self._load_done()
        remaining = [b for b in briefs if (b.item_id or "") not in done]

        if not remaining:
            logger.info("[CACHE HIT] %s already complete (%d items).", self.out_path, len(done))
            print(f"[CACHE HIT] {self.out_path.name}: {len(done)} items already done.")
            return self._load_existing()

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Setting up method '%s'…", self.method.name)
        self.method.setup()
        n_errors = 0
        try:
            n = len(remaining)
            with self.out_path.open("a", encoding="utf-8") as f:
                for i, brief in enumerate(remaining, start=1):
                    try:
                        result = self.method.generate(brief)
                        result.variant = variant
                        f.write(result.model_dump_json() + "\n")
                        f.flush()
                        status = "ok" if result.parse_error is None else result.parse_error
                        print(f"  [{i:>3}/{n}] {brief.item_id} {status} ({result.latency_ms:.0f} ms)")
                    except Exception as exc:  # recoverable item errors do not abort the run
                        # Record it instead of dropping it: otherwise n_predictions
                        # silently differs across methods and a crashing method looks
                        # artificially better. See errors*.jsonl next to predictions.
                        logger.exception("Generation failed for %s: %s", brief.item_id, exc)
                        self._record_error(brief, variant, exc)
                        n_errors += 1
                        print(f"  [{i:>3}/{n}] {brief.item_id} ERROR: {exc}")
                        if _is_fatal_cuda_error(exc):
                            raise
        finally:
            self.method.teardown()

        if n_errors:
            print(f"  {n_errors} item(s) failed — logged to {self.errors_path.name}")

        return self._load_existing()
