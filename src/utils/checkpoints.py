"""Publishing a Trainer checkpoint as complete.

The Trainer writes a checkpoint directory file by file: weights, optimizer,
scheduler, RNG state, ``trainer_state.json``. A process killed anywhere in that
sequence leaves a directory that *looks* like a checkpoint and is missing
something a resume needs. Structural validation catches most of it, but "all
expected files exist" is not the same statement as "the writer finished".

So the last step of a successful save is to publish a marker,
``checkpoint_complete.json``, written to a temporary file and moved into place
with a single atomic ``os.replace``. The marker either exists in full or does
not exist at all; there is no partial state in between. A checkpoint written
before this mechanism existed simply has no marker and is judged on its files
alone, so older runs stay resumable.

Kept separate from the trainer module so validation (in
``experiments/scripts/train.py``) can import it without importing torch or trl.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.utils.resume import CHECKPOINT_COMPLETE_FILENAME, atomic_write_json, utc_now

logger = logging.getLogger(__name__)


def mark_checkpoint_complete(
    checkpoint_dir: str | Path, *, global_step: Optional[int] = None
) -> Optional[Path]:
    """Publish ``checkpoint_dir`` as fully written; return the marker path.

    Never raises: a checkpoint that cannot be marked is still validated
    structurally, and failing the training run because a marker could not be
    written would trade a small safety net for the whole job.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        logger.warning("Cannot mark a missing checkpoint directory: %s", checkpoint_dir)
        return None
    try:
        return atomic_write_json(
            checkpoint_dir / CHECKPOINT_COMPLETE_FILENAME,
            {
                "completed_utc": utc_now(),
                "global_step": int(global_step) if global_step is not None else None,
                "checkpoint": checkpoint_dir.name,
            },
        )
    except OSError:
        logger.exception("Could not write the completion marker for %s", checkpoint_dir)
        return None
