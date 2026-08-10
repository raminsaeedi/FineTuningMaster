"""Capture the current git commit hash (read-only) for run provenance."""

from __future__ import annotations

import subprocess


from typing import Optional


def get_git_hash() -> str:
    """Return the current HEAD commit hash, or 'unknown' if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def is_git_dirty() -> Optional[bool]:
    """Return whether the working tree has uncommitted changes.

    A commit hash alone is misleading when the tree is dirty: the recorded
    commit is not the code that ran. ``None`` means the state could not be
    determined (no git, not a repository), which is distinct from "clean".
    """
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        )
        return bool(out.decode().strip())
    except Exception:
        return None
