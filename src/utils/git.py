"""Capture the current git commit hash (read-only) for run provenance."""

from __future__ import annotations

import subprocess


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
