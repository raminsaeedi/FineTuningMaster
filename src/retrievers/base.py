"""Shared helpers for retrievers."""

from __future__ import annotations

from typing import Any, List, Mapping

from src.core.interfaces import BaseRetriever  # re-exported for convenience

__all__ = ["BaseRetriever", "get_cfg", "format_passages"]


def get_cfg(cfg: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Read a key from a dict or OmegaConf DictConfig uniformly."""
    try:
        return cfg.get(key, default)
    except AttributeError:
        return getattr(cfg, key, default)


def format_passages(passages: List[dict]) -> str:
    """Render retrieved passages into a text block for prompt injection."""
    lines: List[str] = []
    for i, p in enumerate(passages, start=1):
        source = p.get("source", "guideline")
        heading = p.get("heading", "")
        title = f"{source} — {heading}".strip(" —")
        lines.append(f"[{i}] {title}\n{p.get('text', '').strip()}")
    return "\n\n".join(lines)
