"""Small, dependency-light helpers for model loading and chat templates."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping


class ModelAccessError(RuntimeError):
    """A safe, actionable model-repository or cache access failure."""


def _get(cfg: Mapping[str, Any], key: str, default: Any = None) -> Any:
    try:
        return cfg.get(key, default)
    except AttributeError:
        return getattr(cfg, key, default)


def model_identifier(model_cfg: Mapping[str, Any]) -> str:
    return str(_get(model_cfg, "hf_id") or _get(model_cfg, "name") or "")


def model_revision(model_cfg: Mapping[str, Any]) -> str | None:
    value = _get(model_cfg, "revision")
    return str(value) if value not in (None, "", "null") else None


def hf_token() -> str | None:
    """Read the Hub token from the environment only.

    No config file, command-line argument, manifest, or log receives this value.
    """
    value = os.environ.get("HF_TOKEN")
    return value.strip() if value and value.strip() else None


def from_pretrained_kwargs(
    model_cfg: Mapping[str, Any],
    *,
    cache_dir: str | None = None,
    trust_remote_code: bool | None = None,
) -> dict[str, Any]:
    """Build safe common kwargs for tokenizer/model ``from_pretrained`` calls."""
    kwargs: dict[str, Any] = {}
    revision = model_revision(model_cfg)
    if revision:
        kwargs["revision"] = revision
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if trust_remote_code is None:
        trust_remote_code = bool(_get(model_cfg, "trust_remote_code", True))
    kwargs["trust_remote_code"] = trust_remote_code
    token = hf_token()
    if token:
        kwargs["token"] = token
    return kwargs


_CACHE_CORRUPTION_MARKERS = (
    "jsondecodeerror",
    "expecting value",
    "is not a valid json file",
    "unexpected end of json input",
)
_CACHE_PATH_PATTERN = re.compile(r"\bat ['\"](?P<path>[^'\"]+)['\"]", re.IGNORECASE)


def _exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_corrupt_hf_cache_error(exc: BaseException) -> bool:
    """Identify invalid cached Hub metadata without masking auth failures."""
    for current in _exception_chain(exc):
        message = str(current).lower()
        if any(marker in message for marker in _CACHE_CORRUPTION_MARKERS):
            return True
    return False


def _corrupt_cache_paths(exc: BaseException) -> list[Path]:
    """Extract safe Hugging Face snapshot JSON paths from a cache error."""
    paths: list[Path] = []
    for current in _exception_chain(exc):
        for match in _CACHE_PATH_PATTERN.finditer(str(current)):
            path = Path(match.group("path"))
            if (
                path.is_absolute()
                and path.suffix.lower() == ".json"
                and path.parent.parent.name.lower() == "snapshots"
                and path.parent.parent.parent.name.lower().startswith("models--")
                and path not in paths
            ):
                paths.append(path)
    return paths


def _remove_corrupt_cache_file(path: Path) -> list[Path]:
    """Remove one corrupt snapshot file and its exact content-addressed blob."""
    targets = [path]
    if path.is_symlink():
        resolved = path.resolve(strict=False)
        if (
            resolved != path
            and resolved.parent.name.lower() == "blobs"
            and resolved.parent.parent.name.lower().startswith("models--")
        ):
            targets.append(resolved)

    removed: list[Path] = []
    for target in targets:
        if not (target.is_symlink() or target.exists()):
            continue
        try:
            target.unlink()
        except OSError:
            continue
        removed.append(target)
    return removed


def load_pretrained_with_cache_repair(
    loader: Callable[..., Any],
    model_name: str,
    *,
    kwargs: Mapping[str, Any] | None = None,
    logger: Any = None,
    component: str = "Hugging Face asset",
) -> Any:
    """Load a Hub asset once, then force-refresh only on corrupt-cache errors.

    Interrupted downloads can leave an empty or truncated ``config.json`` in
    the local Hub snapshot. A normal ``from_pretrained`` call trusts that file
    and fails before it reaches the network. Retry with ``force_download`` for
    JSON/cache corruption; authentication, permissions, and unrelated model
    errors still fail immediately.
    """
    load_kwargs = dict(kwargs or {})
    try:
        return loader(model_name, **load_kwargs)
    except Exception as exc:
        if not is_corrupt_hf_cache_error(exc):
            raise

        removed_paths: list[Path] = []
        for cache_path in _corrupt_cache_paths(exc):
            removed_paths.extend(_remove_corrupt_cache_file(cache_path))

        if logger is not None:
            if removed_paths:
                logger.warning(
                    "Removed corrupt Hugging Face cache file(s): %s",
                    ", ".join(str(path) for path in removed_paths),
                )
            logger.warning(
                "Corrupt Hugging Face cache detected for %s (%s); "
                "retrying with force_download=True",
                model_name,
                component,
            )
        retry_kwargs = dict(load_kwargs)
        retry_kwargs["force_download"] = True
        return loader(model_name, **retry_kwargs)


def chat_template_kwargs(model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return supported template kwargs without injecting literal ``/think``.

    Qwen3 exposes ``enable_thinking`` in its Hub chat template. Other model
    families do not accept that keyword, so they receive no family-specific
    argument even when the profile records the common false setting.
    """
    family = str(_get(model_cfg, "family", "")).lower()
    template_cfg = _get(model_cfg, "chat_template", {}) or {}
    if family.startswith("qwen3"):
        value = _get(template_cfg, "enable_thinking")
        if value is not None:
            return {"enable_thinking": bool(value)}
    return {}


def safe_model_access_error(model_name: str, exc: BaseException) -> ModelAccessError:
    """Convert Hub/cache errors to a message that cannot leak credentials."""
    del exc
    return ModelAccessError(
        "MODEL_ACCESS_DENIED\n"
        f"Model: {model_name}\n"
        "Check: HF_TOKEN is set for gated models, the account has access, "
        "and the model cache/network is available."
    )
