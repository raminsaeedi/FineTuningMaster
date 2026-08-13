"""Small, dependency-light helpers for model loading and chat templates."""

from __future__ import annotations

import os
from typing import Any, Mapping


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
