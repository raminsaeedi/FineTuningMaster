"""OpenAI-compatible provider layer for the LLM enrichment stage (Phase 3).

The enrichment stage talks to an OpenAI-compatible gateway (adesso AI Hub) via
the official ``openai`` package. Everything environment-specific is read from the
process environment, so the same code runs locally and on a remote host:

    ENRICHMENT_BASE_URL          endpoint, e.g. https://adesso-ai-hub.3asabc.de/v1
    ENRICHMENT_API_KEY           API key (the *only* place a secret ever lives)
    ENRICHMENT_MODEL             model id, e.g. deepseek-v4-flash-sovereign
    ENRICHMENT_REASONING_EFFORT  requested reasoning effort, e.g. xhigh

Variables may also be placed in the git-ignored ``.env`` file at the project root
(see :func:`load_dotenv_values`); real process environment variables always win.

Defaults for everything except the key come from
``src/config/data/enrichment_provider.yaml``, which contains no secrets.

Secret handling contract:
  * the key is read from the environment only, never from a file, never a literal;
  * the key value never reaches a log line, report, manifest, or exception text --
    :func:`sanitize_error` scrubs it (and any bearer token) from every message
    that leaves this module.

Reasoning-effort contract (capability-safe, never overclaimed):
  * the request is attempted with ``reasoning_effort`` only if the installed SDK
    accepts the parameter (:func:`sdk_supports_reasoning_effort`);
  * if the endpoint rejects it, :data:`REASONING_EFFORT_UNSUPPORTED_BY_PROVIDER`
    is recorded and the request is retried once without the parameter, using the
    model's default reasoning mode;
  * :attr:`CompletionResult.applied_reasoning_effort` is ``None`` whenever the
    provider did not accept the requested value, so no artifact can claim that
    ``xhigh`` was used when it was not.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.utils.io import read_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "data" / "enrichment_provider.yaml"

REASONING_EFFORT_UNSUPPORTED_BY_PROVIDER = "REASONING_EFFORT_UNSUPPORTED_BY_PROVIDER"
STRUCTURED_OUTPUT_UNSUPPORTED_BY_PROVIDER = "STRUCTURED_OUTPUT_UNSUPPORTED_BY_PROVIDER"

REDACTED = "[REDACTED]"

# Wordings an OpenAI-compatible gateway uses when it does not know a parameter.
_UNSUPPORTED_MARKERS = (
    "unsupported parameter",
    "unsupported_parameter",
    "unsupported value",
    "unrecognized request argument",
    "unknown parameter",
    "unknown field",
    "unknown argument",
    "extra inputs are not permitted",
    "extra_forbidden",
    "unexpected keyword argument",
    "not supported",
    "not permitted",
    "is invalid",
    "invalid value",
    "invalid_request_error",
)


DOTENV_PATH = PROJECT_ROOT / ".env"


def load_dotenv_values(path: str | Path | None = None) -> Dict[str, str]:
    """Parse ``KEY=VALUE`` lines from the git-ignored ``.env`` (no dependency).

    Comments and blank lines are skipped; surrounding quotes are stripped. Values
    are returned, never logged -- callers must keep them out of every artifact.
    """
    dotenv = Path(path) if path is not None else DOTENV_PATH
    if not dotenv.exists():
        return {}
    values: Dict[str, str] = {}
    for line in dotenv.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if name.startswith("export "):
            name = name[len("export "):].strip()
        value = value.strip().strip('"').strip("'")
        if name:
            values[name] = value
    return values


def resolve_env(env: Optional[Mapping[str, str]] = None,
                dotenv_path: str | Path | None = None) -> Mapping[str, str]:
    """Effective environment: ``.env`` values overlaid by the real environment.

    An explicitly passed ``env`` is used verbatim (tests stay hermetic).
    """
    if env is not None:
        return env
    merged: Dict[str, str] = dict(load_dotenv_values(dotenv_path))
    merged.update({k: v for k, v in os.environ.items() if v})
    return merged


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved provider settings. Holds no secret -- only the env var name."""

    base_url: str
    model: str
    reasoning_effort: Optional[str]
    temperature: float
    max_tokens: int
    timeout_seconds: float
    max_retries: int
    api_key_env: str = "ENRICHMENT_API_KEY"
    config_path: Optional[str] = None

    def public_summary(self) -> Dict[str, Any]:
        """Secret-free view for manifests and reports."""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "requested_reasoning_effort": self.reasoning_effort,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "api_key_env": self.api_key_env,
            "api_key_value_recorded": False,
        }


def load_provider_config(
    path: str | Path | None = None,
    env: Optional[Mapping[str, str]] = None,
) -> ProviderConfig:
    """Load YAML defaults, then let environment variables (or ``.env``) win."""
    env = resolve_env(env)
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw: Dict[str, Any] = read_yaml(cfg_path) if cfg_path.exists() else {}

    api_key_env = str(raw.get("api_key_env", "ENRICHMENT_API_KEY"))
    base_url = env.get("ENRICHMENT_BASE_URL") or str(raw.get("base_url", ""))
    model = env.get("ENRICHMENT_MODEL") or str(raw.get("model", ""))

    effort_raw = env.get("ENRICHMENT_REASONING_EFFORT")
    if effort_raw is None:
        effort_raw = raw.get("reasoning_effort")
    effort = str(effort_raw).strip() if effort_raw is not None else ""

    return ProviderConfig(
        base_url=base_url.strip().rstrip("/"),
        model=model.strip(),
        reasoning_effort=effort or None,
        temperature=float(raw.get("temperature", 0.1)),
        max_tokens=int(raw.get("max_tokens", 1200)),
        timeout_seconds=float(raw.get("timeout_seconds", 120)),
        max_retries=int(raw.get("max_retries", 2)),
        api_key_env=api_key_env,
        config_path=str(cfg_path),
    )


def validate_config(cfg: ProviderConfig) -> List[str]:
    """Return human-readable problems with the resolved config (no secrets)."""
    problems: List[str] = []
    if not cfg.base_url:
        problems.append("base_url is empty (set ENRICHMENT_BASE_URL or the config default)")
    elif not cfg.base_url.startswith(("http://", "https://")):
        problems.append(f"base_url must start with http:// or https:// (got {cfg.base_url!r})")
    if not cfg.model:
        problems.append("model is empty (set ENRICHMENT_MODEL or the config default)")
    if cfg.max_tokens <= 0:
        problems.append("max_tokens must be > 0")
    if cfg.timeout_seconds <= 0:
        problems.append("timeout_seconds must be > 0")
    if not cfg.api_key_env:
        problems.append("api_key_env is empty (cannot locate the API key variable)")
    return problems


def api_key_present(cfg: ProviderConfig, env: Optional[Mapping[str, str]] = None) -> bool:
    """True when the key variable is set and non-blank. The value is not returned."""
    env = resolve_env(env)
    return bool((env.get(cfg.api_key_env) or "").strip())


def _api_key(cfg: ProviderConfig, env: Optional[Mapping[str, str]] = None) -> str:
    env = resolve_env(env)
    key = (env.get(cfg.api_key_env) or "").strip()
    if not key:
        raise MissingCredentialsError(
            f"{cfg.api_key_env} is not set in the environment or .env; "
            "see docs/project/ENRICHMENT_PROVIDER_SETUP.md"
        )
    return key


class MissingCredentialsError(RuntimeError):
    """Raised when the API key environment variable is unset."""


def build_client(cfg: ProviderConfig, env: Optional[Mapping[str, str]] = None):
    """Build the OpenAI-compatible client. Key comes from the environment only."""
    from openai import OpenAI

    return OpenAI(
        api_key=_api_key(cfg, env),
        base_url=cfg.base_url,
        timeout=cfg.timeout_seconds,
        max_retries=cfg.max_retries,
    )


def sdk_supports_reasoning_effort() -> bool:
    """True when the installed openai SDK accepts ``reasoning_effort``."""
    try:
        from openai.resources.chat.completions import Completions
    except Exception:  # pragma: no cover - SDK layout change
        return False
    try:
        return "reasoning_effort" in inspect.signature(Completions.create).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return False


def sanitize_error(error: Any, cfg: Optional[ProviderConfig] = None,
                   env: Optional[Mapping[str, str]] = None) -> str:
    """Return a one-line error message with any secret material removed."""
    env = resolve_env(env)
    text = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)

    secrets = []
    if cfg is not None:
        secrets.append((env.get(cfg.api_key_env) or "").strip())
    secrets.extend((env.get(name) or "").strip() for name in
                   ("ENRICHMENT_API_KEY", "OPENAI_API_KEY"))
    for secret in secrets:
        if secret and len(secret) >= 8:
            text = text.replace(secret, REDACTED)

    text = re.sub(r"(?i)bearer\s+[A-Za-z0-9._\-]+", f"Bearer {REDACTED}", text)
    text = re.sub(r"(?i)(authorization\W{0,4})([^,\s'\"]+)", rf"\1{REDACTED}", text)
    text = re.sub(r"(?i)((?:api[_-]?key|token|secret)\W{0,4})([A-Za-z0-9._\-]{8,})",
                  rf"\1{REDACTED}", text)
    # Covers full keys and provider-masked forms such as "sk-...AbCd".
    text = re.sub(r"\bsk-[A-Za-z0-9._\-]{4,}", REDACTED, text)
    return " ".join(text.split())[:500]


def is_unsupported_parameter_error(error: BaseException, param: str) -> bool:
    """True when ``error`` looks like the endpoint/SDK rejecting ``param``."""
    text = str(error).lower()
    name = param.lower()
    if isinstance(error, TypeError):
        return name in text or "unexpected keyword argument" in text
    status = getattr(error, "status_code", None)
    if status is not None and int(status) not in (400, 404, 415, 422, 501):
        return False
    if name not in text and name.replace("_", " ") not in text:
        return False
    return any(marker in text for marker in _UNSUPPORTED_MARKERS)


@dataclass
class CompletionResult:
    """Outcome of one enrichment request, safe to serialize into a manifest."""

    text: str
    model: str
    requested_reasoning_effort: Optional[str]
    applied_reasoning_effort: Optional[str]
    reasoning_effort_supported: Optional[bool]
    structured_output_requested: bool
    structured_output_applied: bool
    latency_ms: int
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    @property
    def truncated(self) -> bool:
        """True when the provider stopped at the output-token cap, not at an end of answer."""
        return self.finish_reason == "length"

    @property
    def applied_reasoning_mode(self) -> str:
        """Human-readable applied mode; never claims an unconfirmed effort."""
        if self.applied_reasoning_effort:
            return self.applied_reasoning_effort
        return "provider_default"


def _usage_of(response: Any) -> Dict[str, Optional[int]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _text_of(response: Any) -> str:
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError, KeyError, TypeError):
        return ""


def _finish_reason_of(response: Any) -> Optional[str]:
    try:
        return response.choices[0].finish_reason
    except (AttributeError, IndexError, KeyError, TypeError):
        return None


def create_completion(
    client: Any,
    cfg: ProviderConfig,
    messages: Sequence[Mapping[str, str]],
    *,
    json_object: bool = False,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> CompletionResult:
    """One chat completion with capability-safe reasoning-effort and JSON mode.

    Fallback order (at most one retry per unsupported parameter):
      1. requested ``reasoning_effort`` (+ ``response_format`` when asked);
      2. on rejection of ``reasoning_effort``: retry without it, default mode;
      3. on rejection of ``response_format``: retry without it, plain text.
    """
    request_kwargs: Dict[str, Any] = {
        "model": cfg.model,
        "messages": list(messages),
        "temperature": cfg.temperature if temperature is None else temperature,
        "max_tokens": cfg.max_tokens if max_tokens is None else max_tokens,
    }
    if json_object:
        request_kwargs["response_format"] = {"type": "json_object"}

    requested_effort = cfg.reasoning_effort
    notes: List[str] = []
    sdk_ok = sdk_supports_reasoning_effort()
    if requested_effort and not sdk_ok:
        notes.append("REASONING_EFFORT_UNSUPPORTED_BY_SDK")
    use_effort = bool(requested_effort) and sdk_ok
    if use_effort:
        request_kwargs["reasoning_effort"] = requested_effort

    effort_supported: Optional[bool] = None if not requested_effort else (True if use_effort else False)
    structured_applied = json_object

    started = time.perf_counter()
    while True:
        try:
            response = client.chat.completions.create(**request_kwargs)
            break
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is a capability issue
            if "reasoning_effort" in request_kwargs and is_unsupported_parameter_error(exc, "reasoning_effort"):
                notes.append(REASONING_EFFORT_UNSUPPORTED_BY_PROVIDER)
                notes.append(f"provider_rejection: {sanitize_error(exc, cfg)}")
                request_kwargs.pop("reasoning_effort")
                effort_supported = False
                continue
            if "response_format" in request_kwargs and is_unsupported_parameter_error(exc, "response_format"):
                notes.append(STRUCTURED_OUTPUT_UNSUPPORTED_BY_PROVIDER)
                notes.append(f"provider_rejection: {sanitize_error(exc, cfg)}")
                request_kwargs.pop("response_format")
                structured_applied = False
                continue
            raise
    latency_ms = int((time.perf_counter() - started) * 1000)

    usage = _usage_of(response)
    applied_effort = request_kwargs.get("reasoning_effort")
    finish_reason = _finish_reason_of(response)
    if finish_reason == "length":
        notes.append(f"OUTPUT_TRUNCATED_AT_MAX_TOKENS={request_kwargs['max_tokens']}")
    return CompletionResult(
        text=_text_of(response),
        finish_reason=finish_reason,
        model=getattr(response, "model", cfg.model) or cfg.model,
        requested_reasoning_effort=requested_effort,
        applied_reasoning_effort=applied_effort,
        reasoning_effort_supported=effort_supported,
        structured_output_requested=json_object,
        structured_output_applied=structured_applied,
        latency_ms=latency_ms,
        notes=notes,
        **usage,
    )


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a single JSON object from a model reply."""
    if not text:
        return None
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if match:
            candidate = match.group(0)
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def provider_manifest_entry(cfg: ProviderConfig, result: Optional[CompletionResult]) -> Dict[str, Any]:
    """Secret-free provider block for a run manifest."""
    entry: Dict[str, Any] = {
        "provider": "openai_compatible",
        "sdk": "openai",
        "sdk_supports_reasoning_effort": sdk_supports_reasoning_effort(),
        **cfg.public_summary(),
    }
    if result is None:
        entry.update({
            "actual_reasoning_effort_applied": None,
            "applied_reasoning_mode": "unknown_no_successful_request",
            "reasoning_effort_supported_by_provider": None,
            "notes": [],
        })
        return entry
    entry.update({
        "actual_reasoning_effort_applied": result.applied_reasoning_effort,
        "applied_reasoning_mode": result.applied_reasoning_mode,
        "reasoning_effort_supported_by_provider": result.reasoning_effort_supported,
        "structured_output_applied": result.structured_output_applied,
        "notes": list(result.notes),
    })
    return entry
