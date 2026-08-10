"""Connection test for the Phase-3 enrichment provider (one minimal request).

Runs, in this order:
  1. configuration validation (src/config/data/enrichment_provider.yaml);
  2. environment-variable validation (presence only -- never the key value);
  3. one minimal chat-completion request;
  4. one structured-JSON capability request.

Reports only: endpoint reachable, authentication successful, model accessible,
structured output supported, requested reasoning effort, actual reasoning effort
applied, latency, token usage when returned, sanitized error message. The API key
and any authorization header are never printed or written.

Exit code 0 on a passing connection test, 1 otherwise. The status line is one of
MISSING_API_CREDENTIALS, CONNECTION_TEST_FAILED, MODEL_NOT_ACCESSIBLE,
STRUCTURED_OUTPUT_UNSUPPORTED or CONNECTION_TEST_PASSED (the latter is an
intermediate gate, not a final pipeline status).

Usage:
    python experiments/scripts/test_enrichment_connection.py
    python experiments/scripts/test_enrichment_connection.py --out data/staging/enrichment/connection_test
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.enrichment_provider import (  # noqa: E402
    DOTENV_PATH,
    CompletionResult,
    MissingCredentialsError,
    api_key_present,
    build_client,
    create_completion,
    load_provider_config,
    parse_json_object,
    provider_manifest_entry,
    resolve_env,
    sanitize_error,
    validate_config,
)
from src.utils.io import write_json  # noqa: E402

DEFAULT_OUT = "data/staging/enrichment/connection_test"

# Output budget for the two probes. A reasoning model spends part of max_tokens on
# reasoning tokens, so a tight cap truncates the answer and the JSON probe would
# fail for the wrong reason. Retried once at double budget on truncation.
PROBE_MAX_TOKENS = 512
JSON_PROBE_MAX_TOKENS = 1024

ENV_VARS = (
    "ENRICHMENT_BASE_URL",
    "ENRICHMENT_API_KEY",
    "ENRICHMENT_MODEL",
    "ENRICHMENT_REASONING_EFFORT",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-3 enrichment provider connection test")
    p.add_argument("--config", default=None, help="provider config YAML (default: src/config/data/enrichment_provider.yaml)")
    p.add_argument("--out", default=DEFAULT_OUT, help="report/manifest output directory")
    p.add_argument("--no-write", action="store_true", help="print the report only; write nothing")
    return p.parse_args()


def _yn(value: Optional[bool]) -> str:
    if value is None:
        return "not tested"
    return "yes" if value else "no"


def _model_not_accessible(exc: BaseException) -> bool:
    text = str(exc).lower()
    status = getattr(exc, "status_code", None)
    if status is not None and int(status) == 404:
        return True
    return "model" in text and any(m in text for m in
                                   ("not found", "does not exist", "unknown", "no such", "not available"))


def _auth_failed(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status is not None and int(status) in (401, 403):
        return True
    text = str(exc).lower()
    return any(m in text for m in ("unauthorized", "forbidden", "invalid api key", "authentication"))


def main() -> int:  # noqa: C901 - linear report script
    args = parse_args()
    cfg = load_provider_config(args.config)

    report: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": cfg.config_path,
        "endpoint_reachable": None,
        "authentication_successful": None,
        "model_accessible": None,
        "structured_output_supported": None,
        "requested_reasoning_effort": cfg.reasoning_effort,
        "actual_reasoning_effort_applied": None,
        "latency_ms": None,
        "input_tokens": None,
        "output_tokens": None,
        "error": None,
        "status": None,
    }
    result: Optional[CompletionResult] = None

    # ---- 1. configuration validation ------------------------------------
    problems = validate_config(cfg)
    if problems:
        report["error"] = "config invalid: " + "; ".join(problems)
        report["status"] = "CONNECTION_TEST_FAILED"
        return _finish(args, cfg, report, result)

    # ---- 2. environment-variable validation (presence only) -------------
    env = resolve_env()
    report["env_vars_set"] = {name: bool((env.get(name) or "").strip()) for name in ENV_VARS}
    report["dotenv_present"] = DOTENV_PATH.exists()
    if not api_key_present(cfg):
        report["error"] = (f"{cfg.api_key_env} is not set in the environment or .env "
                           "(see docs/project/ENRICHMENT_PROVIDER_SETUP.md)")
        report["status"] = "MISSING_API_CREDENTIALS"
        return _finish(args, cfg, report, result)

    # ---- 3. one minimal API request -------------------------------------
    messages = [
        {"role": "system", "content": "Reply with exactly one word."},
        {"role": "user", "content": "Reply with the single word: ready"},
    ]
    try:
        client = build_client(cfg)
        # Reasoning tokens count against max_tokens, so the probe gets real headroom;
        # a truncated probe would otherwise look like a provider capability failure.
        result = create_completion(client, cfg, messages, max_tokens=PROBE_MAX_TOKENS)
    except MissingCredentialsError as exc:
        report["error"] = sanitize_error(exc, cfg)
        report["status"] = "MISSING_API_CREDENTIALS"
        return _finish(args, cfg, report, result)
    except Exception as exc:  # noqa: BLE001 - reported, never raised with secrets
        report["error"] = sanitize_error(exc, cfg)
        if _model_not_accessible(exc):
            report.update({"endpoint_reachable": True, "authentication_successful": True,
                           "model_accessible": False, "status": "MODEL_NOT_ACCESSIBLE"})
        elif _auth_failed(exc):
            report.update({"endpoint_reachable": True, "authentication_successful": False,
                           "status": "CONNECTION_TEST_FAILED"})
        else:
            report.update({"endpoint_reachable": False, "status": "CONNECTION_TEST_FAILED"})
        return _finish(args, cfg, report, result)

    report.update({
        "endpoint_reachable": True,
        "authentication_successful": True,
        "model_accessible": True,
        "actual_reasoning_effort_applied": result.applied_reasoning_effort,
        "applied_reasoning_mode": result.applied_reasoning_mode,
        "reasoning_effort_supported_by_provider": result.reasoning_effort_supported,
        "latency_ms": result.latency_ms,
        "input_tokens": result.prompt_tokens,
        "output_tokens": result.completion_tokens,
        "notes": list(result.notes),
        "resolved_model": result.model,
    })

    # ---- 4. structured JSON capability test ------------------------------
    json_messages = [
        {"role": "system", "content": "You return only valid JSON objects."},
        {"role": "user", "content": 'Return this JSON object exactly: {"status": "ok", "n": 1}'},
    ]
    budget = JSON_PROBE_MAX_TOKENS
    try:
        json_result = create_completion(client, cfg, json_messages, json_object=True, max_tokens=budget)
        parsed = parse_json_object(json_result.text)
        if parsed is None and json_result.truncated:
            # Truncation is a budget problem, not a capability problem: retry once
            # with double headroom before concluding anything about the provider.
            budget *= 2
            json_result = create_completion(client, cfg, json_messages, json_object=True, max_tokens=budget)
            parsed = parse_json_object(json_result.text)
    except Exception as exc:  # noqa: BLE001
        report["structured_output_supported"] = False
        report["error"] = sanitize_error(exc, cfg)
        report["status"] = "STRUCTURED_OUTPUT_UNSUPPORTED"
        return _finish(args, cfg, report, result)

    report["json_probe_max_tokens"] = budget
    report["json_probe_finish_reason"] = json_result.finish_reason
    report["structured_output_supported"] = parsed is not None
    report["structured_output_mode"] = ("response_format=json_object"
                                        if json_result.structured_output_applied
                                        else "prompt_only_json (response_format rejected)")
    report["notes"] = list(dict.fromkeys(list(report.get("notes") or []) + list(json_result.notes)))
    report["latency_ms_json_request"] = json_result.latency_ms
    if json_result.prompt_tokens is not None:
        report["input_tokens_json_request"] = json_result.prompt_tokens
        report["output_tokens_json_request"] = json_result.completion_tokens

    if parsed is None:
        report["error"] = (
            f"JSON probe reply still truncated at max_tokens={budget} "
            "(raise max_tokens in src/config/data/enrichment_provider.yaml)"
            if json_result.truncated
            else f"provider reply was not a parsable JSON object (finish_reason={json_result.finish_reason})"
        )
        report["status"] = "STRUCTURED_OUTPUT_UNSUPPORTED"
        return _finish(args, cfg, report, result)

    report["status"] = "CONNECTION_TEST_PASSED"
    return _finish(args, cfg, report, result)


def _finish(args: argparse.Namespace, cfg, report: Dict[str, Any],
            result: Optional[CompletionResult]) -> int:
    _print_report(report)
    if not args.no_write:
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = _PROJECT_ROOT / out_dir
        manifest = {
            "stage": "phase3_enrichment_connection_test",
            "timestamp_utc": report["timestamp_utc"],
            "status": report["status"],
            "provider": provider_manifest_entry(cfg, result),
            "connection_test": report,
        }
        write_json(manifest, out_dir / "manifest.json")
        print(f"  manifest      : {out_dir / 'manifest.json'}")
    return 0 if report["status"] == "CONNECTION_TEST_PASSED" else 1


def _print_report(report: Dict[str, Any]) -> None:
    print("=" * 60)
    print("PHASE-3 ENRICHMENT PROVIDER - CONNECTION TEST")
    print("=" * 60)
    print(f"  endpoint reachable          : {_yn(report['endpoint_reachable'])}")
    print(f"  authentication successful   : {_yn(report['authentication_successful'])}")
    print(f"  model accessible            : {_yn(report['model_accessible'])}")
    print(f"  structured output supported : {_yn(report['structured_output_supported'])}")
    print(f"  requested reasoning effort  : {report['requested_reasoning_effort'] or '(none)'}")
    print(f"  actual reasoning effort     : {report.get('applied_reasoning_mode') or 'not applied'}")
    latency = report["latency_ms"]
    print(f"  latency                     : {latency} ms" if latency is not None else "  latency                     : n/a")
    tokens_in, tokens_out = report["input_tokens"], report["output_tokens"]
    if tokens_in is not None or tokens_out is not None:
        print(f"  token usage                 : in={tokens_in} out={tokens_out}")
    else:
        print("  token usage                 : not returned by provider")
    for note in report.get("notes") or []:
        print(f"  note                        : {note}")
    if report.get("error"):
        print(f"  error                       : {report['error']}")
    print(f"  status                      : {report['status']}")
    print("=" * 60)


if __name__ == "__main__":
    raise SystemExit(main())
