"""Tests for the Phase-3 enrichment provider layer (offline, no network).

No real credential appears here: the fake key below is a literal test double.
"""

from __future__ import annotations

import pytest

from src.data_pipeline.enrichment_provider import (
    REASONING_EFFORT_UNSUPPORTED_BY_PROVIDER,
    STRUCTURED_OUTPUT_UNSUPPORTED_BY_PROVIDER,
    ProviderConfig,
    api_key_present,
    create_completion,
    is_unsupported_parameter_error,
    load_dotenv_values,
    load_provider_config,
    parse_json_object,
    provider_manifest_entry,
    resolve_env,
    sanitize_error,
    validate_config,
)

FAKE_KEY = "fake-test-key-0123456789"


def _cfg(**overrides) -> ProviderConfig:
    base = dict(
        base_url="https://adesso-ai-hub.3asabc.de",
        model="deepseek-v4-flash-sovereign",
        reasoning_effort="xhigh",
        temperature=0.1,
        max_tokens=1200,
        timeout_seconds=30.0,
        max_retries=0,
        api_key_env="ENRICHMENT_API_KEY",
    )
    base.update(overrides)
    return ProviderConfig(**base)


class _Usage:
    prompt_tokens = 11
    completion_tokens = 3
    total_tokens = 14


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content="ready", model="test-model", finish_reason="stop"):
        choice = _Choice(content)
        choice.finish_reason = finish_reason
        self.choices = [choice]
        self.model = model
        self.usage = _Usage()


class _BadRequest(Exception):
    status_code = 400


class _FakeCompletions:
    """Records calls; optionally rejects named parameters once each."""

    def __init__(self, reject=(), content="ready", finish_reason="stop"):
        self.reject = set(reject)
        self.content = content
        self.finish_reason = finish_reason
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        for param in list(self.reject):
            if param in kwargs:
                self.reject.discard(param)
                raise _BadRequest(f"Unsupported parameter: '{param}' is not supported by this model.")
        return _Response(self.content, finish_reason=self.finish_reason)


class _FakeClient:
    def __init__(self, completions):
        self.chat = type("_Chat", (), {"completions": completions})()


# ---------------------------------------------------------------- config


def test_env_vars_override_yaml_defaults(tmp_path):
    cfg_file = tmp_path / "provider.yaml"
    cfg_file.write_text(
        "api_key_env: ENRICHMENT_API_KEY\n"
        "base_url: https://yaml.invalid/v1\n"
        "model: yaml-model\n"
        "reasoning_effort: medium\n",
        encoding="utf-8",
    )
    env = {
        "ENRICHMENT_BASE_URL": "https://env.invalid/v1/",
        "ENRICHMENT_MODEL": "env-model",
        "ENRICHMENT_REASONING_EFFORT": "xhigh",
    }
    cfg = load_provider_config(cfg_file, env=env)
    assert cfg.base_url == "https://env.invalid/v1"
    assert cfg.model == "env-model"
    assert cfg.reasoning_effort == "xhigh"


def test_yaml_defaults_used_when_env_missing(tmp_path):
    cfg_file = tmp_path / "provider.yaml"
    cfg_file.write_text("base_url: https://yaml.invalid/v1\nmodel: yaml-model\nreasoning_effort: xhigh\n",
                        encoding="utf-8")
    cfg = load_provider_config(cfg_file, env={})
    assert (cfg.base_url, cfg.model, cfg.reasoning_effort) == ("https://yaml.invalid/v1", "yaml-model", "xhigh")


def test_empty_reasoning_effort_env_disables_parameter(tmp_path):
    cfg_file = tmp_path / "provider.yaml"
    cfg_file.write_text("base_url: https://yaml.invalid/v1\nmodel: m\nreasoning_effort: xhigh\n", encoding="utf-8")
    cfg = load_provider_config(cfg_file, env={"ENRICHMENT_REASONING_EFFORT": ""})
    assert cfg.reasoning_effort is None


def test_repository_config_has_no_secret_and_validates():
    cfg = load_provider_config(env={})
    assert validate_config(cfg) == []
    assert cfg.api_key_env == "ENRICHMENT_API_KEY"
    summary = provider_manifest_entry(cfg, None)
    assert FAKE_KEY not in str(summary)
    assert summary["api_key_value_recorded"] is False


def test_validate_config_reports_problems():
    problems = validate_config(_cfg(base_url="", model=""))
    assert any("base_url" in p for p in problems)
    assert any("model" in p for p in problems)


def test_api_key_presence_checks_only_presence():
    cfg = _cfg()
    assert api_key_present(cfg, env={}) is False
    assert api_key_present(cfg, env={"ENRICHMENT_API_KEY": "   "}) is False
    assert api_key_present(cfg, env={"ENRICHMENT_API_KEY": FAKE_KEY}) is True


# ------------------------------------------------------------ .env layer


def test_dotenv_parsing_handles_comments_quotes_and_export(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# comment line\n"
        "\n"
        'ENRICHMENT_MODEL="quoted-model"\n'
        "export ENRICHMENT_REASONING_EFFORT=high\n"
        f"ENRICHMENT_API_KEY={FAKE_KEY}\n"
        "MALFORMED_LINE\n",
        encoding="utf-8",
    )
    values = load_dotenv_values(dotenv)
    assert values["ENRICHMENT_MODEL"] == "quoted-model"
    assert values["ENRICHMENT_REASONING_EFFORT"] == "high"
    assert values["ENRICHMENT_API_KEY"] == FAKE_KEY
    assert "MALFORMED_LINE" not in values


def test_missing_dotenv_is_not_an_error(tmp_path):
    assert load_dotenv_values(tmp_path / "absent.env") == {}


def test_real_environment_wins_over_dotenv(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("ENRICHMENT_MODEL=from-dotenv\nENRICHMENT_API_KEY=dotenv-key-value\n", encoding="utf-8")
    monkeypatch.setenv("ENRICHMENT_MODEL", "from-process-env")
    monkeypatch.delenv("ENRICHMENT_API_KEY", raising=False)
    env = resolve_env(dotenv_path=dotenv)
    assert env["ENRICHMENT_MODEL"] == "from-process-env"
    assert env["ENRICHMENT_API_KEY"] == "dotenv-key-value"


def test_explicit_env_mapping_ignores_dotenv():
    assert resolve_env({}) == {}


def test_dotenv_key_satisfies_credential_check(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"ENRICHMENT_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    monkeypatch.delenv("ENRICHMENT_API_KEY", raising=False)
    monkeypatch.setattr("src.data_pipeline.enrichment_provider.DOTENV_PATH", dotenv)
    assert api_key_present(_cfg()) is True


# ------------------------------------------------------------ sanitizing


def test_sanitize_error_removes_key_bearer_and_header():
    cfg = _cfg()
    env = {"ENRICHMENT_API_KEY": FAKE_KEY}
    msg = f"401 error, Authorization: Bearer {FAKE_KEY} rejected (api_key={FAKE_KEY})"
    clean = sanitize_error(msg, cfg, env=env)
    assert FAKE_KEY not in clean
    assert "REDACTED" in clean


def test_sanitize_error_scrubs_openai_style_token():
    clean = sanitize_error("bad key sk-abcdef1234567890 used", _cfg(), env={})
    assert "sk-abcdef1234567890" not in clean


def test_sanitize_error_handles_exception_objects():
    clean = sanitize_error(ValueError("boom"), _cfg(), env={})
    assert clean.startswith("ValueError: boom")


# ------------------------------------------------- capability detection


@pytest.mark.parametrize("message", [
    "Unsupported parameter: 'reasoning_effort' is not supported",
    "unrecognized request argument supplied: reasoning_effort",
    "Extra inputs are not permitted: reasoning_effort",
])
def test_unsupported_parameter_detected(message):
    assert is_unsupported_parameter_error(_BadRequest(message), "reasoning_effort") is True


def test_unrelated_error_not_treated_as_unsupported_parameter():
    assert is_unsupported_parameter_error(_BadRequest("rate limit exceeded"), "reasoning_effort") is False
    assert is_unsupported_parameter_error(_BadRequest("context length exceeded"), "response_format") is False


def test_sdk_typeerror_detected():
    err = TypeError("create() got an unexpected keyword argument 'reasoning_effort'")
    assert is_unsupported_parameter_error(err, "reasoning_effort") is True


# ------------------------------------------------------------ completion


def test_requested_effort_sent_and_recorded_when_accepted():
    completions = _FakeCompletions()
    result = create_completion(_FakeClient(completions), _cfg(), [{"role": "user", "content": "hi"}])
    assert completions.calls[0]["reasoning_effort"] == "xhigh"
    assert completions.calls[0]["temperature"] == 0.1
    assert result.applied_reasoning_effort == "xhigh"
    assert result.reasoning_effort_supported is True
    assert result.applied_reasoning_mode == "xhigh"
    assert (result.prompt_tokens, result.completion_tokens) == (11, 3)


def test_unsupported_effort_falls_back_once_and_never_claims_xhigh():
    completions = _FakeCompletions(reject=("reasoning_effort",))
    result = create_completion(_FakeClient(completions), _cfg(), [{"role": "user", "content": "hi"}])
    assert len(completions.calls) == 2
    assert "reasoning_effort" not in completions.calls[1]
    assert REASONING_EFFORT_UNSUPPORTED_BY_PROVIDER in result.notes
    assert result.requested_reasoning_effort == "xhigh"
    assert result.applied_reasoning_effort is None
    assert result.reasoning_effort_supported is False
    assert result.applied_reasoning_mode == "provider_default"


def test_structured_output_falls_back_to_prompt_only():
    completions = _FakeCompletions(reject=("response_format",))
    result = create_completion(_FakeClient(completions), _cfg(), [{"role": "user", "content": "hi"}],
                               json_object=True)
    assert STRUCTURED_OUTPUT_UNSUPPORTED_BY_PROVIDER in result.notes
    assert result.structured_output_requested is True
    assert result.structured_output_applied is False


def test_unrelated_error_is_raised_not_swallowed():
    class _Boom(_FakeCompletions):
        def create(self, **kwargs):
            raise _BadRequest("rate limit exceeded")

    with pytest.raises(_BadRequest):
        create_completion(_FakeClient(_Boom()), _cfg(), [{"role": "user", "content": "hi"}])


def test_no_effort_requested_means_no_parameter_and_no_claim():
    completions = _FakeCompletions()
    result = create_completion(_FakeClient(completions), _cfg(reasoning_effort=None),
                               [{"role": "user", "content": "hi"}])
    assert "reasoning_effort" not in completions.calls[0]
    assert result.reasoning_effort_supported is None
    assert result.applied_reasoning_mode == "provider_default"


def test_manifest_entry_records_applied_mode_only():
    completions = _FakeCompletions(reject=("reasoning_effort",))
    cfg = _cfg()
    result = create_completion(_FakeClient(completions), cfg, [{"role": "user", "content": "hi"}])
    entry = provider_manifest_entry(cfg, result)
    assert entry["requested_reasoning_effort"] == "xhigh"
    assert entry["actual_reasoning_effort_applied"] is None
    assert entry["applied_reasoning_mode"] == "provider_default"
    assert entry["reasoning_effort_supported_by_provider"] is False


def test_truncation_is_reported_as_truncation_not_capability_loss():
    completions = _FakeCompletions(content='{"status": "o', finish_reason="length")
    result = create_completion(_FakeClient(completions), _cfg(), [{"role": "user", "content": "hi"}],
                               json_object=True, max_tokens=64)
    assert result.truncated is True
    assert result.finish_reason == "length"
    assert any(note.startswith("OUTPUT_TRUNCATED_AT_MAX_TOKENS=64") for note in result.notes)
    # response_format was accepted -- the failure is budget, not capability.
    assert result.structured_output_applied is True
    assert STRUCTURED_OUTPUT_UNSUPPORTED_BY_PROVIDER not in result.notes
    assert parse_json_object(result.text) is None


def test_complete_reply_is_not_flagged_as_truncated():
    completions = _FakeCompletions(content='{"status": "ok"}')
    result = create_completion(_FakeClient(completions), _cfg(), [{"role": "user", "content": "hi"}],
                               json_object=True)
    assert result.truncated is False
    assert result.notes == []
    assert parse_json_object(result.text) == {"status": "ok"}


# ----------------------------------------------------------------- JSON


@pytest.mark.parametrize("text,expected", [
    ('{"status": "ok"}', {"status": "ok"}),
    ('```json\n{"status": "ok"}\n```', {"status": "ok"}),
    ('here you go: {"status": "ok"} done', {"status": "ok"}),
    ("not json at all", None),
    ("[1, 2, 3]", None),
    ("", None),
])
def test_parse_json_object(text, expected):
    assert parse_json_object(text) == expected
