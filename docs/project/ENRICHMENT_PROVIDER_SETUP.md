# Phase-3 Enrichment Provider — Local Setup

The enrichment stage calls an OpenAI-compatible gateway (adesso AI Hub) through the
official `openai` Python package. All environment-specific values, and **only** the
API key as a secret, come from environment variables. No key is ever stored in
code, YAML, JSON, Markdown, tests, logs or run manifests.

Non-secret defaults live in [`src/config/data/enrichment_provider.yaml`](../../src/config/data/enrichment_provider.yaml).

## Environment variables

| Variable | Purpose | Expected local value |
| --- | --- | --- |
| `ENRICHMENT_BASE_URL` | endpoint base URL | `https://adesso-ai-hub.3asabc.de/v1` |
| `ENRICHMENT_API_KEY` | API key (secret, never committed) | entered locally by the user |
| `ENRICHMENT_MODEL` | model id | `deepseek-v4-flash-sovereign` |
| `ENRICHMENT_REASONING_EFFORT` | requested reasoning effort | `xhigh` |

## Where to put the key

Two supported places — pick one:

| Place | Persistence | Notes |
| --- | --- | --- |
| PowerShell session (`$env:...`) | until the terminal closes | **preferred**: key never touches disk |
| `.env` at the project root | until edited | git-ignored (`.gitignore` line 2); real process environment variables always win over `.env` |

`.env` is not tracked by git and must never be committed. `.env.example` documents the
variable names with an empty key placeholder. Never place the key in
`src/config/data/enrichment_provider.yaml`, in Python, or in any doc.

### `.env` form

```
ENRICHMENT_BASE_URL=https://adesso-ai-hub.3asabc.de/v1
ENRICHMENT_API_KEY=<ENTER_NEW_ROTATED_KEY_HERE>
ENRICHMENT_MODEL=deepseek-v4-flash-sovereign
ENRICHMENT_REASONING_EFFORT=xhigh
```

## PowerShell (current session only)

```powershell
$env:ENRICHMENT_BASE_URL="https://adesso-ai-hub.3asabc.de/v1"
$env:ENRICHMENT_API_KEY="<ENTER_NEW_ROTATED_KEY_HERE>"
$env:ENRICHMENT_MODEL="deepseek-v4-flash-sovereign"
$env:ENRICHMENT_REASONING_EFFORT="xhigh"
```

These values disappear when the terminal closes. Nothing is written to disk.

## Optional: persistent Windows user variables

```powershell
setx ENRICHMENT_BASE_URL "https://adesso-ai-hub.3asabc.de/v1"
setx ENRICHMENT_API_KEY "<ENTER_NEW_ROTATED_KEY_HERE>"
setx ENRICHMENT_MODEL "deepseek-v4-flash-sovereign"
setx ENRICHMENT_REASONING_EFFORT "xhigh"
```

`setx` writes the user environment, so **a new terminal must be opened afterwards** —
the current session does not see the new values. Note that `setx` stores the key in
the Windows user environment (readable by any process running as this user); prefer
the session-only `$env:` form when in doubt.

Placeholders above are literal placeholders. Replace `<ENTER_NEW_ROTATED_KEY_HERE>`
in your own terminal only; never paste a key into this file, into source code, or
into a commit.

## Connection test

```bash
python experiments/scripts/test_enrichment_connection.py
```

Reports only: endpoint reachable, authentication successful, model accessible,
structured output supported, requested reasoning effort, actually applied reasoning
effort, latency, token usage when the provider returns it, and a sanitized error
message. The key and any authorization header are scrubbed from every line.

The secret-free manifest lands in
`data/staging/enrichment/connection_test/manifest.json` (under the git-ignored
`data/staging/` tree).

Status values: `MISSING_API_CREDENTIALS`, `CONNECTION_TEST_FAILED`,
`MODEL_NOT_ACCESSIBLE`, `STRUCTURED_OUTPUT_UNSUPPORTED`, `CONNECTION_TEST_PASSED`.

## Reasoning effort: capability-safe, never overclaimed

`xhigh` is a *request*, not an assumption about the gateway:

1. the request carries `reasoning_effort` only if the installed `openai` SDK accepts
   the parameter (SDK 1.79.0 does);
2. if the endpoint rejects it, `REASONING_EFFORT_UNSUPPORTED_BY_PROVIDER` is recorded
   and the request is retried **once** without the parameter, using the model's
   default reasoning mode;
3. the manifest stores `requested_reasoning_effort` and
   `actual_reasoning_effort_applied` separately — the latter is `null` and
   `applied_reasoning_mode` is `provider_default` whenever the provider did not
   confirm support, so no artifact can claim `xhigh` was used when it was not.

The same fallback logic applies to `response_format={"type": "json_object"}`
(`STRUCTURED_OUTPUT_UNSUPPORTED_BY_PROVIDER`).

## Data policy

The held-out nvBench test split (`test.jsonl`) has
`enrichment_target_use: "prohibited"` in `independent_evaluation_reference.json`,
and human-evaluation test items are never enrichment inputs. Enrichment runs may
read train/val records only.
