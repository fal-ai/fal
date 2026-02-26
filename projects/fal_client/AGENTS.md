# AGENTS.md (projects/fal_client)

Local guidance for changes under `projects/fal_client/`.

## Scope

Applies to:

- `projects/fal_client/src/fal_client/**`
- `projects/fal_client/tests/**`
- `projects/fal_client/docs/**`

Follow root `/AGENTS.md` first; this file adds package-specific rules.

## Setup

From repo root:

```bash
pip install -e 'projects/fal_client[dev]'
```

## Validation workflow

Run in this order:

```bash
pre-commit run --all-files
pytest projects/fal_client/tests
```

If cloud credentials are unavailable, prefer local/unit tests:

```bash
pytest projects/fal_client/tests/unit
```

## Test guidance

- `tests/unit/` should cover parsing, headers, retries, and client internals.
- Top-level `tests/test_sync_client.py` and `tests/test_async_client.py` include cloud-facing behavior and may skip when credentials are missing.
- For request/response contract changes, add tests for both sync and async clients when relevant.
