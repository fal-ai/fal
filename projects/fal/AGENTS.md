# AGENTS.md (projects/fal)

Local guidance for changes under `projects/fal/`.

## Scope

Applies to:

- `projects/fal/src/fal/**`
- `projects/fal/tests/**`
- `projects/fal/openapi-fal-rest/**`
- `projects/fal/docs/**`

Follow root `/AGENTS.md` first; this file adds package-specific rules.

## Setup

From repo root:

```bash
pip install -e 'projects/fal[dev]'
```

## Fast validation loop

1. `pre-commit run --all-files` (**required**)
2. Run the smallest relevant tests:

```bash
pytest -n auto -v projects/fal/tests/unit
```

Integration/e2e require credentials:

```bash
FAL_KEY=... FAL_HOST=api.fal.dev FAL_RUN_HOST=run.fal.dev \
  pytest -n auto -v projects/fal/tests/integration

FAL_KEY=... FAL_GRPC_HOST=api.fal.dev FAL_RUN_HOST=run.fal.dev \
  pytest -n auto -v projects/fal/tests/e2e
```

If credentials are unavailable, run unit tests only and state that clearly.

## Generated code rules

- `projects/fal/openapi-fal-rest/` is generated OpenAPI client code.
- Do not hand-edit files in `openapi-fal-rest/`.
- Regenerate via `openapi-python-client` with `projects/fal/openapi_rest.config.yaml`.

## Important lint rule in this package

`pre-commit` runs an extra Ruff rule (`PLC0415`) that bans lazy imports in selected
serialized/runtime-critical files. If that hook fails, refactor to module-level imports
or follow existing patterns in the touched module.

## Test placement hints

- CLI changes: add/adjust tests under `projects/fal/tests/unit/cli/`
- Toolkit changes: add/adjust tests under `projects/fal/tests/unit/toolkit/`
- API/integration behavior: use `projects/fal/tests/integration/`
- End-to-end behavior across services: use `projects/fal/tests/e2e/`
