# AGENTS.md

Guidance for coding agents working in this repository.

## Repository at a glance

This repo is a Python monorepo with three main packages:

| Project | Purpose | Source | Tests |
| --- | --- | --- | --- |
| `projects/fal` | Main `fal` SDK/CLI/runtime | `projects/fal/src/fal` | `projects/fal/tests/{unit,integration,e2e}` |
| `projects/fal_client` | Lightweight Python client | `projects/fal_client/src/fal_client` | `projects/fal_client/tests` |
| `projects/isolate_proto` | gRPC/protobuf definitions | `projects/isolate_proto/src/isolate_proto` | `projects/isolate_proto/tests` |

Other notable paths:

- Root docs build entrypoint: `Makefile`
- Shared hooks: `.pre-commit-config.yaml`
- CI workflows: `.github/workflows`
- gRPC regeneration tool: `tools/regen_grpc.py`

## Environment setup

From repository root:

```bash
python -m pip install --upgrade pip
pip install -e 'projects/fal[dev]'
pip install -e 'projects/fal_client[dev]'
pip install -e 'projects/isolate_proto[dev]'
```

If changing only one package, install only that package (plus needed extras) to keep iteration fast.

## Linting, formatting, and type checking

Primary path (matches CI behavior):

```bash
pre-commit run --all-files
```

Important hook scope details from `.pre-commit-config.yaml`:

- Ruff format/lint runs on:
  - `projects/fal/src/`
  - `projects/fal/tests/`
  - `projects/fal_client/`
- MyPy hook runs on:
  - `projects/fal/src/`
- A dedicated Ruff rule (`PLC0415`) bans lazy imports in specific serialized/runtime-sensitive files under `projects/fal/src/fal/`.

## Test commands

Run the smallest relevant tests first, then broaden as needed.

### `fal` package

```bash
# Unit
pytest -n auto -v projects/fal/tests/unit

# Integration (requires credentials + network)
FAL_KEY=... FAL_HOST=api.fal.dev FAL_RUN_HOST=run.fal.dev \
  pytest -n auto -v projects/fal/tests/integration

# E2E (requires credentials + network)
FAL_KEY=... FAL_GRPC_HOST=api.fal.dev FAL_RUN_HOST=run.fal.dev \
  pytest -n auto -v projects/fal/tests/e2e
```

CI runs `fal` tests across Python `3.8`-`3.13` and both `pydantic==1.10.18` / `pydantic==2.11.7` (with matrix exclusions).

### `fal_client` package

```bash
pytest projects/fal_client/tests
```

### `isolate_proto` package

```bash
pytest projects/isolate_proto/tests
```

## Documentation build

Build both SDK and client docs from root:

```bash
make docs
```

This delegates to project-level docs builds and assembles output under `docs/_build/html`.

## Regenerating protobuf/gRPC bindings (`isolate_proto`)

When `.proto` definitions change, regenerate bindings instead of editing generated `*_pb2*.py*` files by hand.

From `projects/isolate_proto`:

```bash
pip install -e '.[dev]'
python ../../tools/regen_grpc.py --isolate-version <isolate-version>
pre-commit run --all-files
```

`<isolate-version>` should match a tag from `fal-ai/isolate` without the leading `v`.

## Agent workflow recommendations

1. Scope changes to the relevant package(s); avoid cross-package edits unless required.
2. Prefer minimal, targeted tests for touched code paths before wider suites.
3. Keep compatibility expectations in mind:
   - Python `>=3.8`
   - `fal` supports both Pydantic v1/v2 via CI matrix
4. Do not hand-edit generated artifacts when a generation script exists (`isolate_proto` bindings).
5. Follow Conventional Commits for commit messages (as requested in project README).

## Pre-merge checklist for agents

- [ ] Code is formatted/linted (`pre-commit run --all-files`).
- [ ] Relevant tests pass for touched areas.
- [ ] Any required env vars/secrets were used for integration/e2e validation.
- [ ] Generated files were regenerated via tooling, not manually edited.
- [ ] Commit message follows Conventional Commits.
