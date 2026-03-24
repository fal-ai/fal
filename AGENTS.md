# AGENTS.md

Guidance for coding agents working in this repository.

## AGENTS convention in this monorepo

This repo uses a layered convention (typical for monorepos):

1. Root `AGENTS.md` (this file) defines global defaults.
2. Project-level `AGENTS.md` files under `projects/*` add local rules.
3. If rules conflict, prefer the `AGENTS.md` closest to the files being changed.

## Team preferences (non-negotiable)

1. **Pre-commit is king.** Run `pre-commit` before opening/finishing work.
2. **No secrets, no integration/e2e.** If credentials are unavailable, run unit/local tests only.
3. **Generated code is generated-only.**
   - Do not hand-edit protobuf outputs (`*_pb2.py`, `*_pb2.pyi`, `*_pb2_grpc.py`).
   - Do not hand-edit `projects/fal/openapi-fal-rest/` generated client files.
4. **Commit messages follow Conventional Commits** (`feat:`, `fix:`, `chore:`, etc.).

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
- OpenAPI client config: `projects/fal/openapi_rest.config.yaml`

## Environment setup

From repository root:

```bash
python -m pip install --upgrade pip
pip install -e 'projects/fal[dev]'
pip install -e 'projects/fal_client[dev]'
pip install -e 'projects/isolate_proto[dev]'
```

If changing only one package, install only that package (plus needed extras) to keep iteration fast.

## Allowed vs discouraged tooling

### Allowed / preferred

- `pre-commit run --all-files`
- `pytest ...` with smallest relevant scope first
- `make docs` for docs verification

### Discouraged

- Running ad-hoc formatters/linters while skipping `pre-commit`
- Manually editing generated code
- Running broad, slow test suites when targeted tests are sufficient

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

### `fal_client` package

```bash
pytest projects/fal_client/tests
```

### `isolate_proto` package

```bash
pytest projects/isolate_proto/tests
```

If secrets are unavailable, skip integration/e2e/cloud tests and run only unit/local tests.

## Generated code policy

### Protobuf / gRPC (`isolate_proto`)

When `.proto` definitions change, regenerate bindings instead of editing generated outputs by hand.

From `projects/isolate_proto`:

```bash
pip install -e '.[dev]'
python ../../tools/regen_grpc.py --isolate-version <isolate-version>
pre-commit run --all-files
```

`<isolate-version>` should match a tag from `fal-ai/isolate` without the leading `v`.

### OpenAPI REST client (`projects/fal/openapi-fal-rest`)

This directory is generated client code. Regenerate it with `openapi-python-client` using
`projects/fal/openapi_rest.config.yaml` instead of manual edits. Example pattern:

```bash
cd projects/fal
openapi-python-client generate --config openapi_rest.config.yaml --path <openapi-spec>
```

## Documentation build

Build both SDK and client docs from root:

```bash
make docs
```

This delegates to project-level docs builds and assembles output under `docs/_build/html`.

## Commit message format

Use Conventional Commits:

- `feat: ...`
- `fix: ...`
- `chore: ...`
- `docs: ...`
- `refactor: ...`
- `test: ...`
- `ci: ...`

Scoped form is encouraged when useful: `fix(cli): handle missing key`.

## Pre-merge checklist for agents

- [ ] `pre-commit run --all-files` passes.
- [ ] Relevant tests pass for touched areas.
- [ ] If secrets are unavailable, only unit/local tests were run and documented.
- [ ] Generated files were regenerated via tooling, not manually edited.
- [ ] Commit message follows Conventional Commits.
