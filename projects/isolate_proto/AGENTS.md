# AGENTS.md (projects/isolate_proto)

Local guidance for changes under `projects/isolate_proto/`.

## Scope

Applies to:

- `projects/isolate_proto/src/isolate_proto/**`
- `projects/isolate_proto/tests/**`

Follow root `/AGENTS.md` first; this file adds package-specific rules.

## Setup

From repo root:

```bash
pip install -e 'projects/isolate_proto[dev]'
```

## Generated code policy (strict)

Files matching these patterns are generated and must not be hand-edited:

- `*_pb2.py`
- `*_pb2.pyi`
- `*_pb2_grpc.py`

Regenerate from `.proto` definitions instead.

## Regeneration workflow

From `projects/isolate_proto`:

```bash
pip install -e '.[dev]'
python ../../tools/regen_grpc.py --isolate-version <isolate-version>
pre-commit run --all-files
pytest tests
```

`<isolate-version>` is a tag from `fal-ai/isolate` without the leading `v`.

## Test guidance

- Keep `projects/isolate_proto/tests/test_proto.py` passing after any regeneration.
- When updating proto definitions, verify imports and package exports in
  `src/isolate_proto/__init__.py` and `src/isolate_proto/health/__init__.py`.
