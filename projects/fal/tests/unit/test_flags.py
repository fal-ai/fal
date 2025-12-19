import importlib


def load_flags(monkeypatch, **env):
    for key in env.keys():
        monkeypatch.delenv(key, raising=False)

    for key, value in env.items():
        monkeypatch.setenv(key, value)

    import fal.flags as flags

    return importlib.reload(flags)


def test_defaults(monkeypatch):
    flags = load_flags(monkeypatch)

    assert flags.GRPC_HOST == "api.alpha.fal.ai"
    assert flags.REST_HOST == "rest.alpha.fal.ai"
    assert flags.FAL_RUN_HOST == "fal.run"
    assert flags.FAL_QUEUE_RUN_HOST == "queue.fal.run"
    assert flags.REST_URL == "https://rest.alpha.fal.ai"


def test_from_explicit_env(monkeypatch):
    flags = load_flags(
        monkeypatch,
        FAL_GRPC_HOST="custom.grpc",
        FAL_REST_HOST="custom.rest",
        FAL_RUN_HOST="custom.run",
        FAL_QUEUE_RUN_HOST="custom.queue",
        FAL_HOST="ignored.host",
    )

    assert flags.GRPC_HOST == "custom.grpc"
    assert flags.REST_HOST == "custom.rest"
    assert flags.FAL_RUN_HOST == "custom.run"
    assert flags.FAL_QUEUE_RUN_HOST == "custom.queue"


def test_from_dev_preview_fal_host(monkeypatch):
    flags = load_flags(
        monkeypatch,
        FAL_GRPC_HOST="api.pr5395.preview.fal.dev",
    )

    assert flags.GRPC_HOST == "api.pr5395.preview.fal.dev"
    assert flags.REST_HOST == "rest.pr5395.preview.fal.dev"
    assert flags.FAL_RUN_HOST == "pr5395.preview.run.fal.dev"
    assert flags.FAL_QUEUE_RUN_HOST == "pr5395.preview.queue.run.fal.dev"


def test_from_dev_fal_host(monkeypatch):
    flags = load_flags(monkeypatch, FAL_HOST="api.fal.dev")

    assert flags.GRPC_HOST == "api.fal.dev"
    assert flags.REST_HOST == "rest.fal.dev"
    assert flags.FAL_RUN_HOST == "run.fal.dev"
    assert flags.FAL_QUEUE_RUN_HOST == "queue.run.fal.dev"
