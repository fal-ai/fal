import pytest

from fal.cli.main import parse_args
from fal.cli.volumes import _create


def test_create():
    args = parse_args(["volumes", "create", "my-volume"])
    assert args.func == _create
    assert args.name == "my-volume"


class _FakeClient:
    instances: list = []

    def __init__(self, host=None, team=None):
        self.host, self.team = host, team
        _FakeClient.instances.append(self)

    def _create_host(self):
        return ("host", self.host, self.team)


@pytest.fixture
def fake_client(monkeypatch):
    import fal.cli.volumes

    _FakeClient.instances = []
    monkeypatch.setattr(fal.cli.volumes, "SyncServerlessClient", _FakeClient)
    return _FakeClient


def test_weights_pull_parses_defaults():
    from fal.cli.volumes import _weights_pull

    args = parse_args(["volumes", "weights", "pull", "Qwen/Qwen2.5-0.5B-Instruct"])
    assert args.func == _weights_pull
    assert args.repo == "Qwen/Qwen2.5-0.5B-Instruct"
    assert args.revision is None
    assert args.dest is None
    assert args.workers == 8
    assert args.machine_type == "M"


def test_weights_pull_forwards_options(monkeypatch, fake_client):
    import fal.weights

    calls = []
    monkeypatch.setattr(
        fal.weights,
        "pull_weights",
        lambda repo, **kw: (
            calls.append((repo, kw))
            or {"path": "/data/m", "bytes": 2**30, "seconds": 3}
        ),
    )
    args = parse_args(
        [
            "volumes", "weights", "pull", "org/model", "--revision", "v1",
            "--dest", "/data/models/m", "--workers", "16", "--machine-type", "L",
            "--team", "acme",
        ]
    )  # fmt: skip
    args.func(args)
    assert calls == [
        (
            "org/model",
            {
                "revision": "v1",
                "dest": "/data/models/m",
                "workers": 16,
                "machine_type": "L",
                "host": ("host", args.host, "acme"),
            },
        )
    ]


def test_weights_pull_rejects_dest_outside_data():
    from fal.cli.parser import FalParserExit

    with pytest.raises(FalParserExit):
        args = parse_args(["volumes", "weights", "pull", "org/m", "--dest", "/tmp/x"])
        args.func(args)


def test_validate_dest():
    from fal.weights import validate_dest

    assert validate_dest(None) is None
    assert validate_dest("/data/models/../models/m/") == "/data/models/m"
    for bad in ["/tmp/x", "data/x", "/data/../etc", "/database", "/data", "/data/"]:
        with pytest.raises(ValueError):
            validate_dest(bad)


def test_weights_pull_rejects_bad_workers():
    from fal.cli.parser import FalParserExit

    with pytest.raises(FalParserExit):
        args = parse_args(["volumes", "weights", "pull", "org/m", "--workers", "0"])
        args.func(args)


def _fake_hub(monkeypatch, snapshot_download):
    import sys
    import types

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)


@pytest.mark.parametrize(
    "dest,expected",
    [
        (None, {"cache_dir": "/data/.fal/hf/hub"}),
        ("/data/models/m", {"local_dir": "/data/models/m"}),
    ],
)
def test_download_uses_cache_or_local_dir(monkeypatch, tmp_path, dest, expected):
    import fal
    from fal import weights

    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        (tmp_path / "w.bin").write_bytes(b"x" * 10)
        return str(tmp_path)

    _fake_hub(monkeypatch, fake_snapshot_download)
    function_kwargs = []
    monkeypatch.setattr(
        fal, "function", lambda **kw: function_kwargs.append(kw) or (lambda fn: fn)
    )

    host = object()
    result = weights.pull_weights(
        "org/m", revision="v1", dest=dest, workers=4, machine_type="L", host=host
    )
    assert calls == [("org/m", {"revision": "v1", "max_workers": 4, **expected})]
    assert function_kwargs == [
        {
            "requirements": weights._REQUIREMENTS,
            "machine_type": "L",
            "keep_alive": 0,
            "request_timeout": weights.REQUEST_TIMEOUT_S,
            "host": host,
        }
    ]
    assert result["bytes"] == 10


def test_default_cache_matches_llm_app():
    from fal import weights

    assert weights.DEFAULT_HF_CACHE_DIR == weights.DEFAULT_HF_HOME + "/hub"
    from fal.toolkit.llm import _cache as llm_cache

    assert weights.DEFAULT_HF_HOME == str(llm_cache.HF_HOME_DIR)
    env = llm_cache.engine_cache_env()
    assert env["HF_HUB_CACHE"] == weights.DEFAULT_HF_CACHE_DIR


@pytest.mark.parametrize("error", ["GatedRepoError", "RepositoryNotFoundError"])
def test_download_reports_private_or_gated(monkeypatch, error):
    from fal import weights

    exc_type = type(error, (Exception,), {})

    def fake_snapshot_download(repo_id, **kwargs):
        raise exc_type("nope")

    _fake_hub(monkeypatch, fake_snapshot_download)
    with pytest.raises(RuntimeError, match="Only public repositories"):
        weights._download("org/private", None, None, 1)


def test_download_counts_snapshot_symlinks(monkeypatch, tmp_path):
    from fal import weights

    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshots" / "abc"
    blobs.mkdir()
    snapshot.mkdir(parents=True)
    (blobs / "b1").write_bytes(b"x" * 7)
    (snapshot / "model.safetensors").symlink_to(blobs / "b1")
    (snapshot / "dangling.bin").symlink_to(blobs / "missing")
    # Two names for one blob count once.
    (snapshot / "copy.safetensors").symlink_to(blobs / "b1")
    _fake_hub(monkeypatch, lambda repo_id, **kwargs: str(snapshot))

    result = weights._download("org/m", None, None, 1)
    assert result["path"] == str(snapshot)
    assert result["bytes"] == 7


@pytest.mark.parametrize("dest,hint", [(None, True), ("/data/models/m", False)])
def test_weights_pull_cache_hint(monkeypatch, fake_client, dest, hint):
    import fal.weights

    monkeypatch.setattr(
        fal.weights,
        "pull_weights",
        lambda repo, **kw: {
            "path": "/data/.fal/hf/hub/models--org--m/snapshots/abc123",
            "bytes": 0,
            "seconds": 0,
        },
    )
    argv = ["volumes", "weights", "pull", "org/m"]
    if dest:
        argv += ["--dest", dest]
    args = parse_args(argv)
    printed = []
    monkeypatch.setattr(args.console, "print", printed.append)
    args.func(args)
    hints = [msg for msg in printed if "HF_HUB_CACHE=/data/.fal/hf/hub" in msg]
    assert bool(hints) is hint
    if hint:
        assert "abc123" in hints[0]


def test_weights_pull_failure_suggests_rerun(monkeypatch, fake_client):
    import fal.weights

    def fail(repo, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(fal.weights, "pull_weights", fail)
    args = parse_args(["volumes", "weights", "pull", "org/m"])
    printed = []
    monkeypatch.setattr(args.console, "print", printed.append)
    with pytest.raises(RuntimeError, match="network down"):
        args.func(args)
    assert any("If it was interrupted" in msg for msg in printed)


def _fake_hub_with_info(monkeypatch, snapshot_download, sha, last_modified=None):
    import sys
    import types

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = snapshot_download

    class HfApi:
        def model_info(self, repo_id):
            return types.SimpleNamespace(sha=sha, last_modified=last_modified)

    fake_hub.HfApi = HfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)


def _snapshot(tmp_path, commit):
    snapshot = tmp_path / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    return str(snapshot)


def test_download_advances_marker_for_default_branch_head(monkeypatch, tmp_path):
    import datetime

    from fal import weights

    commit = "a" * 40
    path = _snapshot(tmp_path, commit)
    when = datetime.datetime(2026, 11, 1, tzinfo=datetime.timezone.utc)
    _fake_hub_with_info(monkeypatch, lambda repo_id, **kw: path, commit, when)
    calls = []

    def advance(model, path, commit, commit_date, *, hf_home=None):
        calls.append((model, path, commit, commit_date, hf_home))
        return True

    result = weights._download("org/m", None, None, 1, advance)
    assert calls == [("org/m", path, commit, when.isoformat(), weights.DEFAULT_HF_HOME)]
    assert result["marker_advanced"] is True


def test_download_skips_marker_when_not_default_branch_head(monkeypatch, tmp_path):
    from fal import weights

    path = _snapshot(tmp_path, "b" * 40)
    _fake_hub_with_info(monkeypatch, lambda repo_id, **kw: path, "c" * 40)
    calls = []

    result = weights._download("org/m", "dev", None, 1, lambda *a, **k: calls.append(a))
    assert calls == []
    assert result["marker_advanced"] is None


def test_download_skips_marker_for_plain_dest(monkeypatch, tmp_path):
    from fal import weights

    path = _snapshot(tmp_path, "d" * 40)
    _fake_hub_with_info(monkeypatch, lambda repo_id, **kw: path, "d" * 40)
    calls = []

    result = weights._download(
        "org/m", None, "/data/models/m", 1, lambda *a, **k: calls.append(a)
    )
    assert calls == []
    assert "marker_advanced" not in result


def test_download_marker_failure_keeps_pull(monkeypatch, tmp_path):
    from fal import weights

    commit = "e" * 40
    path = _snapshot(tmp_path, commit)
    _fake_hub_with_info(monkeypatch, lambda repo_id, **kw: path, commit)

    def advance(*args, **kwargs):
        raise OSError("read-only")

    result = weights._download("org/m", None, None, 1, advance)
    assert result["path"] == path
    assert "read-only" in result["marker_error"]


def test_pull_weights_ships_marker_helper_when_available(monkeypatch):
    import sys
    import types

    import fal
    from fal import weights

    def helper(*args, **kwargs):
        return True

    fake_cache = types.ModuleType("fal.toolkit.llm._cache")
    fake_cache.advance_snapshot_marker = helper
    monkeypatch.setitem(sys.modules, "fal.toolkit.llm._cache", fake_cache)
    monkeypatch.setattr(fal, "function", lambda **kw: lambda fn: lambda *a: a)

    args = weights.pull_weights("org/m")
    assert args[-1] is helper


@pytest.mark.parametrize(
    "extra,expected",
    [
        ({"marker_advanced": True}, "will serve this commit"),
        ({"marker_error": "OSError('x')"}, "Could not update"),
    ],
)
def test_weights_pull_reports_marker(monkeypatch, fake_client, extra, expected):
    import fal.weights

    monkeypatch.setattr(
        fal.weights,
        "pull_weights",
        lambda repo, **kw: {
            "path": "/data/.fal/hf/hub/models--org--m/snapshots/abc123",
            "bytes": 0,
            "seconds": 0,
            **extra,
        },
    )
    args = parse_args(["volumes", "weights", "pull", "org/m"])
    printed = []
    monkeypatch.setattr(args.console, "print", printed.append)
    args.func(args)
    assert any(expected in msg for msg in printed)
