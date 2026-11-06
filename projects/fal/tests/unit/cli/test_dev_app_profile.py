import contextlib
import gzip
import json
import sys
import threading

import pytest
from fastapi.testclient import TestClient

import fal
from fal.dev.remote import profile as app_profile
from fal.dev.remote import proxy
from fal.dev.remote.child import DevAppServer

APP_SOURCE = """
from __future__ import annotations

import fal
from pydantic import BaseModel, Field


class Input(BaseModel):
    prompt: str = Field(examples=["a fox"])
    steps: int = 4


class Demo(fal.App):
    def setup(self):
        self.ready = True

    @fal.endpoint("/")
    def generate(self, input: Input) -> dict:
        return {"prompt": input.prompt, "steps": input.steps}

    @fal.endpoint("/async")
    async def generate_async(self, input: Input) -> dict:
        return {"prompt": input.prompt}

    @fal.endpoint("/ping")
    def ping(self) -> dict:
        return {"ok": True}
"""


@pytest.fixture
def server(tmp_path, monkeypatch):
    (tmp_path / "profile_app.py").write_text(APP_SOURCE)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(fal.ref, "current_app", None)
    captured = []

    @contextlib.contextmanager
    def capture(self, spec):
        captured.append((spec["path"], threading.get_ident()))
        yield
        self.finished = {"status": "ok", "id": spec["id"]}

    monkeypatch.setattr(app_profile.Profiler, "capture", capture)
    server = DevAppServer(tmp_path, "profile_app", "Demo")
    with TestClient(server) as client:
        yield server, client, captured, tmp_path
    sys.modules.pop("profile_app", None)


def _arm(client, tmp_path, path, profile_id="a" * 32):
    spec = {"id": profile_id, "path": path, "mode": "light", "dir": str(tmp_path)}
    return client.post("/__fal_dev_child/profile/arm", json=spec)


@pytest.mark.parametrize("path", ["/", "/async"])
def test_armed_profile_wraps_one_request(server, path):
    server, client, captured, tmp_path = server
    assert _arm(client, tmp_path, path).status_code == 200

    # A request to another endpoint does not claim it.
    assert client.post("/ping", json={}).json() == {"ok": True}
    assert captured == []

    response = client.post(path, json={"prompt": "a cat"})
    assert response.status_code == 200
    assert response.json()["prompt"] == "a cat"
    assert [entry[0] for entry in captured] == [path]

    # Only the first matching request is profiled.
    client.post(path, json={"prompt": "again"})
    assert len(captured) == 1
    result = client.post("/__fal_dev_child/profile/result", json={"id": "a" * 32})
    assert result.json()["status"] == "ok"


def test_profile_not_reached_reports_not_captured(server):
    server, client, captured, tmp_path = server
    _arm(client, tmp_path, "/")
    # A body that fails validation never reaches the endpoint.
    assert client.post("/", json={"steps": "x"}).status_code == 422
    result = client.post("/__fal_dev_child/profile/result", json={"id": "a" * 32})
    assert result.json()["status"] == "not_captured"
    assert captured == []


def test_wrapped_routes_survive_reload(server):
    server, client, captured, tmp_path = server
    result = client.post("/__fal_dev_child/reload", json={"files": []}).json()
    assert result["action"] == "reload"
    # The profiler wrapper must not leak into the instance state copied over.
    assert "collect_routes" not in vars(server.instance)
    _arm(client, tmp_path, "/")
    client.post("/", json={"prompt": "x"})
    assert [entry[0] for entry in captured] == ["/"]


def test_arm_rejects_unknown_mode(server):
    server, client, captured, tmp_path = server
    spec = {"id": "b" * 32, "path": "/", "mode": "huge", "dir": str(tmp_path)}
    assert client.post("/__fal_dev_child/profile/arm", json=spec).status_code == 422


def test_capture_writes_a_perfetto_trace(tmp_path):
    torch = pytest.importorskip("torch")
    profiler = app_profile.Profiler()
    spec = {"id": "c" * 32, "path": "/", "mode": "standard", "dir": str(tmp_path)}
    with profiler.capture(spec):
        torch.ones(64, 64) @ torch.ones(64, 64)
    status = profiler.result("c" * 32)
    assert status["status"] == "ok", status
    trace = json.loads(
        gzip.decompress((tmp_path / app_profile.TRACE_NAME).read_bytes())
    )
    assert trace["traceEvents"]
    assert "aten::mm" in status["top_ops"]


def test_profile_warms_up_arms_and_collects(tmp_path, monkeypatch):
    child_calls, forwarded = [], []

    def child_call(action, payload, timeout):
        child_calls.append(action)
        if action == "profile/arm":
            return 200, {"armed": payload["id"]}
        return 200, {"status": "ok", "trace_bytes": 10, "top_ops": "table"}

    def forward(request, timeout):
        forwarded.append(request["body"])
        return {"status_code": 200, "elapsed_seconds": 1.0, "body": "{}"}

    monkeypatch.setattr(proxy, "child_call", child_call)
    monkeypatch.setattr(proxy, "forward", forward)
    result = proxy.profile(
        {"path": "/", "body": {"prompt": "x"}, "warmups": 2}, tmp_path
    )

    assert child_calls == ["profile/arm", "profile/result"]
    # Two warmups, then the profiled request.
    assert forwarded == [{"prompt": "x"}] * 3
    assert result["status"] == "ok"
    assert result["request"]["status_code"] == 200


def test_profile_stops_at_a_failing_warmup(tmp_path, monkeypatch):
    monkeypatch.setattr(
        proxy,
        "forward",
        lambda request, timeout: {
            "status_code": 422,
            "elapsed_seconds": 0.1,
            "body": "bad input",
        },
    )
    result = proxy.profile({"path": "/", "body": {}}, tmp_path)
    assert result["status"] == "request_failed"
    assert result["request"]["body"] == "bad input"


def test_artifact_reads_chunks_and_refuses_other_files(tmp_path, monkeypatch):
    monkeypatch.setattr(proxy, "ARTIFACT_CHUNK_BYTES", 4)
    folder = tmp_path / ("d" * 32)
    folder.mkdir()
    (folder / "trace.json.gz").write_bytes(b"0123456789")
    chunk = proxy.artifact(
        {"id": "d" * 32, "name": "trace.json.gz", "offset": 4}, tmp_path
    )
    assert chunk["size"] == 10
    assert chunk["data"] == "NDU2Nw=="  # b"4567"
    for bad in ({"id": "../x", "name": "trace.json.gz"}, {"id": "d" * 32, "name": "x"}):
        with pytest.raises(ValueError):
            proxy.artifact(bad, tmp_path)


def _fake_profile_box(monkeypatch, trace_bytes):
    import base64

    from fal.dev.app import client

    trace = gzip.compress(json.dumps({"traceEvents": [{"name": "k"}]}).encode())
    files = {"trace.json.gz": trace, "ops.txt": b"ops table\n"}
    calls = []

    def call(_session, action, payload=None, **_):
        calls.append((action, payload))
        if action == "status":
            return {"phase": "ready", "module": "demo_app"}
        if action == "profile":
            return {
                "id": "e" * 32,
                "status": "ok",
                "elapsed_seconds": 2.5,
                "trace_bytes": trace_bytes or len(trace),
                "top_ops": "TOP OPS",
                "request": {"status_code": 200},
            }
        data = files[payload["name"]]
        piece = data[payload["offset"] : payload["offset"] + 7]
        return {"size": len(data), "data": base64.b64encode(piece).decode()}

    monkeypatch.setattr(client, "call", call)
    return calls, trace


def _profile_session(tmp_path, monkeypatch):
    from fal.dev.app import session as app_session

    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path / "sessions")
    app_session.Session(
        name="dev-demo-me",
        file_path="/src/demo_app.py",
        root="/src",
        module="demo_app",
        class_name="Demo",
        run_url="https://fal.run/me/dev-demo-me",
        token="t",
    ).save()


def test_profile_command_downloads_the_trace(tmp_path, monkeypatch, capsys):
    from fal.cli.main import parse_args

    _profile_session(tmp_path, monkeypatch)
    calls, trace = _fake_profile_box(monkeypatch, None)
    out = tmp_path / "profiles"
    args = parse_args(["dev", "app", "profile", "/", "--output-dir", str(out)])
    args.func(args)

    (folder,) = out.iterdir()
    assert (folder / "trace.json.gz").read_bytes() == trace
    assert (folder / "ops.txt").read_text() == "ops table\n"
    printed = capsys.readouterr().out
    assert "TOP OPS" in printed
    assert "ui.perfetto.dev" in printed
    # Without --data the request body is empty.
    (sent,) = (p for action, p in calls if action == "profile")
    assert sent["body"] == {}


def test_profile_command_sends_the_given_body(tmp_path, monkeypatch):
    from fal.cli.main import parse_args

    _profile_session(tmp_path, monkeypatch)
    calls, _ = _fake_profile_box(monkeypatch, None)
    args = parse_args(
        ["dev", "app", "profile", "/", "--data", '{"prompt": "x"}', "--light"]
        + ["--output-dir", str(tmp_path / "profiles")]
    )
    args.func(args)
    (sent,) = (p for action, p in calls if action == "profile")
    assert sent["body"] == {"prompt": "x"}
    assert sent["mode"] == "light"


def test_profile_command_skips_traces_over_the_cap(tmp_path, monkeypatch, capsys):
    from fal.cli.main import parse_args

    _profile_session(tmp_path, monkeypatch)
    calls, _ = _fake_profile_box(monkeypatch, 300 * 1024 * 1024)
    out = tmp_path / "profiles"
    args = parse_args(["dev", "app", "profile", "/", "--output-dir", str(out)])
    args.func(args)

    (folder,) = out.iterdir()
    assert not (folder / "trace.json.gz").exists()
    assert (folder / "ops.txt").exists()
    assert "Trace not downloaded" in capsys.readouterr().out


class _FakeProfile:
    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeProfile.instances.append(self)

    def start(self):
        pass


@pytest.mark.parametrize(
    "mode,cuda,activities,deep",
    [
        ("light", True, ["CUDA"], False),
        ("standard", True, ["CPU", "CUDA"], False),
        ("deep", True, ["CPU", "CUDA"], True),
        ("light", False, ["CPU"], False),
    ],
)
def test_profiler_modes(monkeypatch, mode, cuda, activities, deep):
    import types

    activity = types.SimpleNamespace(CPU="CPU", CUDA="CUDA")
    fake = types.SimpleNamespace(
        profiler=types.SimpleNamespace(ProfilerActivity=activity, profile=_FakeProfile),
        cuda=types.SimpleNamespace(
            is_available=lambda: cuda, reset_peak_memory_stats=lambda: None
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    profiler, has_cuda = app_profile._start(mode)
    assert has_cuda is cuda
    assert profiler.kwargs["activities"] == activities
    for flag in ("record_shapes", "profile_memory", "with_stack", "with_flops"):
        assert profiler.kwargs[flag] is deep


def test_profile_disarms_when_the_request_fails(tmp_path, monkeypatch):
    child_calls = []

    def child_call(action, payload, timeout):
        child_calls.append(action)
        return 200, {"armed": "x"} if action == "profile/arm" else {}

    def forward(request, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(proxy, "child_call", child_call)
    monkeypatch.setattr(proxy, "forward", forward)
    with pytest.raises(TimeoutError):
        proxy.profile({"path": "/", "body": {}, "warmups": 0}, tmp_path)
    assert child_calls == ["profile/arm", "profile/result"]


def test_profile_warmups_share_the_time_budget(tmp_path, monkeypatch):
    timeouts = []
    clock = iter([0.0, 100.0, 3400.0])
    monkeypatch.setattr(proxy.time, "monotonic", lambda: next(clock))

    def forward(request, timeout):
        timeouts.append(timeout)
        return {"status_code": 200, "elapsed_seconds": 1.0}

    monkeypatch.setattr(proxy, "forward", forward)
    with pytest.raises(ValueError, match="time budget"):
        proxy.profile({"path": "/", "body": {}, "warmups": 2}, tmp_path)
    assert timeouts == [proxy.RUNS_BUDGET_SECONDS - 100]


def test_capture_reports_a_profiler_that_cannot_start(tmp_path, monkeypatch):
    profiler = app_profile.Profiler()

    def broken(mode):
        raise RuntimeError("no torch")

    monkeypatch.setattr(app_profile, "_start", broken)
    spec = {"id": "f" * 32, "path": "/", "mode": "light", "dir": str(tmp_path / "p")}
    ran = []
    with profiler.capture(spec):
        ran.append(1)
    # The request still ran.
    assert ran == [1]
    assert profiler.result("f" * 32)["status"] == "error"


def test_arming_again_replaces_the_armed_profile():
    profiler = app_profile.Profiler()
    profiler.arm({"id": "1", "path": "/a", "mode": "light", "dir": "/tmp/x"})
    profiler.arm({"id": "2", "path": "/b", "mode": "light", "dir": "/tmp/x"})
    assert profiler.claim("/a") is None
    assert profiler.claim("/b")["id"] == "2"


def test_child_control_routes_reject_non_objects(server):
    server, client, captured, tmp_path = server
    response = client.post("/__fal_dev_child/status", content=b"[1]")
    assert response.status_code == 422
