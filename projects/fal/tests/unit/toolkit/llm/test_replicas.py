"""Engine replicas: several engines in one runner behind the relay."""

from __future__ import annotations

import asyncio
import json
import random
import socket
import subprocess
import sys
import threading
import time
import types
from collections import OrderedDict
from pathlib import Path

import httpx
import pytest
import uvicorn

from fal.toolkit.llm import LLMApp, _app
from fal.toolkit.llm._app import _Replica, _split_devices, _with_port

FAKE = str(Path(__file__).parent / "fake_engine.py")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _free_port_pair() -> int:
    # Replicas listen on engine_port + i. Drawn below the ephemeral range, so
    # a port handed out by _free_port (here or in a parallel worker) cannot
    # land on the second one.
    for _ in range(100):
        port = random.randrange(20000, 40000)
        try:
            for p in (port, port + 1):
                with socket.socket() as s:
                    s.bind(("127.0.0.1", p))
        except OSError:
            continue
        return port
    raise RuntimeError("no free port pair")


def _wait_for(path: Path, timeout: float = 5.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if path.exists():
            return True
        time.sleep(0.1)
    return False


def _wait_until(check, timeout: float = 5.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if check():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture(autouse=True)
def isolate_agent_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IS_ISOLATE_AGENT", "1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


@pytest.fixture
def served(tmp_path, monkeypatch, request):
    abort_file = tmp_path / "abort"
    monkeypatch.setenv("FAKE_ENGINE_ABORT_FILE", str(abort_file))
    shard_dir = tmp_path / "model"
    shard_dir.mkdir()
    (shard_dir / "m.safetensors").write_bytes(b"0" * 4096)
    port = _free_port()

    class Replicated(LLMApp):
        model = str(shard_dir)
        num_gpus = 2
        engine_replicas = 2
        engine_port = _free_port_pair()
        persistent_caches = False
        preread_workers = 0
        engine_startup_timeout_s = 30.0
        locals().update(getattr(request, "param", {}))

        def engine_command(self, model_path: str) -> list[str]:
            return [sys.executable, FAKE, "--port", str(self.engine_port)]

    app = Replicated()
    server = uvicorn.Server(
        uvicorn.Config(
            app._build_app(), host="127.0.0.1", port=port, log_level="warning"
        )
    )
    loop = asyncio.new_event_loop()
    app._test_loop = loop  # type: ignore[attr-defined]
    thread = threading.Thread(
        target=lambda: loop.run_until_complete(server.serve()), daemon=True
    )
    thread.start()
    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.1)
    assert server.started
    yield app, f"http://127.0.0.1:{port}", abort_file
    server.should_exit = True
    thread.join(timeout=15)
    app.teardown()


def _abort_file(abort_file: Path, replica: _Replica) -> Path:
    return abort_file.with_name(f"{abort_file.name}.{replica.port}")


def _router(n: int) -> LLMApp:
    """An app with ``n`` replicas and no engines, for routing decisions."""

    class Router(LLMApp):
        model = "m"
        num_gpus = n
        engine_replicas = n

    app = Router()
    app._replicas = [_Replica(app, i, 9000 + i) for i in range(n)]
    app._sessions = OrderedDict()
    app._response_routes = OrderedDict()
    return app


def test_sticky_sessions_and_least_inflight():
    app = _router(2)
    r0, r1 = app._replicas
    # Idle replicas tie: the first one wins, and the session remembers it.
    assert app._pick_replica("a") is r0
    r0.inflight = 3
    # A new session goes to the least loaded replica...
    assert app._pick_replica("b") is r1
    # ...and an old one stays put, even on the busier replica.
    assert app._pick_replica("a") is r0
    r1.inflight = 5
    assert app._pick_replica("a") is r0
    assert app._pick_replica("b") is r1
    # No session id: least in flight, nothing remembered.
    assert app._pick_replica(None) is r0
    assert len(app._sessions) == 2


def test_hot_session_moves_off_an_overloaded_replica():
    app = _router(2)
    r0, r1 = app._replicas
    assert app._pick_replica("hot") is r0
    # Within the least load plus max(slack, least load), it stays.
    r0.inflight, r1.inflight = 6, 2
    assert app._pick_replica("hot") is r0
    # Past it, the session moves and stays on its new replica.
    r0.inflight = 7
    assert app._pick_replica("hot") is r1
    r1.inflight = 5
    assert app._pick_replica("hot") is r1


_PICKLE_APP = """
import cloudpickle
from fal._serialization import include_module
from fal.toolkit.llm import LLMApp

include_module("fal")

class App(LLMApp):
    model = "m"
    num_gpus = 2
    engine_replicas = 2

cloudpickle.loads(cloudpickle.dumps(App))
"""


def test_app_class_pickles_by_value():
    # fal ships its own package by value, so every module global an LLMApp
    # method reads must pickle. A subprocess keeps that setting out of the
    # other tests.
    subprocess.run([sys.executable, "-c", _PICKLE_APP], check=True)


def test_session_routes_are_bounded(monkeypatch):
    monkeypatch.setattr(_app, "_SESSION_ROUTES_MAX", 3)
    app = _router(2)
    for i in range(10):
        app._pick_replica(f"s{i}")
    assert len(app._sessions) == 3


def test_response_routes_are_bounded(monkeypatch):
    monkeypatch.setattr(_app, "_RESPONSE_ROUTES_MAX", 3)
    app = _router(2)
    for i in range(10):
        app._note_response(f"resp_{i}", app._replicas[i % 2])
    assert list(app._response_routes) == ["resp_7", "resp_8", "resp_9"]
    payload = {"previous_response_id": "resp_9"}
    assert app._continuation_replica("/v1/responses", payload) is app._replicas[1]
    assert app._continuation_replica("/v1/chat/completions", payload) is None
    assert app._continuation_replica("/v1/responses", {}) is None
    expired = {"previous_response_id": "resp_0"}
    assert app._continuation_replica("/v1/responses", expired) is None
    # An engine that exited lost the conversation: route as usual.
    app._replicas[1].proc = types.SimpleNamespace(poll=lambda: 1)
    assert app._continuation_replica("/v1/responses", payload) is None
    assert "resp_9" not in app._response_routes


def test_unhealthy_replica_is_routed_around():
    app = _router(2)
    r0, r1 = app._replicas
    assert app._pick_replica("a") is r0
    r0.healthy = False
    # The session moves to a usable replica and stays there.
    assert app._pick_replica("a") is r1
    r0.healthy = True
    assert app._pick_replica("a") is r1
    # With none usable, requests still go somewhere (the health check is
    # already failing, and the engine's reply says what happened).
    r0.healthy = r1.healthy = False
    assert app._pick_replica("b") in (r0, r1)


def test_single_replica_needs_no_routing_state():
    class One(LLMApp):
        model = "m"

    app = One()
    app._replicas = [_Replica(app, 0, 9000)]
    assert app._pick_replica("a") is app._replicas[0]
    assert not hasattr(app, "_sessions")


def test_replicas_must_split_the_gpus():
    with pytest.raises(ValueError, match="positive integer"):

        class Zero(LLMApp):
            model = "m"
            engine_replicas = 0

    with pytest.raises(ValueError, match="does not split"):

        class Uneven(LLMApp):
            model = "m"
            num_gpus = 4
            engine_replicas = 3

    class Even(LLMApp):
        model = "m"
        num_gpus = 8
        engine_replicas = 2
        engine_presets = False

    # Presets and recipes see one engine's GPUs.
    Even._gpu_name = staticmethod(lambda: None)  # type: ignore[method-assign]
    Even._gpu_type = staticmethod(lambda: "RTX")  # type: ignore[method-assign]
    assert Even()._machine().num_gpus == 4


def test_split_devices_and_ports():
    assert _split_devices(None, 8, 2) == ["0,1,2,3", "4,5,6,7"]
    assert _split_devices("3, 5,6,7", 4, 2) == ["3,5", "6,7"]
    assert _split_devices("GPU-a,GPU-b", 2, 2) == ["GPU-a", "GPU-b"]
    with pytest.raises(ValueError, match="cannot split"):
        _split_devices("0,1,2", 4, 2)
    assert _with_port(["serve", "--port", "8000", "-x"], 8001) == [
        "serve",
        "--port",
        "8001",
        "-x",
    ]
    assert _with_port(["serve", "--port=8000"], 8002) == ["serve", "--port=8002"]
    with pytest.raises(ValueError, match="--port"):
        _with_port(["serve"], 8001)


def test_replicas_get_own_port_and_gpus(served):
    app, _, _ = served
    seen = [httpx.get(f"{r.url}/whoami").json() for r in app._replicas]
    assert seen == [
        {"port": app.engine_port, "gpus": "0"},
        {"port": app.engine_port + 1, "gpus": "1"},
    ]
    assert app._boot["engine_replicas"] == 2


def test_requests_follow_their_session(served):
    app, url, _ = served
    r0, r1 = app._replicas
    # Hold a stream on the first replica so the next session lands on the
    # second one.
    with httpx.stream(
        "POST",
        f"{url}/v1/chat/completions",
        json={"stream": True, "n_chunks": 20},
        headers={"X-Fal-Runner-Hint": "busy"},
    ) as r:
        lines = r.iter_lines()
        next(lines)
        assert (r0.inflight, r1.inflight) == (1, 0)
        for _ in range(3):
            resp = httpx.post(
                f"{url}/v1/chat/completions", json={"prompt_cache_key": "other"}
            )
            assert resp.status_code == 200
        assert (r0.requests, r1.requests) == (1, 3)
        list(lines)
    assert _wait_until(lambda: (r0.inflight, r1.inflight) == (0, 0))
    # Idle again: the session keeps its replica even though both are free.
    httpx.post(f"{url}/v1/chat/completions", json={"user": "other"})
    assert r1.requests == 4


@pytest.mark.parametrize("stream", [False, True])
def test_responses_continuation_stays_on_the_creating_engine(served, stream):
    app, url, _ = served
    r0, r1 = app._replicas

    def create(**body):
        r = httpx.post(
            f"{url}/v1/responses",
            json={"stream": stream, **body},
            headers={"X-Fal-Runner-Hint": "conv"},
        )
        assert r.status_code == 200, r.text
        if not stream:
            return r.json()["id"]
        events = [
            json.loads(line[5:])
            for line in r.text.splitlines()
            if line.startswith("data:")
        ]
        return events[-1]["response"]["id"]

    first = create()
    assert (r0.requests, r1.requests) == (1, 0)
    # Load the creating replica past the sticky slack, so the session alone
    # would move the continuation to the other engine.
    r0.inflight += 50
    try:
        second = create(previous_response_id=first)
        third = create(previous_response_id=second)
    finally:
        r0.inflight -= 50
    assert (r0.requests, r1.requests) == (3, 0)
    assert third.startswith(f"resp_{r0.port}_")
    # An id this runner never saw is routed as usual: by load, to r1.
    r0.inflight += 50
    try:
        r = httpx.post(
            f"{url}/v1/responses",
            json={"stream": stream, "previous_response_id": "resp_unknown"},
        )
    finally:
        r0.inflight -= 50
    assert r.status_code == 404
    assert (r0.requests, r1.requests) == (3, 1)


def test_abort_reaches_the_routed_replica(served, monkeypatch):
    app, url, abort_file = served
    r0, r1 = app._replicas
    monkeypatch.setattr(app, "_pick_replica", lambda session: r1)
    with httpx.stream(
        "POST", f"{url}/v1/chat/completions", json={"stream": True, "n_chunks": 100}
    ) as r:
        next(r.iter_lines())
    marker = _abort_file(abort_file, r1)
    assert _wait_for(marker)
    assert _wait_until(lambda: r1.inflight == 0)
    with pytest.raises(httpx.ReadTimeout):
        httpx.post(f"{url}/v1/chat/completions", json={"delay_ticks": 300}, timeout=1)
    assert _wait_until(lambda: marker.read_text().count("abort") == 2, timeout=8.0)
    assert not _abort_file(abort_file, r0).exists()
    assert _wait_until(lambda: r1.inflight == 0)
    assert (r0.requests, r1.requests) == (0, 2)


def test_health_fails_when_one_replica_dies_and_routing_avoids_it(served):
    app, url, _ = served
    r0, r1 = app._replicas
    h = httpx.post(f"{url}/engine-health").json()
    # Counts add up across replicas; fractions average.
    assert h["engine"]["vllm:num_requests_running"] == 4.0
    assert h["engine"]["vllm:kv_cache_usage_perc"] == 0.25
    assert h["engine"]["spec_decode_acceptance_rate"] == 0.75
    assert [x["index"] for x in h["replicas"]] == [0, 1]

    r1.proc.kill()
    r1.proc.wait()
    r = httpx.post(f"{url}/engine-health")
    assert r.status_code == 503
    assert "vllm replica 1 exited" in r.json()["detail"]
    for session in ("a", "b", "c"):
        resp = httpx.post(f"{url}/v1/chat/completions", json={"user": session})
        assert resp.status_code == 200
    assert r1.requests == 0
    s = httpx.get(f"{url}/engine-stats").json()
    assert [x["healthy"] for x in s["replicas"]] == [True, False]
    # A partial sum would dip the counters, so the merged view is empty
    # while the live replica still reports its own.
    assert s["stats"] == {}
    assert s["replicas"][0]["stats"]["vllm:num_requests_running"] == 2.0


def test_engine_errors_count_per_replica(served, monkeypatch):
    app, url, _ = served
    r0, r1 = app._replicas
    monkeypatch.setattr(type(app), "engine_error_threshold", 2)
    fail = {"fail": True, "fail_status": 500}
    pick = {"replica": r1}
    monkeypatch.setattr(app, "_pick_replica", lambda session: pick["replica"])
    httpx.post(f"{url}/v1/chat/completions", json=fail)
    # A success on another replica does not hide the failing one.
    pick["replica"] = r0
    assert httpx.post(f"{url}/v1/chat/completions", json={}).status_code == 200
    pick["replica"] = r1
    httpx.post(f"{url}/v1/chat/completions", json=fail)
    r = httpx.post(f"{url}/engine-health")
    assert r.status_code == 503
    assert "replica 1 failed 2 requests in a row" in r.json()["detail"]
    assert not r1.usable() and r0.usable()


@pytest.mark.parametrize("served", [{"shutdown_grace_s": 0.5}], indirect=True)
def test_drain_aborts_streams_on_every_replica_and_teardown_stops_all(served):
    app, url, abort_file = served
    r0, r1 = app._replicas
    results: dict = {}

    def stream(session: str) -> None:
        with httpx.stream(
            "POST",
            f"{url}/v1/chat/completions",
            json={"stream": True, "n_chunks": 300},
            headers={"X-Fal-Runner-Hint": session},
        ) as r:
            results[session] = [c for c in r.iter_lines() if c.startswith("data:")]

    threads = [threading.Thread(target=stream, args=(s,)) for s in ("x", "y")]
    for started, t in enumerate(threads, 1):
        t.start()
        # Started one at a time, so least-inflight spreads them.
        end = time.time() + 5
        while r0.inflight + r1.inflight < started and time.time() < end:
            time.sleep(0.05)
    assert (r0.inflight, r1.inflight) == (1, 1)

    async def call():
        app.handle_exit()

    asyncio.run_coroutine_threadsafe(call(), app._test_loop).result(timeout=5)
    for t in threads:
        t.join(timeout=10)
    for data in results.values():
        assert '"finish_reason": "shutdown"' in data[-2]
    assert _wait_for(_abort_file(abort_file, r0))
    assert _wait_for(_abort_file(abort_file, r1))
    assert app._usage_totals["shutdown_aborts"] == 2

    app.teardown()
    assert r0.proc.poll() is not None and r1.proc.poll() is not None
