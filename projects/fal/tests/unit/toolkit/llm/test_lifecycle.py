from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import threading
import time
import types
from collections import deque

import httpx
import pytest
from fastapi import Request
from llm_helpers import _make_probe, _run, _wait_for, _wait_inflight

from fal.toolkit.llm import LLMApp, _app


def test_engine_logs_record_milestones_and_throttle_progress(capsys):
    script = (
        "import sys\n"
        "w = sys.stdout.write\n"
        "w('Loading shards 0/48\\rLoading shards 1/48\\rLoading shards 2/48\\r\\n')\n"
        "import time\n"
        "w('INFO Loading weights took 61.2 seconds\\n')\n"
        "w('INFO Graph capturing finished in 36 secs\\n'); sys.stdout.flush()\n"
        "time.sleep(0.2)\n"
        "w('INFO Graph capturing finished in 70 secs\\n')\n"
        "w('INFO Application startup complete.\\n')\n"
    )

    class Probe(LLMApp):
        model = "m"

    app = Probe()
    app._setup_start = time.monotonic()
    app._boot = {}
    app._engine_log_tail = deque(maxlen=50)
    proc = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    app._pump_engine_logs(proc)
    proc.wait()

    assert {"weights_loaded_s", "server_ready_s"} <= set(app._boot)
    # "_end" phases keep the last match: the second graph capture pass, 0.2s
    # after the first. Lines are stamped when read, so allow scheduling slack.
    assert app._boot["cuda_graph_end_s"] - app._boot["weights_loaded_s"] >= 0.1
    out = capsys.readouterr().out
    # The first redraw is logged, later ones inside the interval are dropped,
    # and the redraw ending in a newline always is.
    assert "Loading shards 0/48" in out
    assert "Loading shards 1/48" not in out
    assert "Loading shards 2/48" in out


def test_engine_logs_chunk_boundaries(capsys, monkeypatch):
    monkeypatch.setattr("fal.toolkit.llm._app._MAX_LINE_BYTES", 32)

    class Chunks:
        def __init__(self, chunks):
            self._chunks = iter(chunks)

        def read1(self, _n):
            return next(self._chunks, b"")

    class FakeProc:
        stdout = Chunks(
            [
                b"bar 1/2\r",
                b"bar 2/2\r",
                # \r\n split across reads is a normal line, not a redraw.
                b"\nINFO Loading weights took 1s\r",
                b"\n" + b"x" * 40,
                b"\nTraceback tail without newline",
            ]
        )

    class Probe(LLMApp):
        model = "m"

    app = Probe()
    app._setup_start = time.monotonic()
    app._boot = {}
    app._engine_log_tail = deque(maxlen=50)
    app._pump_engine_logs(FakeProc())  # type: ignore[arg-type]

    out = capsys.readouterr().out
    assert "bar 1/2" in out and "bar 2/2" in out
    assert "weights_loaded_s" in app._boot
    # Overlong runs are flushed and truncated at the cap.
    assert "x" * 32 in out and "x" * 33 not in out
    assert app._engine_log_tail[-1] == "Traceback tail without newline"


def test_startup_budget_capped_by_platform_timeout(tmp_path, monkeypatch):
    app = _make_probe(
        tmp_path, monkeypatch, startup_timeout=1000, engine_startup_timeout_s=3300.0
    )
    assert app._startup_budget_s() == 970.0
    # Tiny platform timeouts keep a proportional margin, never a negative budget.
    app = _make_probe(
        tmp_path, monkeypatch, startup_timeout=10, engine_startup_timeout_s=3300.0
    )
    assert app._startup_budget_s() == 9.0
    app = _make_probe(tmp_path, monkeypatch, engine_startup_timeout_s=5.0)
    assert app._startup_budget_s() == 5.0
    # Off a runner there is no platform timeout to stay under.
    app = _make_probe(
        tmp_path, monkeypatch, startup_timeout=10, engine_startup_timeout_s=3300.0
    )
    monkeypatch.delenv("IS_ISOLATE_AGENT")
    assert app._startup_budget_s() == 3300.0


def test_budget_exceeded_waiting_for_engine(tmp_path, monkeypatch, capsys):
    app = _make_probe(
        tmp_path,
        monkeypatch,
        engine_startup_timeout_s=2.0,
        persistent_caches=False,
    )
    app.engine_command = lambda p: [  # type: ignore[method-assign]
        sys.executable,
        "-c",
        "import time; print('engine booting', flush=True); time.sleep(60)",
    ]
    with pytest.raises(RuntimeError, match="resolve_s") as exc:
        app.setup()
    assert "engine booting" in str(exc.value)
    assert app._proc.poll() is not None
    assert "boot report" in capsys.readouterr().out


def test_budget_exhausted_after_resolve_never_starts_engine(tmp_path, monkeypatch):
    app = _make_probe(
        tmp_path,
        monkeypatch,
        engine_startup_timeout_s=0.2,
        persistent_caches=False,
    )
    monkeypatch.setattr(LLMApp, "_resolve_model", lambda self: time.sleep(0.5) or "p")
    started = []
    app._start_engine = started.append  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="resolve_s"):
        app.setup()
    assert started == []


def test_engine_exit_reports_boot_and_logs(tmp_path, monkeypatch):
    app = _make_probe(tmp_path, monkeypatch, persistent_caches=False)
    app.engine_command = lambda p: [  # type: ignore[method-assign]
        sys.executable,
        "-c",
        "print('boom: bad config', flush=True); raise SystemExit(3)",
    ]
    with pytest.raises(RuntimeError, match="exited with code 3") as exc:
        app.setup()
    assert "resolve_s" in str(exc.value)


def test_endpoint_annotations_are_not_strings():
    # `from __future__ import annotations` would make FastAPI read `request` as
    # a query parameter once the module is pickled by value.
    assert LLMApp.chat_completions.__annotations__["request"] is Request
    assert LLMApp.completions.__annotations__["request"] is Request


def test_watchdog_reports_stage_when_resolve_overruns(tmp_path, monkeypatch, capsys):
    app = _make_probe(
        tmp_path,
        monkeypatch,
        engine_startup_timeout_s=0.2,
        persistent_caches=False,
    )
    monkeypatch.setattr(LLMApp, "_resolve_model", lambda self: time.sleep(1.0) or "p")
    app._start_engine = lambda p: None  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        app.setup()
    out = capsys.readouterr().out
    assert "startup budget exceeded during model resolve; boot report:" in out


def test_watchdog_cancelled_after_setup(tmp_path, monkeypatch, capsys):
    app = _make_probe(tmp_path, monkeypatch, persistent_caches=False)
    app.setup()
    try:
        app._watchdog.join(timeout=5)
        assert not app._watchdog.is_alive()
    finally:
        app.teardown()
    assert "startup budget exceeded" not in capsys.readouterr().out


def test_fail_prints_report_before_stop_and_survives_stop_error(
    tmp_path, monkeypatch, capsys
):
    app = _make_probe(tmp_path, monkeypatch, persistent_caches=False)
    app._setup_start = time.monotonic()
    app._boot = {"resolve_s": 1.0}
    app._engine_log_tail = deque(maxlen=50)
    seen = []

    def stop(timeout):
        seen.append((timeout, "boot report" in capsys.readouterr().out))
        raise OSError("teardown broke")

    app._stop_engine = stop  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="real failure"):
        app._fail("real failure")
    assert seen == [(_app._FAIL_STOP_TIMEOUT_S, True)]


@pytest.mark.parametrize("served", [{"shutdown_grace_s": 0.5}], indirect=True)
def test_stream_error_body_read_ends_on_shutdown(served):
    app, url, abort_file, _ = served
    result: dict = {}

    def call():
        result["r"] = httpx.post(
            f"{url}/v1/chat/completions",
            json={"stream": True, "fail": True, "error_body_ticks": 300},
            timeout=30,
        )

    t = threading.Thread(target=call)
    t.start()
    assert _wait_inflight(app)
    time.sleep(0.5)  # error headers are in; the body is not
    start = time.monotonic()
    _handle_exit(app)
    t.join(timeout=10)
    assert time.monotonic() - start < 5
    assert result["r"].status_code == 503
    assert result["r"].headers["x-fal-needs-retry"] == "true"
    assert app._usage_totals["shutdown_aborts"] == 1
    assert _wait_for(abort_file, timeout=8.0)


SLOW_GRACE = {"shutdown_grace_s": 0.5}


LONG_GRACE = {"shutdown_grace_s": 20.0, "termination_grace_period_seconds": 30}


def _handle_exit(app) -> None:
    # The real call site is a signal handler on the event loop thread.
    async def call():
        app.handle_exit()

    asyncio.run_coroutine_threadsafe(call(), app._test_loop).result(timeout=5)


@pytest.mark.parametrize("served", [LONG_GRACE], indirect=True)
def test_draining_rejects_new_requests_and_health(served):
    app, url, _, _ = served
    _handle_exit(app)
    r = httpx.post(f"{url}/v1/chat/completions", json={"stream": True})
    assert r.status_code == 503
    assert r.headers["retry-after"] == "1"
    assert r.headers["x-fal-needs-retry"] == "true"
    assert r.json() == {
        "error": {"message": "runner is shutting down", "type": "shutting_down"}
    }
    assert httpx.post(f"{url}/v1/completions", json={"prompt": "x"}).status_code == 503
    assert httpx.get(f"{url}/health").status_code == 503


@pytest.mark.parametrize("served", [LONG_GRACE], indirect=True)
def test_inflight_stream_completes_within_grace(served):
    app, url, abort_file, _ = served
    with httpx.stream(
        "POST", f"{url}/v1/chat/completions", json={"stream": True, "n_chunks": 4}
    ) as r:
        lines = r.iter_lines()
        first = next(lines)
        _handle_exit(app)
        rest = [first, *lines]
    data = [c for c in rest if c.startswith("data:")]
    assert data[-1] == "data: [DONE]"
    assert any('"usage"' in c for c in data)
    assert not any("shutdown" in c for c in data)
    assert not abort_file.exists()
    assert "shutdown_aborts" not in app._usage_totals


@pytest.mark.parametrize("served", [SLOW_GRACE], indirect=True)
def test_stream_aborted_when_grace_expires(served):
    app, url, abort_file, _ = served
    with httpx.stream(
        "POST", f"{url}/v1/chat/completions", json={"stream": True, "n_chunks": 300}
    ) as r:
        lines = r.iter_lines()
        first = next(lines)
        _handle_exit(app)
        data = [c for c in [first, *lines] if c.startswith("data:")]
    assert data[-1] == "data: [DONE]"
    final = json.loads(data[-2][5:])
    assert final["choices"] == [{"index": 0, "delta": {}, "finish_reason": "shutdown"}]
    assert _wait_for(abort_file)
    assert app._usage_totals["shutdown_aborts"] == 1


@pytest.mark.parametrize("served", [SLOW_GRACE], indirect=True)
def test_completions_stream_shutdown_chunk_uses_text(served):
    app, url, _, _ = served
    with httpx.stream(
        "POST", f"{url}/v1/completions", json={"stream": True, "n_chunks": 300}
    ) as r:
        lines = r.iter_lines()
        first = next(lines)
        _handle_exit(app)
        data = [c for c in [first, *lines] if c.startswith("data:")]
    final = json.loads(data[-2][5:])
    assert final["choices"] == [{"index": 0, "text": "", "finish_reason": "shutdown"}]


@pytest.mark.parametrize("served", [SLOW_GRACE], indirect=True)
def test_unary_aborted_when_grace_expires(served):
    app, url, abort_file, _ = served
    result: dict = {}

    def call():
        result["r"] = httpx.post(
            f"{url}/v1/chat/completions", json={"delay_ticks": 300}, timeout=20.0
        )

    t = threading.Thread(target=call)
    t.start()
    assert _wait_inflight(app)
    _handle_exit(app)
    t.join(timeout=15)
    r = result["r"]
    assert r.status_code == 503
    assert r.headers["x-fal-needs-retry"] == "true"
    assert _wait_for(abort_file, timeout=8.0)
    assert app._usage_totals["shutdown_aborts"] == 1


@pytest.mark.parametrize("served", [SLOW_GRACE], indirect=True)
def test_stream_aborted_while_waiting_for_engine_headers(served):
    app, url, abort_file, _ = served
    result: dict = {}

    def call():
        result["r"] = httpx.post(
            f"{url}/v1/chat/completions",
            json={"stream": True, "header_delay_ticks": 300},
            timeout=20.0,
        )

    t = threading.Thread(target=call)
    t.start()
    assert _wait_inflight(app)
    _handle_exit(app)
    t.join(timeout=15)
    r = result["r"]
    assert r.status_code == 503
    assert r.headers["x-fal-needs-retry"] == "true"
    assert _wait_for(abort_file, timeout=8.0)
    assert app._usage_totals["shutdown_aborts"] == 1
    assert not app._inflight


@pytest.mark.parametrize("served", [SLOW_GRACE], indirect=True)
def test_shutdown_before_deadline_ends_stream_with_shutdown(served):
    app, url, abort_file, _ = served
    with httpx.stream(
        "POST",
        f"{url}/v1/chat/completions",
        json={"stream": True, "n_chunks": 300},
        headers={"X-Fal-Request-Deadline": "30"},
    ) as r:
        lines = r.iter_lines()
        first = next(lines)
        _handle_exit(app)
        data = [c for c in [first, *lines] if c.startswith("data:")]
    final = json.loads(data[-2][5:])
    assert final["choices"][0]["finish_reason"] == "shutdown"
    assert _wait_for(abort_file)
    assert app._usage_totals["shutdown_aborts"] == 1
    assert "deadline_aborts" not in app._usage_totals


@pytest.mark.parametrize("served", [LONG_GRACE], indirect=True)
def test_deadline_still_applies_while_draining(served):
    app, url, abort_file, _ = served
    with httpx.stream(
        "POST",
        f"{url}/v1/chat/completions",
        json={"stream": True, "n_chunks": 300},
        headers={"X-Fal-Request-Deadline": "0.5"},
    ) as r:
        lines = r.iter_lines()
        first = next(lines)
        _handle_exit(app)
        data = [c for c in [first, *lines] if c.startswith("data:")]
    final = json.loads(data[-2][5:])
    assert final["choices"][0]["finish_reason"] == "deadline"
    assert _wait_for(abort_file)
    assert app._usage_totals["deadline_aborts"] == 1
    assert "shutdown_aborts" not in app._usage_totals
    assert not app._inflight


@pytest.mark.parametrize("served", [SLOW_GRACE], indirect=True)
def test_unary_shutdown_before_deadline_is_503(served):
    app, url, _, _ = served
    result: dict = {}

    def call():
        result["r"] = httpx.post(
            f"{url}/v1/chat/completions",
            json={"delay_ticks": 300},
            headers={"X-Fal-Request-Deadline": "30"},
            timeout=20.0,
        )

    t = threading.Thread(target=call)
    t.start()
    assert _wait_inflight(app)
    _handle_exit(app)
    t.join(timeout=15)
    assert result["r"].status_code == 503
    assert result["r"].headers["x-fal-needs-retry"] == "true"
    assert app._usage_totals["shutdown_aborts"] == 1
    assert "deadline_aborts" not in app._usage_totals


def test_teardown_stops_engine_without_handle_exit(served):
    app, _, _, _ = served
    assert app._proc.poll() is None
    app.teardown()
    assert app._proc.poll() is not None


def test_drain_budget_leaves_engine_and_slack():
    class A(LLMApp):
        model = "m"

    assert A()._drain_budget() == 30.0  # 60s class default, capped by shutdown_grace_s
    A.termination_grace_period_seconds = None
    assert A()._drain_budget() == 1.0  # 5s platform window - 3s engine - 1s slack
    A.termination_grace_period_seconds = 2
    assert A()._drain_budget() == 0.0
    A.termination_grace_period_seconds = 60
    assert A()._drain_budget() == 30.0
    # The SDK does not clamp: the platform enforces its own cap.
    A.termination_grace_period_seconds = 3600
    A.shutdown_grace_s = 1000.0
    assert A()._drain_budget() == 1000.0


def test_termination_grace_default_reaches_host_kwargs():
    class A(LLMApp):
        model = "m"

    class B(LLMApp):
        model = "m"
        termination_grace_period_seconds = 120

    assert A.termination_grace_period_seconds == 60
    assert A.host_kwargs["termination_grace_period_seconds"] == 60
    assert A()._termination_window() == 60.0
    assert B.host_kwargs["termination_grace_period_seconds"] == 120
    assert B()._termination_window() == 120.0


def test_pyproject_grace_overrides_inherited_default_only():
    from fal.api import Options
    from fal.app import App
    from fal.utils import (
        _merge_host_config,
        _toml_host_keys_overriding_app_defaults,
    )

    class A(LLMApp):
        model = "m"

    class B(LLMApp):
        model = "m"
        termination_grace_period_seconds = 120

    class Base(App):
        termination_grace_period_seconds = 30

    class C(Base):
        pass

    options = Options(host={"termination_grace_period_seconds": 300})
    key = "termination_grace_period_seconds"
    overrides = _toml_host_keys_overriding_app_defaults(A, App, options)
    assert key in overrides
    merged = dict(A.host_kwargs)
    _merge_host_config(merged, options.host, overrides)
    assert merged[key] == 300
    assert key not in _toml_host_keys_overriding_app_defaults(B, App, options)
    # A user's own base class still wins, as before.
    assert key not in _toml_host_keys_overriding_app_defaults(C, App, options)


def test_engine_stop_wait_tracks_remaining_window():
    class A(LLMApp):
        model = "m"
        termination_grace_period_seconds = 60

    app = A()
    assert app._engine_stop_wait() == 30.0  # handle_exit never ran
    app._drain_started = time.monotonic()
    assert 29.0 < app._engine_stop_wait() <= 30.0
    app._drain_started = time.monotonic() - 45
    assert 3.0 < app._engine_stop_wait() <= 15.0
    app._drain_started = time.monotonic() - 100
    assert app._engine_stop_wait() == 3.0


@pytest.mark.parametrize("served", [SLOW_GRACE], indirect=True)
def test_handle_exit_without_running_loop_uses_serving_loop(served):
    app, url, abort_file, _ = served
    assert httpx.get(f"{url}/health").status_code == 200
    with httpx.stream(
        "POST", f"{url}/v1/chat/completions", json={"stream": True, "n_chunks": 300}
    ) as r:
        lines = r.iter_lines()
        first = next(lines)
        app.handle_exit()  # test thread: no running loop
        data = [c for c in [first, *lines] if c.startswith("data:")]
    assert "shutdown" in data[-2]
    assert _wait_for(abort_file)


def test_handle_exit_without_any_loop_warns(served, capsys):
    app, _, _, _ = served
    app.handle_exit()
    assert app._draining
    assert "no event loop" in capsys.readouterr().out


def test_request_registering_after_abort_sweep_gets_503(served):
    app, _, _, _ = served

    async def go():
        gate = asyncio.Event()

        class Req:
            async def json(self):
                await gate.wait()
                return {"stream": True}

        task = asyncio.ensure_future(app._proxy(Req(), "/v1/chat/completions"))
        await asyncio.sleep(0.1)
        app._abort_inflight()
        gate.set()
        return await task

    assert _run(app, go()).status_code == 503


@pytest.mark.parametrize("budget,expected", [(3000.0, 2990), (10.0, 600)])
def test_engine_ready_timeout_follows_the_startup_budget(monkeypatch, budget, expected):
    seen = {}

    class Proc:
        stdout = io.BytesIO(b"")

        def poll(self):
            return 0

    def popen(cmd, env, **kwargs):
        seen.update(env)
        return Proc()

    class A(LLMApp):
        model = "m"
        engine_presets = False

    monkeypatch.delenv("VLLM_ENGINE_READY_TIMEOUT_S", raising=False)
    monkeypatch.setattr(_app.subprocess, "Popen", popen)
    app = A()
    app._engine_log_tail = deque(maxlen=5)
    app._boot = {}
    app._deadline = time.monotonic() + budget
    app._start_engine("/m")
    value = int(seen["VLLM_ENGINE_READY_TIMEOUT_S"])
    assert expected - 15 <= value <= expected + 10
    # A value the app sets wins.
    A.engine_env = {"VLLM_ENGINE_READY_TIMEOUT_S": "42"}
    app._start_engine("/m")
    assert seen["VLLM_ENGINE_READY_TIMEOUT_S"] == "42"


@pytest.mark.parametrize(
    "engine,path", [("vllm", "/health"), ("sglang", "/health_generate")]
)
def test_wait_ready_probes_serving_endpoint_per_engine(monkeypatch, engine, path):
    class App(LLMApp):
        model = "org/m"

    App.engine = engine
    app = object.__new__(App)
    proc = types.SimpleNamespace(poll=lambda: None)
    app._replicas = [types.SimpleNamespace(url="http://r0", proc=proc)]
    app._deadline = time.monotonic() + 30
    seen = []

    def fake_get(url, timeout):
        assert 1.0 <= timeout <= (30.0 if engine == "sglang" else 5.0)
        seen.append(url)
        # SGLang's /health is 200 before warmup; only /health_generate gates.
        code = 200 if url.endswith(path) else 500
        return types.SimpleNamespace(status_code=code)

    monkeypatch.setattr(httpx, "get", fake_get)
    app._wait_ready()
    assert seen == [f"http://r0{path}"]


def test_wait_ready_probe_timeout_capped_by_remaining_budget(monkeypatch):
    class App(LLMApp):
        model = "org/m"

    App.engine = "sglang"
    app = object.__new__(App)
    proc = types.SimpleNamespace(poll=lambda: None)
    app._replicas = [types.SimpleNamespace(url="http://r0", proc=proc)]
    app._deadline = time.monotonic() + 5
    timeouts = []

    def fake_get(url, timeout):
        timeouts.append(timeout)
        return types.SimpleNamespace(status_code=200)

    monkeypatch.setattr(httpx, "get", fake_get)
    app._wait_ready()
    assert timeouts[0] <= 5.0


@pytest.mark.skipif(os.name == "nt", reason="process groups are POSIX-only")
@pytest.mark.parametrize("parent_exits", [False, True])
def test_stop_engine_kills_the_process_group(tmp_path, parent_exits):
    child_pid_file = tmp_path / "child.pid"
    script = (
        "import os, subprocess, sys, time\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(child_pid_file)!r} + '.tmp', 'w').write(str(c.pid))\n"
        f"os.replace({str(child_pid_file)!r} + '.tmp', {str(child_pid_file)!r})\n"
        f"time.sleep({0 if parent_exits else 60})\n"
    )

    class A(LLMApp):
        model = "m"

    app = A()
    replica = _app._Replica(app, 0, 9000)
    replica.proc = subprocess.Popen(
        [sys.executable, "-c", script], start_new_session=True
    )
    replica.proc._own_group = True  # type: ignore[attr-defined]
    app._replicas = [replica]
    assert _wait_for(child_pid_file)
    child = int(child_pid_file.read_text())
    if parent_exits:
        replica.proc.wait(timeout=5)
    app._stop_engine(5.0)
    assert replica.proc.poll() is not None
    end = time.time() + 5
    while time.time() < end:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(child, 9)
        pytest.fail("engine worker survived _stop_engine")


@pytest.mark.skipif(os.name == "nt", reason="process groups are POSIX-only")
def test_engine_leads_its_own_process_group(served):
    app, _, _, _ = served
    assert os.getpgid(app._proc.pid) == app._proc.pid
    assert app._proc._own_group
