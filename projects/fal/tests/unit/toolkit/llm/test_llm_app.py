from __future__ import annotations

import asyncio
import io
import json
import os
import resource
import socket
import subprocess
import sys
import threading
import time
import types
from collections import deque
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import Request

from fal.toolkit.llm import LLMApp, _app, _cache
from fal.toolkit.llm._app import (
    _deadline_budget,
    _event_cut,
    _RelayResponse,
    _Stream,
    _wait_limit,
)

FAKE = str(Path(__file__).parent / "fake_engine.py")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def isolate_agent_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IS_ISOLATE_AGENT", "1")


@pytest.fixture
def served(tmp_path, monkeypatch, request):
    abort_file = tmp_path / "abort"
    body_file = tmp_path / "body.json"
    monkeypatch.setenv("FAKE_ENGINE_ABORT_FILE", str(abort_file))
    monkeypatch.setenv("FAKE_ENGINE_BODY_FILE", str(body_file))
    shard_dir = tmp_path / "model"
    shard_dir.mkdir()
    (shard_dir / "m.safetensors").write_bytes(b"0" * 4096)
    port = _free_port()

    class FakeLLM(LLMApp):
        model = str(shard_dir)
        served_model_name = "served-name"
        engine_port = _free_port()
        persistent_caches = False
        preread_workers = 2
        prefix_hints = True
        engine_startup_timeout_s = 30.0
        locals().update(getattr(request, "param", {}))

        def engine_command(self, model_path: str) -> list[str]:
            return [sys.executable, FAKE, "--port", str(self.engine_port)]

    app = FakeLLM()
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
    yield app, f"http://127.0.0.1:{port}", abort_file, body_file
    server.should_exit = True
    thread.join(timeout=15)


def _wait_for(path: Path, timeout: float = 5.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if path.exists():
            return True
        time.sleep(0.1)
    return False


def _wait_inflight(app, timeout: float = 10.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if app._inflight:
            return True
        time.sleep(0.05)
    return False


def _wait_idle(app, timeout: float = 5.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if not app._inflight:
            return True
        time.sleep(0.05)
    return False


def test_unary_usage_model_rewrite_and_hints(served):
    app, url, _, body_file = served
    r = httpx.post(
        f"{url}/v1/chat/completions",
        json={"model": "whatever", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert json.loads(body_file.read_text())["model"] == "served-name"
    assert app._usage_totals == {
        "prompt_tokens": 7,
        "completion_tokens": 5,
        "cached_tokens": 3,
        "requests": 1,
    }
    # No session id sent: nothing advertised.
    assert r.headers["x-fal-runner-hints"] == ""
    assert httpx.post(f"{url}/v1/completions", json={"prompt": "x"}).status_code == 200
    assert httpx.get(f"{url}/v1/models").json()["data"][0]["id"] == "fake"


def test_hints_follow_client_session_ids(served):
    app, url, _, _ = served
    msgs = [{"role": "user", "content": "hi"}]
    post = httpx.post
    r = post(
        f"{url}/v1/chat/completions",
        json={"messages": msgs},
        headers={"X-Fal-Runner-Hint": "sess-a"},
    )
    assert r.headers["x-fal-runner-hints"] == "sess-a"
    # prompt_cache_key and user work too; the header wins over both.
    post(f"{url}/v1/chat/completions", json={"messages": msgs, "user": "u-1"})
    r = post(
        f"{url}/v1/responses",
        json={"input": "x", "prompt_cache_key": "k-2", "user": "ignored"},
    )
    assert r.headers["x-fal-runner-hints"] == "sess-a,u-1,k-2"
    # A repeat moves the id to the most recent slot instead of duplicating it.
    r = post(
        f"{url}/v1/chat/completions",
        json={"messages": msgs},
        headers={"X-Fal-Runner-Hint": "sess-a"},
    )
    assert r.headers["x-fal-runner-hints"] == "u-1,k-2,sess-a"
    # Ids the header cannot carry (or that could never match) are not advertised.
    before = r.headers["x-fal-runner-hints"]
    for bad in ("a,b c", "a\r\nX: y", "z" * 65, "naïve"):
        r = post(f"{url}/v1/chat/completions", json={"user": bad})
        assert r.status_code == 200
        assert r.headers["x-fal-runner-hints"] == before
    # A non-string first choice does not hide a valid later one.
    r = post(f"{url}/v1/chat/completions", json={"prompt_cache_key": 5, "user": "u-ok"})
    assert r.headers["x-fal-runner-hints"].split(",")[-1] == "u-ok"


@pytest.mark.parametrize("served", [{"prefix_hint_max": 3}], indirect=True)
def test_hints_are_bounded(served):
    _, url, _, _ = served
    for i in range(5):
        r = httpx.post(f"{url}/v1/chat/completions", json={"user": f"s{i}"})
    assert r.headers["x-fal-runner-hints"] == "s2,s3,s4"


@pytest.mark.parametrize("served", [{"prefix_hints": False}], indirect=True)
def test_hints_off_by_default(served):
    _, url, _, _ = served
    r = httpx.post(f"{url}/v1/chat/completions", json={"user": "s"})
    assert "x-fal-runner-hints" not in r.headers


def test_responses_unary_passthrough_and_usage(served):
    app, url, _, body_file = served
    r = httpx.post(f"{url}/v1/responses", json={"model": "x", "input": "hi"})
    assert r.status_code == 200
    assert json.loads(body_file.read_text())["model"] == "served-name"
    assert app._usage_totals["prompt_tokens"] == 7
    assert app._usage_totals["completion_tokens"] == 5
    assert app._usage_totals["cached_tokens"] == 3


@pytest.mark.parametrize("pad", [0, 20000])
def test_responses_stream_passthrough_usage(served, pad):
    app, url, _, body_file = served
    with httpx.stream(
        "POST", f"{url}/v1/responses", json={"stream": True, "pad_chars": pad}
    ) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    events = [ln[7:] for ln in text.splitlines() if ln.startswith("event: ")]
    assert events == ["response.output_text.delta"] * 3 + ["response.completed"]
    assert "[DONE]" not in text
    # No chat-only stream_options is injected for the Responses API.
    assert "stream_options" not in json.loads(body_file.read_text())
    assert app._usage_totals["completion_tokens"] == 5
    assert app._usage_totals["cached_tokens"] == 3


def test_responses_stream_deadline_ends_with_incomplete_event(served):
    _, url, abort_file, _ = served
    with httpx.stream(
        "POST",
        f"{url}/v1/responses",
        json={"stream": True},
        headers={"X-Fal-Request-Deadline": "0.3"},
    ) as r:
        text = "".join(r.iter_text())
    assert "response.incomplete" in text
    assert '"reason": "deadline"' in text
    assert _wait_for(abort_file)


def test_stream_usage_parsing():
    chat = b'data: {"choices": [], "usage": {"prompt_tokens": 2}}\n\ndata: [DONE]\n\n'
    assert _app._parse_stream_usage(chat) == {"prompt_tokens": 2}
    cut = (
        b'..."output_text": "xx"}, "usage": {"input_tokens": 9, '
        b'"input_tokens_details": {"cached_tokens": 4}, "output_tokens": 1}}}\n\n'
        b"event: response.completed\n"
    )
    # The start of the data line fell out of the tail.
    usage = _app._parse_stream_usage(cut)
    assert usage["input_tokens"] == 9
    assert _app._cached_tokens(usage) == 4
    assert _app._cached_tokens({"prompt_tokens_details": None}) == 0


def test_upstream_error_and_bad_body(served):
    _, url, _, _ = served
    r = httpx.post(f"{url}/v1/chat/completions", json={"fail": True})
    assert r.status_code == 400
    assert httpx.post(f"{url}/v1/chat/completions", content=b"nope").status_code == 400


def test_stalled_stream_error_body_is_bounded(served, monkeypatch):
    app, url, _, _ = served
    closed = []

    class Stalled:
        status_code = 500

        async def aread(self):
            await asyncio.sleep(3600)

        async def aclose(self):
            closed.append(1)

    async def open_upstream(path, payload):
        return Stalled()

    monkeypatch.setattr(_app, "_ERROR_BODY_TIMEOUT_S", 0.2)
    monkeypatch.setattr(app, "_open_upstream", open_upstream)
    r = httpx.post(f"{url}/v1/chat/completions", json={"stream": True}, timeout=10)
    assert r.status_code == 500
    assert "timed out" in r.text
    assert closed == [1]


def test_stream_passthrough_usage(served):
    app, url, _, _ = served
    chunks = []
    with httpx.stream(
        "POST", f"{url}/v1/chat/completions", json={"stream": True, "n_chunks": 3}
    ) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        chunks = [c for c in r.iter_lines() if c.startswith("data:")]
    assert chunks[-1] == "data: [DONE]"
    assert any('"usage"' in c for c in chunks)
    assert app._usage_totals["completion_tokens"] == 5


def test_stream_disconnect_aborts_engine(served):
    _, url, abort_file, _ = served
    with httpx.stream(
        "POST", f"{url}/v1/chat/completions", json={"stream": True, "n_chunks": 100}
    ) as r:
        next(r.iter_lines())
    assert _wait_for(abort_file)


def test_unary_disconnect_aborts_engine(served):
    _, url, abort_file, _ = served
    with pytest.raises(httpx.ReadTimeout):
        httpx.post(f"{url}/v1/chat/completions", json={"delay_ticks": 300}, timeout=1.0)
    assert _wait_for(abort_file, timeout=8.0)


def test_llm_app_opts_into_gateway_cancel():
    assert LLMApp.cancel_on_gateway_cancel is True


def _gateway_cancel(url: str, path: str, request_id: str) -> dict:
    r = httpx.post(f"{url}{path}/cancel", headers={"x-fal-request-id": request_id})
    assert r.status_code == 200
    assert r.headers["x-fal-billable-units"] == "0"
    return r.json()


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
def test_gateway_cancel_aborts_unary_engine_request(served, path):
    app, url, abort_file, _ = served
    result: dict = {}

    def call():
        result["r"] = httpx.post(
            f"{url}{path}",
            json={"delay_ticks": 300},
            headers={"x-fal-request-id": "req-unary"},
            timeout=20.0,
        )

    t = threading.Thread(target=call)
    t.start()
    assert _wait_inflight(app)
    assert _gateway_cancel(url, path, "req-unary") == {"cancelled": True}
    t.join(timeout=10)
    assert result["r"].status_code == 499
    assert result["r"].headers["x-fal-billable-units"] == "0"
    assert _wait_for(abort_file, timeout=8.0)
    assert app._usage_totals["cancel_aborts"] == 1
    assert _wait_idle(app)


@pytest.mark.parametrize(
    "served", [{}, {"raw_stream_client": False}], indirect=True, ids=["raw", "httpx"]
)
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
def test_gateway_cancel_aborts_stream_engine_request(served, path):
    app, url, abort_file, _ = served
    with httpx.stream(
        "POST",
        f"{url}{path}",
        json={"stream": True, "n_chunks": 300},
        headers={"x-fal-request-id": "req-stream"},
        timeout=20.0,
    ) as r:
        lines = r.iter_lines()
        next(lines)
        assert _gateway_cancel(url, path, "req-stream") == {"cancelled": True}
        # Mid-stream the connection is cut rather than ended cleanly.
        with pytest.raises(httpx.HTTPError):
            for _ in lines:
                pass
    assert _wait_for(abort_file)
    assert app._usage_totals["cancel_aborts"] == 1
    assert _wait_idle(app)


@pytest.mark.parametrize(
    "served", [{}, {"raw_stream_client": False}], indirect=True, ids=["raw", "httpx"]
)
def test_gateway_cancel_before_stream_headers_aborts_engine_request(served):
    app, url, abort_file, _ = served
    result: dict = {}

    def call():
        result["r"] = httpx.post(
            f"{url}/v1/chat/completions",
            json={"stream": True, "header_delay_ticks": 300},
            headers={"x-fal-request-id": "req-headers"},
            timeout=20.0,
        )

    t = threading.Thread(target=call)
    t.start()
    assert _wait_inflight(app)
    assert _gateway_cancel(url, "/v1/chat/completions", "req-headers") == {
        "cancelled": True
    }
    t.join(timeout=10)
    assert result["r"].status_code == 499
    assert _wait_for(abort_file, timeout=8.0)
    assert app._usage_totals["cancel_aborts"] == 1
    assert _wait_idle(app)


def test_gateway_cancel_of_unknown_or_finished_request_is_a_no_op(served):
    app, url, abort_file, _ = served
    r = httpx.post(
        f"{url}/v1/chat/completions",
        json={},
        headers={"x-fal-request-id": "req-done"},
    )
    assert r.status_code == 200
    assert _gateway_cancel(url, "/v1/chat/completions", "req-done") == {
        "cancelled": False
    }
    assert not abort_file.exists()
    assert "cancel_aborts" not in app._usage_totals


def test_health_and_engine_stats(served):
    app, url, _, _ = served
    h = httpx.get(f"{url}/health").json()
    assert h["status"] == "ok"
    assert h["engine"]["vllm:num_requests_running"] == 2.0
    s = httpx.get(f"{url}/engine-stats").json()
    assert s["stats"]["vllm:kv_cache_usage_perc"] == 0.25
    assert "preread_s" in s["boot"] and "engine_ready_s" in s["boot"]
    # The fake engine prints no vLLM milestones, so none are recorded.
    assert "weights_loaded_s" not in s["boot"]
    assert "other_metric" not in s["stats"]
    assert s["stats"]["spec_decode_acceptance_rate"] == 0.75
    # The per-position counter is not summed into the accepted total.
    assert s["stats"]["vllm:spec_decode_num_accepted_tokens_total"] == 30.0
    assert h["engine"]["spec_decode_acceptance_rate"] == 0.75

    app._proc.kill()
    app._proc.wait()
    assert httpx.get(f"{url}/health").status_code == 503


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


def _make_probe(tmp_path, monkeypatch, **attrs):
    """An LLMApp with caches redirected under tmp_path and a stub engine."""
    monkeypatch.setattr(_cache, "HF_HOME_DIR", tmp_path / "hf")
    monkeypatch.setattr(_cache, "ENGINE_CACHE_ROOT", tmp_path / "engine")
    monkeypatch.setattr(_app, "namespace_dir", lambda ns: tmp_path / "engine" / ns)
    for key in list(os.environ):
        if key.startswith(("HF_", "VLLM_", "TRITON_", "TORCHINDUCTOR_")):
            monkeypatch.delenv(key)
    shard_dir = tmp_path / "model"
    shard_dir.mkdir(exist_ok=True)
    (shard_dir / "m.safetensors").write_bytes(b"0" * 64)

    class Probe(LLMApp):
        model = str(shard_dir)
        engine_port = _free_port()
        cache_namespace = "ns"
        preread_workers = 0
        engine_startup_timeout_s = 30.0

        def engine_command(self, model_path: str) -> list[str]:
            return [
                sys.executable,
                FAKE,
                "--port",
                str(self.engine_port),
                *self.engine_args,
            ]

    for key, value in attrs.items():
        setattr(Probe, key, value)
    return Probe()


def test_compile_marker_absent_then_present(tmp_path, monkeypatch):
    first = _make_probe(tmp_path, monkeypatch)
    first.setup()
    try:
        assert first._boot["compile_cache_warm"] is False
        assert "previous_boot" not in first._boot
    finally:
        first.teardown()
    markers = list((tmp_path / "engine" / "ns").glob("boot-*.json"))
    assert len(markers) == 1
    assert not list((tmp_path / "engine" / "ns").glob("*.tmp"))

    second = _make_probe(tmp_path, monkeypatch)
    second.setup()
    try:
        assert second._boot["compile_cache_warm"] is True
        assert set(second._boot["previous_boot"]) == {
            "total_setup_s",
            "engine_ready_s",
        }
    finally:
        second.teardown()

    # A different engine config is a different compile cache.
    other = _make_probe(tmp_path, monkeypatch, engine_args=["--tp", "8"])
    assert other._config_key() != second._config_key()


def test_compile_marker_keyed_on_resolved_snapshot(tmp_path, monkeypatch):
    seen: list = []

    def boot(snapshot):
        snapshot.mkdir(exist_ok=True)
        app = _make_probe(tmp_path, monkeypatch, model="org/m")

        def engine_command(path):
            seen.append(path)
            # Like a preset gated on the snapshot's config.json.
            gated = ["--gated"] if os.path.isdir(path) else []
            port = str(app.engine_port)
            return [sys.executable, FAKE, "--port", port, "--model", path, *gated]

        app.engine_command = engine_command  # type: ignore[method-assign]
        app._resolve_model = lambda: str(snapshot)  # type: ignore[method-assign]
        app.setup()
        try:
            return dict(app._boot)
        finally:
            app.teardown()

    first = boot(tmp_path / "snap-a")
    # Keyed after resolve, from the real snapshot path (not the Hub id).
    assert "org/m" not in seen and seen[0] == str(tmp_path / "snap-a")
    assert first["compile_cache_warm"] is False
    # A new snapshot dir of the same config is still the same compile cache.
    second = boot(tmp_path / "snap-b")
    assert second["compile_cache_warm"] is True
    assert len(list((tmp_path / "engine" / "ns").glob("boot-*.json"))) == 1


def _fake_snapshot_download(monkeypatch, tmp_path):
    # huggingface_hub is not a dev dependency, so stand in for the module.
    snap = tmp_path / "snap"
    snap.mkdir(exist_ok=True)
    (snap / "config.json").write_text("{}")

    def fake(repo_id, **kwargs):
        calls.append(bool(kwargs.get("local_files_only")))
        calls.revisions.append(kwargs.get("revision"))
        return str(snap)

    class Calls(list):
        def __init__(self):
            super().__init__()
            self.revisions = []

    calls = Calls()
    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = fake  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return calls


def test_resolve_local_first_gated_by_marker(tmp_path, monkeypatch):
    calls = _fake_snapshot_download(monkeypatch, tmp_path)
    app = _make_probe(tmp_path, monkeypatch, model="org/model")
    app._boot = {}

    # No marker: an interrupted download could be in the HF cache, so go to
    # the network and record the completed download.
    assert app._resolve_model() == str(tmp_path / "snap")
    assert calls == [False]
    assert app._boot["resolve_source"] == "network"

    # Marker present: the recorded snapshot is reused with no Hub call, so a
    # newer partial download on the shared volume is never picked up.
    calls.clear()
    assert app._resolve_model() == str(tmp_path / "snap")
    assert calls == []
    assert app._boot["resolve_source"] == "local"


def test_resolve_pruned_local_snapshot_downloads(tmp_path, monkeypatch):
    calls = _fake_snapshot_download(monkeypatch, tmp_path)
    app = _make_probe(tmp_path, monkeypatch, model="org/model")
    app._boot = {}
    app._resolve_model()
    (tmp_path / "snap" / "config.json").unlink()
    calls.clear()
    app._resolve_model()
    assert calls == [False]
    assert app._boot["resolve_source"] == "network"

    # A pruned weight blob leaves a dangling symlink.
    (tmp_path / "snap" / "config.json").write_text("{}")
    (tmp_path / "snap" / "w.safetensors").symlink_to(tmp_path / "gone")
    calls.clear()
    app._resolve_model()
    assert calls == [False]


def test_resolve_local_dir_skips_hub(tmp_path, monkeypatch):
    calls = _fake_snapshot_download(monkeypatch, tmp_path)
    app = _make_probe(tmp_path, monkeypatch)
    app._boot = {}
    assert app._resolve_model() == app.model
    assert calls == [] and app._boot["resolve_source"] == "path"


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


def test_revision_passed_and_in_marker_key(tmp_path, monkeypatch):
    calls = _fake_snapshot_download(monkeypatch, tmp_path)
    app = _make_probe(tmp_path, monkeypatch, model="org/model", model_revision="v1")
    app._boot = {}
    app._resolve_model()
    app._resolve_model()
    assert calls == [False]
    assert calls.revisions == ["v1"]
    assert _cache.snapshot_marker_path("org/model", "v1").exists()
    assert not _cache.snapshot_marker_path("org/model").exists()


def test_snapshot_marker_follows_hf_hub_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "custom" / "hub"))
    path = _cache.snapshot_marker_path("org/model")
    assert path.parent == tmp_path / "custom" / "llm_app_snapshots"


def test_resolve_without_persistent_caches_skips_marker(tmp_path, monkeypatch):
    calls = _fake_snapshot_download(monkeypatch, tmp_path)
    app = _make_probe(tmp_path, monkeypatch, model="org/model", persistent_caches=False)
    app._boot = {}
    app._resolve_model()
    app._resolve_model()
    assert calls == [False, False]
    assert not _cache.snapshot_marker_path("org/model").exists()


def test_config_key_engine_env_and_port_forms(tmp_path, monkeypatch, capsys):
    base = _make_probe(tmp_path, monkeypatch)
    with_env = _make_probe(tmp_path, monkeypatch, engine_env={"A": "1"})
    assert base._config_key() != with_env._config_key()

    def cmd_with(port_args):
        app = _make_probe(tmp_path, monkeypatch)
        app.engine_command = lambda p: ["serve", *port_args, "--tp", "2"]  # type: ignore[method-assign]
        return app._config_key()

    assert cmd_with(["--port", "1"]) == cmd_with(["--port=2"]) == cmd_with([])

    broken = _make_probe(tmp_path, monkeypatch)

    def boom(p):
        raise ValueError("bad")

    broken.engine_command = boom  # type: ignore[method-assign]
    broken._config_key()
    assert "falls back to engine_args" in capsys.readouterr().out


def test_config_key_uses_resolved_model_config(tmp_path, monkeypatch):
    app = _make_probe(tmp_path, monkeypatch)
    seen = []

    def cmd(path):
        seen.append(path)
        # Like a config-gated preset: the flag depends on the snapshot's files.
        extra = ["--kv-cache-dtype", "fp8"] if os.path.isdir(path) else []
        return ["serve", path, *extra]

    app.engine_command = cmd  # type: ignore[method-assign]
    snap_a, snap_b = tmp_path / "snap-a", tmp_path / "snap-b"
    snap_a.mkdir()
    snap_b.mkdir()
    keyed = app._config_key(str(snap_a))
    assert seen[-1] == str(snap_a)
    # The snapshot dir is replaced by the model id; the gated flag is kept.
    assert keyed == app._config_key(str(snap_b))
    # Without the snapshot's files the gated flag is absent, so the key differs.
    assert keyed != app._config_key(str(tmp_path / "missing"))


def test_config_key_follows_the_served_repo(tmp_path, monkeypatch):
    app = _make_probe(tmp_path, monkeypatch, model="org/m")
    app.engine_command = lambda path: ["serve", path]  # type: ignore[method-assign]
    snap = str(tmp_path / "snap")
    app._model_repo = "org/m-FP8"
    fp8 = app._config_key(snap)
    app._model_repo = "org/m"
    # Same flags, different weights: an FP8 checkpoint and the bf16 model
    # compile different kernels, so they must not share a boot marker.
    assert app._config_key(snap) != fp8
    app._model_repo = "org/m-FP8"
    assert app._config_key(str(tmp_path / "other-snap")) == fp8


def test_write_marker_leaves_no_tmp_and_tolerates_concurrency(tmp_path):
    path = tmp_path / "d" / "m.json"
    errors = []

    def write(i):
        try:
            for _ in range(20):
                _cache.write_marker(path, {"i": i})
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert list(path.parent.iterdir()) == [path]
    assert _cache.read_marker(path) is not None


def test_write_marker_cleans_tmp_on_error(tmp_path):
    with pytest.raises(TypeError):
        _cache.write_marker(tmp_path / "m.json", {"x": object()})
    assert list(tmp_path.iterdir()) == []


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


def _sse(r: httpx.Response) -> list[str]:
    return [c for c in r.iter_lines() if c.startswith("data:")]


def test_stream_deadline_ends_with_deadline_chunk_and_spares_siblings(served):
    app, url, abort_file, _ = served
    body = {"stream": True, "n_chunks": 100}
    sibling: dict = {}

    def run_sibling():
        with httpx.stream(
            "POST",
            f"{url}/v1/chat/completions",
            json={"stream": True, "n_chunks": 6},
            timeout=30,
        ) as r:
            sibling["lines"] = _sse(r)

    t = threading.Thread(target=run_sibling)
    t.start()
    with httpx.stream(
        "POST",
        f"{url}/v1/chat/completions",
        json=body,
        headers={"X-Fal-Request-Deadline": "0.5"},
        timeout=30,
    ) as r:
        lines = _sse(r)
    t.join(timeout=30)

    assert lines[-1] == "data: [DONE]"
    last = json.loads(lines[-2][5:])
    assert last["object"] == "chat.completion.chunk"
    assert last["choices"] == [{"index": 0, "delta": {}, "finish_reason": "deadline"}]
    assert _wait_for(abort_file)
    assert sibling["lines"][-1] == "data: [DONE]"
    assert any('"usage"' in c for c in sibling["lines"])
    assert not any("deadline" in c for c in sibling["lines"])
    assert app._usage_totals["deadline_aborts"] == 1
    assert app._usage_totals["requests"] == 1


def test_completions_stream_deadline_chunk_shape(served):
    _, url, _, _ = served
    with httpx.stream(
        "POST",
        f"{url}/v1/completions",
        json={"stream": True, "n_chunks": 100},
        headers={"X-Fal-Request-Deadline": "0.5"},
        timeout=30,
    ) as r:
        lines = _sse(r)
    assert lines[-1] == "data: [DONE]"
    last = json.loads(lines[-2][5:])
    assert last["object"] == "text_completion"
    assert last["choices"] == [{"index": 0, "text": "", "finish_reason": "deadline"}]


def test_stream_deadline_mid_event_keeps_events_whole(served):
    _, url, _, _ = served
    # Half events arrive every 0.2s (first halves at 0.0, 0.4, ...), so 0.5s
    # expires 0.1s after a first half and 0.1s before its second half.
    with httpx.stream(
        "POST",
        f"{url}/v1/chat/completions",
        json={"stream": True, "n_chunks": 100, "split_events": True},
        headers={"X-Fal-Request-Deadline": "0.5"},
        timeout=30,
    ) as r:
        body = r.read().decode()
    events = [e for e in body.split("\n\n") if e]
    assert events[-1] == "data: [DONE]"
    for event in events[:-1]:
        assert event.startswith("data: ")
        json.loads(event[6:])
    assert json.loads(events[-2][6:])["choices"][0]["finish_reason"] == "deadline"


def test_stream_deadline_before_headers(served):
    app, url, abort_file, _ = served
    with httpx.stream(
        "POST",
        f"{url}/v1/chat/completions",
        json={"stream": True, "header_delay_ticks": 300},
        headers={"X-Fal-Request-Deadline": "0.5"},
        timeout=30,
    ) as r:
        assert r.status_code == 200
        lines = _sse(r)
    assert lines[-1] == "data: [DONE]"
    assert json.loads(lines[-2][5:])["choices"][0]["finish_reason"] == "deadline"
    assert app._usage_totals["deadline_aborts"] == 1
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


def test_event_cut():
    assert _event_cut(b"") == 0
    assert _event_cut(b"data: a") == 0
    assert _event_cut(b"data: a\n\ndata: b") == 9
    assert _event_cut(b"data: a\r\n\r\ndata: b\r\n\r\n") == 22


def test_unary_deadline_returns_504_and_aborts_engine(served):
    app, url, abort_file, _ = served
    r = httpx.post(
        f"{url}/v1/chat/completions",
        json={"delay_ticks": 300},
        headers={"X-Fal-Request-Deadline": "0.5"},
        timeout=30,
    )
    assert r.status_code == 504
    assert r.headers["x-fal-needs-retry"] == "false"
    assert r.json() == {
        "error": {
            "message": "request deadline of 0.5s exceeded",
            "type": "deadline_exceeded",
        }
    }
    assert _wait_for(abort_file, timeout=8.0)
    assert app._usage_totals["deadline_aborts"] == 1


def test_gateway_header_alone_triggers_deadline(served):
    app, url, abort_file, _ = served
    # remaining 1.5s minus the 1s margin leaves a 0.5s budget.
    with httpx.stream(
        "POST",
        f"{url}/v1/chat/completions",
        json={"stream": True, "n_chunks": 100},
        headers={"X-Fal-Request-Timeout-Remaining": "1.500"},
        timeout=30,
    ) as r:
        lines = _sse(r)
    assert json.loads(lines[-2][5:])["choices"][0]["finish_reason"] == "deadline"
    assert _wait_for(abort_file)


@pytest.mark.parametrize("served", [{"request_deadline_s": 0.5}], indirect=True)
def test_class_deadline_applies_without_header(served):
    _, url, _, _ = served
    r = httpx.post(f"{url}/v1/chat/completions", json={"delay_ticks": 300}, timeout=30)
    assert r.status_code == 504


@pytest.mark.parametrize("served", [{"request_deadline_s": 0.5}], indirect=True)
def test_longer_deadline_header_is_clamped_to_class_default(served):
    _, url, _, _ = served
    r = httpx.post(
        f"{url}/v1/chat/completions",
        json={"delay_ticks": 300},
        headers={"X-Fal-Request-Deadline": "60"},
        timeout=30,
    )
    assert r.status_code == 504


def test_invalid_deadline_header_is_400(served):
    _, url, _, _ = served
    for bad in ("abc", "0", "-1", "nan"):
        r = httpx.post(
            f"{url}/v1/chat/completions",
            json={},
            headers={"X-Fal-Request-Deadline": bad},
        )
        assert r.status_code == 400
        assert "error" in r.json()


def test_deadline_budget_precedence():
    h = {}
    assert _deadline_budget(h, None, 1.0) is None
    assert _deadline_budget(h, 30.0, 1.0) == 30.0
    # Override shortens the class default but is clamped to it, not rejected.
    assert _deadline_budget({"x-fal-request-deadline": "10"}, 30.0, 1.0) == 10.0
    assert _deadline_budget({"x-fal-request-deadline": "60"}, 30.0, 1.0) == 30.0
    # Without a class default the override stands alone.
    assert _deadline_budget({"x-fal-request-deadline": "60"}, None, 1.0) == 60.0
    gw = {"x-fal-request-timeout-remaining": "297.350"}
    assert _deadline_budget(gw, None, 1.0) == pytest.approx(296.35)
    # The smaller of override/default and gateway wins.
    assert _deadline_budget(gw, 10.0, 1.0) == 10.0
    assert _deadline_budget({**gw, "x-fal-request-deadline": "500"}, None, 1.0) == (
        pytest.approx(296.35)
    )
    # Unparseable gateway values are ignored.
    assert (
        _deadline_budget({"x-fal-request-timeout-remaining": "soon"}, 5.0, 1.0) == 5.0
    )
    with pytest.raises(ValueError):
        _deadline_budget({"x-fal-request-deadline": "0"}, None, 1.0)


def test_teardown_stops_engine_without_handle_exit(served):
    app, _, _, _ = served
    assert app._proc.poll() is None
    app.teardown()
    assert app._proc.poll() is not None


def _run(app, coro, timeout: float = 10.0):
    return asyncio.run_coroutine_threadsafe(coro, app._test_loop).result(timeout)


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


class _FakeUpstream:
    def __init__(self, chunks, hang=False):
        self._chunks = chunks
        self._hang = hang
        self.closed = 0

    async def aiter_raw(self):
        for c in self._chunks:
            yield c
        if self._hang:
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed += 1


def test_no_shutdown_chunk_after_done_forwarded(served):
    app, _, _, _ = served

    async def go():
        up = _FakeUpstream([b"data: {}\n\n", b"data: [DONE]\n\n"], hang=True)
        abort = asyncio.Event()
        stream = _Stream(app, up, "/v1/chat/completions", abort)
        out = []
        async for c in app._relay(stream, "/v1/chat/completions"):
            out.append(c)
            if c.startswith(b"data: [DONE]"):
                abort.set()
        return b"".join(out)

    body = _run(app, go())
    assert b"shutdown" not in body
    assert "shutdown_aborts" not in app._usage_totals


def test_abort_not_starved_by_buffered_chunks(served):
    app, _, _, _ = served

    async def go():
        # Always-ready chunks: a read wins every race unless abort is checked
        # first.
        up = _FakeUpstream([b"data: {}\n\n"] * 1000)
        abort = asyncio.Event()
        stream = _Stream(app, up, "/v1/chat/completions", abort)
        out = []
        async for c in app._relay(stream, "/v1/chat/completions"):
            out.append(c)
            if len(out) == 3:
                abort.set()
        return out

    out = _run(app, go())
    # A read already finished when the abort lands is forwarded first.
    assert len(out) in (4, 5) and b"shutdown" in out[-1]
    assert app._usage_totals["shutdown_aborts"] == 1


def test_read_finishing_with_abort_is_forwarded(served):
    app, _, _, _ = served
    finish = b'data: {"choices": [{"finish_reason": "stop"}]}\n\n'

    async def go():
        abort = asyncio.Event()

        class Up(_FakeUpstream):
            async def aiter_raw(self):
                # The abort lands while this read completes.
                abort.set()
                yield finish
                await asyncio.Event().wait()

        stream = _Stream(app, Up([]), "/v1/chat/completions", abort)
        return [c async for c in app._relay(stream, "/v1/chat/completions")]

    out = _run(app, go())
    # The real finish event survives and only the terminator is added.
    assert out == [finish, b"data: [DONE]\n\n"]


def test_expired_deadline_wins_over_abort(served):
    app, _, _, _ = served

    async def go():
        up = _FakeUpstream([b"data: {}\n\n"], hang=True)
        abort = asyncio.Event()
        stream = _Stream(app, up, "/v1/chat/completions", abort)
        deadline = time.monotonic() + 0.2
        out = []
        async for c in app._relay(stream, "/v1/chat/completions", True, deadline):
            out.append(c)
            if len(out) == 1:
                await asyncio.sleep(0.3)
                abort.set()
                # Let the relay's abort waiter finish, so both are set.
                await asyncio.sleep(0)
        return out

    out = _run(app, go())
    assert (
        json.loads(out[-1].split(b"\n\n")[0][6:])["choices"][0]["finish_reason"]
        == "deadline"
    )
    assert app._usage_totals["deadline_aborts"] == 1
    assert "shutdown_aborts" not in app._usage_totals


def test_stream_idle_timeout_ends_stream_and_closes_upstream(served, capsys):
    app, _, _, _ = served
    app.stream_idle_timeout_s = 0.2

    async def go():
        up = _FakeUpstream([b"data: {}\n\n"], hang=True)
        stream = _Stream(app, up, "/v1/chat/completions", asyncio.Event())
        # A far deadline: the idle limit is what ends this stream.
        deadline = time.monotonic() + 30
        out = [
            c async for c in app._relay(stream, "/v1/chat/completions", True, deadline)
        ]
        return out, up.closed

    out, closed = _run(app, go())
    assert out[0] == b"data: {}\n\n"
    final = json.loads(out[-1].split(b"\n\n")[0][6:])
    assert final["choices"][0]["finish_reason"] == "idle_timeout"
    assert out[-1].endswith(b"data: [DONE]\n\n")
    assert closed == 1  # the connection close aborts the engine request
    assert app._usage_totals["idle_aborts"] == 1
    assert "deadline_aborts" not in app._usage_totals
    assert "nothing from" in capsys.readouterr().out


def test_stream_idle_timeout_after_done_adds_nothing(served):
    app, _, _, _ = served
    app.stream_idle_timeout_s = 0.2

    async def go():
        up = _FakeUpstream([b"data: {}\n\n", b"data: [DONE]\n\n"], hang=True)
        stream = _Stream(app, up, "/v1/chat/completions", asyncio.Event())
        return [c async for c in app._relay(stream, "/v1/chat/completions")]

    assert _run(app, go()) == [b"data: {}\n\n", b"data: [DONE]\n\n"]
    assert "idle_aborts" not in app._usage_totals


@pytest.mark.parametrize("served", [{"stream_idle_timeout_s": 0.5}], indirect=True)
def test_stream_idle_timeout_before_headers(served):
    app, url, abort_file, _ = served
    with httpx.stream(
        "POST",
        f"{url}/v1/chat/completions",
        json={"stream": True, "header_delay_ticks": 300},
        timeout=30,
    ) as r:
        assert r.status_code == 200
        lines = _sse(r)
    assert lines[-1] == "data: [DONE]"
    assert json.loads(lines[-2][5:])["choices"][0]["finish_reason"] == "idle_timeout"
    assert app._usage_totals["idle_aborts"] == 1
    assert _wait_for(abort_file, timeout=8.0)


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


def test_wait_limit_picks_the_sooner_bound():
    assert _wait_limit(None, None) == (None, False)
    assert _wait_limit(None, 300.0) == (300.0, True)
    soon = time.monotonic() + 1.0
    limit, idle = _wait_limit(soon, 300.0)
    assert 0.0 < limit <= 1.0 and not idle
    later = time.monotonic() + 600.0
    assert _wait_limit(later, 300.0) == (300.0, True)
    limit, idle = _wait_limit(soon, None)
    assert limit <= 1.0 and not idle


def test_unary_expired_deadline_wins_over_abort(served):
    app, _, abort_file, _ = served

    class Req:
        async def receive(self):
            await asyncio.Event().wait()

    async def go():
        abort = asyncio.Event()
        abort.set()
        return await app._proxy_unary(
            Req(),
            "/v1/chat/completions",
            {"delay_ticks": 300},
            abort,
            time.monotonic() - 1,
            1.0,
        )

    r = _run(app, go())
    assert r.status_code == 504
    assert r.headers["x-fal-needs-retry"] == "false"
    assert app._usage_totals["deadline_aborts"] == 1
    assert "shutdown_aborts" not in app._usage_totals


def test_unframed_body_flushed_past_line_limit(served, monkeypatch):
    app, _, _, _ = served
    monkeypatch.setattr(_app, "_MAX_LINE_BYTES", 10)

    async def go():
        up = _FakeUpstream([b"x" * 20], hang=True)
        stream = _Stream(app, up, "/v1/chat/completions", asyncio.Event())
        relay = app._relay(stream, "/v1/chat/completions", True, time.monotonic() + 0.3)
        return [c async for c in relay]

    out = _run(app, go())
    assert out[0] == b"x" * 20
    assert b"deadline" in out[-1]


@pytest.mark.parametrize("single_choice", [True, False])
def test_abort_after_finish_reason_only_sends_done(served, single_choice):
    app, _, _, _ = served

    async def go():
        up = _FakeUpstream(
            [b'data: {"choices": [{"finish_reason": "stop"}]}\n\n'], hang=True
        )
        abort = asyncio.Event()
        stream = _Stream(app, up, "/v1/chat/completions", abort)
        out = []
        relay = app._relay(stream, "/v1/chat/completions", single_choice)
        async for c in relay:
            out.append(c)
            abort.set()
        return out

    out = _run(app, go())
    assert out[-1].endswith(b"data: [DONE]\n\n")
    # With n > 1 other choices may still be open, so they get the reason.
    assert any(b"shutdown" in c for c in out) is not single_choice


def test_upstream_close_retried_after_cancelled_disconnect():
    # Starlette cancels the body task on disconnect; the first close is
    # interrupted at its checkpoint and the response's own cleanup finishes it.
    class Up:
        closed = False

        async def aiter_raw(self):
            while True:
                await asyncio.sleep(0.05)
                yield b'data: {"a": 1}\n\n'

        async def aclose(self):
            await asyncio.sleep(0.01)
            self.closed = True

    class Probe(LLMApp):
        model = "m"

    app = Probe()
    app._inflight = set()
    app._usage_totals = {}

    async def go():
        up = Up()
        abort = asyncio.Event()
        app._inflight.add(abort)
        stream = _Stream(app, up, "/v1/chat/completions", abort)
        resp = _RelayResponse(
            app._relay(stream, "/v1/chat/completions"),
            stream.close,
            media_type="text/event-stream",
        )

        async def receive():
            await asyncio.sleep(0.3)
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        scope = {"type": "http", "asgi": {"spec_version": "2.3"}}
        await resp(scope, receive, send)
        return up.closed, abort in app._inflight

    closed, still = asyncio.run(go())
    assert closed and not still


def test_response_cleanup_runs_once_when_generator_never_starts():
    calls = []
    started = []

    async def body():
        started.append(1)
        yield b"x"

    async def cleanup():
        calls.append(1)

    async def send(message):
        raise OSError("client gone")

    async def receive():
        return {"type": "http.disconnect"}

    async def go():
        resp = _RelayResponse(body(), cleanup)
        with pytest.raises(Exception):
            await resp({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)

    asyncio.run(go())
    assert calls == [1]
    assert not started


def test_response_cleanup_runs_once_after_full_iteration():
    calls = []
    sent = []

    async def body():
        yield b"x"

    async def cleanup():
        calls.append(1)

    async def send(message):
        sent.append(message)

    async def receive():
        await asyncio.Event().wait()

    async def go():
        resp = _RelayResponse(body(), cleanup)
        await resp({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)

    asyncio.run(go())
    assert calls == [1]
    assert sent[1]["body"] == b"x"


def test_stream_close_is_idempotent_and_records_usage_once(served):
    app, _, _, _ = served

    async def go():
        up = _FakeUpstream([])
        abort = asyncio.Event()
        app._inflight.add(abort)
        stream = _Stream(app, up, "/v1/chat/completions", abort)
        stream.tail = (
            b'data: {"usage": {"prompt_tokens": 1, "completion_tokens": 2}}\n\n'
        )
        await stream.close()
        await stream.close()
        return up.closed, abort in app._inflight

    closed, still = _run(app, go())
    assert closed == 1 and not still
    assert app._usage_totals["requests"] == 1


A_SHA = "a" * 40
B_SHA = "b" * 40
OLD = "2026-01-01T00:00:00+00:00"
NEW = "2026-02-01T00:00:00+00:00"


class _Hub:
    """Stub huggingface_hub: snapshot_download lands in a per-commit dir."""

    def __init__(self, tmp_path, monkeypatch):
        self.root = tmp_path / "snaps"
        self.downloads: list = []
        self.next_commit = A_SHA
        self.dates = {A_SHA: OLD, B_SHA: NEW}
        self.upstream = A_SHA
        self.fail_info = False
        self.info_calls: list = []

        def snapshot_download(repo_id, **kwargs):
            self.downloads.append(kwargs.get("revision"))
            return str(self.snap(self.next_commit))

        module = types.ModuleType("huggingface_hub")
        module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)
        monkeypatch.setattr(_app, "_hub_commit", self.hub_commit)

    def snap(self, commit):
        path = self.root / commit
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}")
        return path

    def hub_commit(self, model, revision, timeout):
        self.info_calls.append(revision)
        if self.fail_info:
            return None
        # A pinned lookup describes the commit the download landed on.
        sha = self.upstream if revision is None else self.next_commit
        return sha, self.dates.get(sha)


def _resolver(tmp_path, monkeypatch, **attrs):
    hub = _Hub(tmp_path, monkeypatch)
    app = _make_probe(tmp_path, monkeypatch, model="org/model", **attrs)
    app._boot = {}
    return hub, app


def _unpinned():
    return _cache.read_marker(_cache.snapshot_marker_path("org/model"))


def test_marker_records_commit_and_date(tmp_path, monkeypatch):
    hub, app = _resolver(tmp_path, monkeypatch)
    path = app._resolve_model()
    assert _unpinned() == {
        "model": "org/model",
        "revision": None,
        "path": path,
        "commit": A_SHA,
        "commit_date": OLD,
    }


def test_marker_commit_falls_back_to_dir_name_without_hub(tmp_path, monkeypatch):
    hub, app = _resolver(tmp_path, monkeypatch)
    hub.fail_info = True
    app._resolve_model()
    assert _unpinned()["commit"] == A_SHA
    assert _unpinned()["commit_date"] is None


def test_advance_replaces_older_keeps_newer_noop_same(tmp_path, monkeypatch):
    hub, app = _resolver(tmp_path, monkeypatch)
    app._resolve_model()
    a_path = str(hub.snap(A_SHA))
    b_path = str(hub.snap(B_SHA))
    advance = _cache.advance_snapshot_marker

    assert advance("org/model", b_path, B_SHA, NEW) is True
    assert _unpinned()["path"] == b_path
    # An older snapshot downloaded later does not win.
    assert advance("org/model", a_path, A_SHA, OLD) is False
    assert _unpinned()["commit"] == B_SHA
    # Same commit is a no-op.
    assert advance("org/model", b_path, B_SHA, NEW) is False
    # Unknown new date never replaces a known commit.
    assert advance("org/model", a_path, A_SHA, None) is False
    assert _unpinned()["commit"] == B_SHA


def test_advance_unknown_existing_date_and_old_format(tmp_path, monkeypatch):
    hub, app = _resolver(tmp_path, monkeypatch)
    app._resolve_model()
    b_path = str(hub.snap(B_SHA))
    marker = _cache.snapshot_marker_path("org/model")
    # Known commit with unknown date is replaced by a dated one.
    _cache.write_marker(marker, {**_unpinned(), "commit_date": None})
    assert _cache.advance_snapshot_marker("org/model", b_path, B_SHA, NEW)
    assert _unpinned()["commit_date"] == NEW

    # An old-format marker still resolves locally, and is replaced on advance.
    old_path = str(hub.snap(A_SHA))
    _cache.write_marker(marker, {"model": "org/model", "path": old_path})
    hub.downloads.clear()
    assert app._resolve_model() == old_path
    assert hub.downloads == []
    assert app._served_commit == A_SHA
    assert _cache.advance_snapshot_marker("org/model", b_path, B_SHA, NEW)
    assert _unpinned()["commit"] == B_SHA

    # Missing or unreadable marker is replaced regardless of dates.
    marker.write_text("not json")
    assert _cache.advance_snapshot_marker("org/model", old_path, A_SHA, None)
    assert _unpinned()["commit"] == A_SHA


def test_advance_repairs_marker_whose_snapshot_was_pruned(tmp_path, monkeypatch):
    hub, app = _resolver(tmp_path, monkeypatch)
    app._resolve_model()
    (hub.snap(A_SHA) / "config.json").unlink()
    # No date for the replacement, but the recorded copy is gone.
    b_path = str(hub.snap(B_SHA))
    assert _cache.advance_snapshot_marker("org/model", b_path, B_SHA, None)
    assert _unpinned()["path"] == b_path


def test_pinned_branch_off_main_does_not_advance_unpinned(tmp_path, monkeypatch):
    hub, app = _resolver(tmp_path, monkeypatch)
    app._resolve_model()
    # A newer-dated commit that is not main's head (a dev branch, a PR ref).
    hub.next_commit = B_SHA
    pinned = _make_probe(tmp_path, monkeypatch, model="org/model", model_revision="dev")
    pinned._boot = {}
    pinned._resolve_model()
    assert _unpinned()["commit"] == A_SHA


def test_pinned_download_advances_unpinned_marker(tmp_path, monkeypatch):
    hub, app = _resolver(tmp_path, monkeypatch, model_revision="v2")
    hub.next_commit = B_SHA
    hub.upstream = B_SHA  # the pinned revision is main's current head
    path = app._resolve_model()
    pinned = _cache.read_marker(_cache.snapshot_marker_path("org/model", "v2"))
    assert pinned["revision"] == "v2" and pinned["commit"] == B_SHA
    assert _unpinned()["path"] == path

    unpinned = _make_probe(tmp_path, monkeypatch, model="org/model")
    unpinned._boot = {}
    hub.downloads.clear()
    assert unpinned._resolve_model() == path
    assert hub.downloads == []
    assert unpinned._boot["resolve_source"] == "local"


def test_pin_hint_only_when_unpinned(tmp_path, monkeypatch, capsys):
    hub, app = _resolver(tmp_path, monkeypatch)
    app._resolve_model()
    assert "Pin model_revision" in capsys.readouterr().out
    (tmp_path / "p").mkdir()
    hub, pinned = _resolver(tmp_path / "p", monkeypatch, model_revision="v1")
    pinned._resolve_model()
    assert "Pin model_revision" not in capsys.readouterr().out


def _gauge():
    from prometheus_client import REGISTRY  # noqa: PLC0415

    return REGISTRY.get_sample_value(_app._UPSTREAM_GAUGE)


def test_upstream_newer_sets_boot_fields_and_gauge(tmp_path, monkeypatch):
    hub, app = _resolver(tmp_path, monkeypatch)
    app._resolve_model()
    marker_before = _unpinned()
    hub.upstream = B_SHA
    hub.downloads.clear()
    app._check_upstream()
    assert app._boot["model_commit"] == A_SHA
    assert app._boot["upstream_commit"] == B_SHA
    assert app._boot["upstream_newer"] is True
    assert _gauge() == 1.0
    # Report only: no download, markers untouched.
    assert hub.downloads == [] and _unpinned() == marker_before

    hub.upstream = A_SHA
    app._check_upstream()
    assert app._boot["upstream_newer"] is False
    assert _gauge() == 0.0


def test_upstream_check_queries_the_served_repo(tmp_path, monkeypatch):
    hub, app = _resolver(tmp_path, monkeypatch)
    app._resolve_model()
    asked = []
    monkeypatch.setattr(
        _app, "_hub_commit", lambda repo, rev, timeout: asked.append(repo) or None
    )
    # A recipe checkpoint was served: its own main is the comparison.
    app._model_repo = "org/model-FP8"
    app._check_upstream()
    assert asked == ["org/model-FP8"]


def test_upstream_check_failure_and_start_conditions(tmp_path, monkeypatch):
    hub, app = _resolver(tmp_path, monkeypatch)
    app._resolve_model()
    hub.fail_info = True
    app._check_upstream()
    assert "upstream_newer" not in app._boot and "model_commit" not in app._boot

    # Pinned apps and local paths never start the check.
    ran = []
    monkeypatch.setattr(LLMApp, "_check_upstream", lambda self: ran.append(1))
    (tmp_path / "p").mkdir()
    (tmp_path / "q").mkdir()
    _, pinned = _resolver(tmp_path / "p", monkeypatch, model_revision="v1")
    pinned._resolve_model()
    pinned._start_upstream_check()
    local = _make_probe(tmp_path / "q", monkeypatch)
    local._boot = {}
    local._resolve_model()
    local._start_upstream_check()
    # The unpinned Hub app does start it.
    app._start_upstream_check()
    deadline = time.monotonic() + 5
    while not ran and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ran == [1]


def test_stream_tail_keeps_recent_bytes_and_usage_survives_many_chunks(served):
    app, _, _, _ = served
    stream = _Stream(app, None, "/v1/chat/completions", asyncio.Event())
    for _ in range(5000):
        stream.feed(b'data: {"choices": [{"delta": {"content": "x"}}]}\n\n')
    usage = b'data: {"choices": [], "usage": {"completion_tokens": 3}}\n\n'
    stream.feed(usage)
    stream.feed(b"data: [DONE]\n\n")
    tail = stream.tail
    assert len(tail) == _app._STREAM_TAIL_BYTES
    # Eviction keeps memory bounded: about one tail plus one chunk.
    assert stream._recent_len < _app._STREAM_TAIL_BYTES + 200
    assert len(stream._recent) < 200
    assert tail.endswith(b"data: [DONE]\n\n")
    assert _app._parse_stream_usage(tail) == {"completion_tokens": 3}


def test_engine_pool_is_sized_for_many_streams(served):
    app, _, _, _ = served
    assert type(app).engine_max_connections >= 1024


def test_pool_fits_the_fd_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(resource, "getrlimit", lambda _: (1024, 1 << 20))
    monkeypatch.setattr(resource, "setrlimit", lambda *a: calls.append(a))
    assert _app._fit_pool_to_fd_limit(4096) == 4096
    assert calls and calls[0][1][0] == 4 * 4096 + _app._FD_RESERVE
    monkeypatch.setattr(resource, "getrlimit", lambda _: (1024, 1024))
    assert _app._fit_pool_to_fd_limit(4096) == 64


@pytest.mark.parametrize("served", [{"raw_stream_client": False}], indirect=True)
def test_httpx_stream_client_still_works_when_selected(served):
    app, url, abort_file, _ = served
    with httpx.stream(
        "POST", f"{url}/v1/chat/completions", json={"stream": True, "n_chunks": 3}
    ) as r:
        chunks = [c for c in r.iter_lines() if c.startswith("data:")]
    assert chunks[-1] == "data: [DONE]"
    assert app._usage_totals["completion_tokens"] == 5
    with httpx.stream(
        "POST", f"{url}/v1/chat/completions", json={"stream": True, "n_chunks": 100}
    ) as r:
        next(r.iter_lines())
    assert _wait_for(abort_file)


def test_kv_cpu_offload_adds_connector_sized_from_container_ram(monkeypatch):
    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 100 << 30)

    class A(LLMApp):
        model = "m"
        engine_presets = False
        kv_cpu_offload = 0.5

    cmd = A().engine_command("/m")
    cfg = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
    assert cfg["kv_connector"] == "OffloadingConnector"
    assert cfg["kv_connector_extra_config"]["cpu_bytes_to_use"] == 50 << 30


def test_kv_cpu_offload_off_by_default_and_user_flag_wins(monkeypatch):
    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 100 << 30)

    class Off(LLMApp):
        model = "m"
        engine_presets = False

    class User(LLMApp):
        model = "m"
        engine_presets = False
        kv_cpu_offload = 0.5
        engine_args = ["--kv-transfer-config", "{}"]

    assert "--kv-transfer-config" not in Off().engine_command("/m")
    cmd = User().engine_command("/m")
    assert cmd.count("--kv-transfer-config") == 1 and "{}" in cmd


def test_kv_cpu_offload_rejects_bad_fraction_and_sglang(monkeypatch):
    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 100 << 30)

    with pytest.raises(ValueError, match="kv_cpu_offload"):

        class Bad(LLMApp):
            model = "m"
            engine_presets = False
            kv_cpu_offload = 1.5

    class Sg(LLMApp):
        model = "m"
        engine = "sglang"
        engine_presets = False
        kv_cpu_offload = 0.5

    assert "--kv-transfer-config" not in Sg().engine_command("/m")


def test_kv_cpu_offload_caps_fraction_and_detects_equals_form(monkeypatch):
    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 100 << 30)

    with pytest.raises(ValueError, match="kv_cpu_offload"):

        class High(LLMApp):
            model = "m"
            engine_presets = False
            kv_cpu_offload = 0.95

    class Eq(LLMApp):
        model = "m"
        engine_presets = False
        kv_cpu_offload = 0.5
        engine_args = ["--kv_transfer_config={}"]

    assert "--kv-transfer-config" not in Eq().engine_command("/m")


def test_config_key_ignores_kv_connector_size(monkeypatch):
    class A(LLMApp):
        model = "m"
        engine_presets = False
        kv_cpu_offload = 0.5

    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 100 << 30)
    small = A()._config_key("/m")
    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 200 << 30)
    assert A()._config_key("/m") == small


GIB = 1 << 30


def _cgroup_tree(tmp_path, proc_cgroup, files):
    root = tmp_path / "cgroup"
    for rel, value in files.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(value)
    root.mkdir(exist_ok=True)
    proc = tmp_path / "proc_cgroup"
    proc.write_text(proc_cgroup)
    return str(root), str(proc)


@pytest.mark.parametrize(
    "proc_cgroup, files, expected",
    [
        # cgroup v2, own cgroup resolved from /proc/self/cgroup.
        ("0::/pods/p1/ctr\n", {"pods/p1/ctr/memory.max": str(60 * GIB)}, 60 * GIB),
        # A parent's tighter limit binds, and memory.high counts.
        (
            "0::/pods/p1/ctr\n",
            {"pods/p1/ctr/memory.max": "max", "pods/p1/memory.max": str(50 * GIB)},
            50 * GIB,
        ),
        (
            "0::/pods/p1/ctr\n",
            {
                "pods/p1/ctr/memory.max": str(60 * GIB),
                "pods/p1/ctr/memory.high": str(40 * GIB),
            },
            40 * GIB,
        ),
        # A cgroup namespace shows the container's own cgroup as the root.
        ("0::/\n", {"memory.max": str(30 * GIB)}, 30 * GIB),
        # cgroup v1 memory controller.
        (
            "5:memory:/docker/abc\n0::/\n",
            {"memory/docker/abc/memory.limit_in_bytes": str(40 * GIB)},
            40 * GIB,
        ),
        # The fal runner case: files present but empty, or unlimited.
        ("0::/\n", {"memory.max": "", "memory.high": ""}, 0),
        (
            "5:memory:/\n",
            {
                "memory/memory.limit_in_bytes": "9223372036854771712",
                "memory.max": "max",
            },
            0,
        ),
        # Nothing readable at all.
        ("", {}, 0),
        # A malformed line does not hide the memory controller after it.
        (
            "garbage\n\n5:memory:/docker/abc\n",
            {"memory/docker/abc/memory.limit_in_bytes": str(40 * GIB)},
            40 * GIB,
        ),
        # Hybrid: v2 and v1 limits both count.
        (
            "0::/pods/p1\n5:memory:/docker/abc\n",
            {
                "pods/p1/memory.max": str(50 * GIB),
                "memory/docker/abc/memory.limit_in_bytes": str(45 * GIB),
            },
            45 * GIB,
        ),
        # No cgroup namespace: /proc shows a host path, but the mount is the
        # container's own cgroup, so its limit sits at the mount root.
        ("0::/docker/abc\n", {"memory.max": str(60 * GIB)}, 60 * GIB),
    ],
)
def test_container_ram_reads_the_tightest_cgroup_limit_never_meminfo(
    tmp_path, proc_cgroup, files, expected
):
    root, proc = _cgroup_tree(tmp_path, proc_cgroup, files)
    assert _app._container_ram_bytes(root, proc) == expected


@pytest.mark.parametrize(
    "gpu, num_gpus, per_gpu_gib",
    [
        (("NVIDIA RTX PRO 6000 Blackwell Server Edition", 97887), 1, 60),
        (("NVIDIA RTX PRO 6000 Blackwell Server Edition", 97887), 2, 60),
        (("NVIDIA H200", 143771), 1, 112),
        (("NVIDIA H200 NVL", 143771), 1, 112),
        (("NVIDIA B200", 183359), 1, 210),
        (("NVIDIA H100 80GB HBM3", 81559), 1, 60),
        (("NVIDIA H100 80GB HBM3 MIG 3g.40gb", 40192), 1, 16),
        (("NVIDIA A30", 24576), 1, 16),
        (None, 1, 16),
    ],
)
def test_kv_cpu_offload_without_cgroup_limit_uses_fal_ram_per_gpu(
    monkeypatch, capsys, gpu, num_gpus, per_gpu_gib
):
    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 0)

    class A(LLMApp):
        model = "m"
        engine_presets = False
        kv_cpu_offload = 0.5

    A.num_gpus = num_gpus
    monkeypatch.setattr(A, "_gpu_name", staticmethod(lambda: gpu))
    monkeypatch.setattr(A, "_gpu_type", staticmethod(lambda: "RTX"))
    app = A()
    cmd = app.engine_command("/m")
    app.engine_command("/m")
    cfg = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
    expected = int(per_gpu_gib * GIB * num_gpus * 0.5)
    assert cfg["kv_connector_extra_config"]["cpu_bytes_to_use"] == expected
    out = capsys.readouterr().out
    assert out.count("no container memory limit is readable") == 1


def test_kv_cpu_offload_gib_overrides_size_and_turns_tier_on(monkeypatch):
    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 100 * GIB)

    class Sized(LLMApp):
        model = "m"
        engine_presets = False
        kv_cpu_offload_gib = 24

    cmd = Sized().engine_command("/m")
    cfg = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
    assert cfg["kv_connector_extra_config"]["cpu_bytes_to_use"] == 24 * GIB

    class Off(Sized):
        kv_cpu_offload = 0

    assert "--kv-transfer-config" not in Off().engine_command("/m")

    for bad in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="kv_cpu_offload_gib"):

            class Bad(LLMApp):
                model = "m"
                engine_presets = False
                kv_cpu_offload_gib = bad


def test_kv_cpu_offload_gib_is_capped_by_a_readable_limit(monkeypatch, capsys):
    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 60 * GIB)

    class Big(LLMApp):
        model = "m"
        engine_presets = False
        kv_cpu_offload_gib = 100

    cmd = Big().engine_command("/m")
    cfg = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
    assert cfg["kv_connector_extra_config"]["cpu_bytes_to_use"] == int(60 * GIB * 0.8)
    assert "exceeds" in capsys.readouterr().out


def test_unused_kv_tier_is_not_sized(monkeypatch, capsys):
    # No warning or GPU probe when the tier would be dropped anyway.
    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 0)

    def no_probe():
        raise AssertionError("GPU probed for an unused tier")

    class Sg(LLMApp):
        model = "m"
        engine = "sglang"
        engine_presets = False
        kv_cpu_offload = 0.5

    class User(LLMApp):
        model = "m"
        engine_presets = False
        kv_cpu_offload = 0.5
        engine_args = ["--kv-transfer-config", "{}"]

    for cls in (Sg, User):
        monkeypatch.setattr(cls, "_gpu_name", staticmethod(no_probe))
        cls().engine_command("/m")
    assert "WARNING" not in capsys.readouterr().out
