from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import Request

from fal.toolkit.llm import LLMApp
from fal.toolkit.llm._app import _has_files

FAKE = str(Path(__file__).parent / "fake_engine.py")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def isolate_agent_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IS_ISOLATE_AGENT", "1")


@pytest.fixture
def served(tmp_path, monkeypatch):
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

        def engine_command(self, model_path: str) -> list[str]:
            return [sys.executable, FAKE, "--port", str(self.engine_port)]

    app = FakeLLM()
    server = uvicorn.Server(
        uvicorn.Config(
            app._build_app(), host="127.0.0.1", port=port, log_level="warning"
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
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
        "requests": 1,
    }
    assert r.headers["x-fal-runner-hints"].startswith("pfx-")
    assert httpx.post(f"{url}/v1/completions", json={"prompt": "x"}).status_code == 200
    assert httpx.get(f"{url}/v1/models").json()["data"][0]["id"] == "fake"


def test_upstream_error_and_bad_body(served):
    _, url, _, _ = served
    r = httpx.post(f"{url}/v1/chat/completions", json={"fail": True})
    assert r.status_code == 400
    assert httpx.post(f"{url}/v1/chat/completions", content=b"nope").status_code == 400


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

    app._proc.kill()
    app._proc.wait()
    assert httpx.get(f"{url}/health").status_code == 503


def test_engine_logs_record_milestones_and_throttle_progress(capsys):
    script = (
        "import sys\n"
        "w = sys.stdout.write\n"
        "w('Loading shards 0/48\\rLoading shards 1/48\\rLoading shards 2/48\\r\\n')\n"
        "w('INFO Loading weights took 61.2 seconds\\n')\n"
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


def test_has_files(tmp_path):
    empty = tmp_path / "empty"
    (empty / "sub").mkdir(parents=True)
    full = tmp_path / "full"
    (full / "sub").mkdir(parents=True)
    (full / "sub" / "kernel.so").write_bytes(b"x")
    assert not _has_files(str(empty))
    assert _has_files(str(empty), str(full))


def test_endpoint_annotations_are_not_strings():
    # `from __future__ import annotations` would make FastAPI read `request` as
    # a query parameter once the module is pickled by value.
    assert LLMApp.chat_completions.__annotations__["request"] is Request
    assert LLMApp.completions.__annotations__["request"] is Request
