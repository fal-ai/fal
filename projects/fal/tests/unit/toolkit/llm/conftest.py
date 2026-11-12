from __future__ import annotations

import asyncio
import sys
import threading
import time

import pytest
import uvicorn
from llm_helpers import FAKE, _free_port

from fal.toolkit.llm import LLMApp


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
