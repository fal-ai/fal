"""Helpers shared by the LLMApp test modules. Fixtures live in conftest.py."""

from __future__ import annotations

import asyncio
import os
import socket
import sys
import time
from pathlib import Path

from fal.toolkit.llm import LLMApp, _app, _cache

FAKE = str(Path(__file__).parent / "fake_engine.py")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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


def _make_probe(tmp_path, monkeypatch, **attrs):
    """An LLMApp with caches redirected under tmp_path and a stub engine."""
    monkeypatch.setattr(_cache, "HF_HOME_DIR", tmp_path / "hf")
    monkeypatch.setattr(_cache, "ENGINE_CACHE_ROOT", tmp_path / "engine")
    monkeypatch.setattr(_app, "namespace_dir", lambda ns: tmp_path / "engine" / ns)
    for key in list(os.environ):
        if key.startswith(
            ("HF_", "VLLM_", "TRITON_", "TORCHINDUCTOR_", "TORCH_EXT", "CUTE_DSL")
        ):
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


def _run(app, coro, timeout: float = 10.0):
    """Run ``coro`` on the event loop of an app from the ``served`` fixture."""
    return asyncio.run_coroutine_threadsafe(coro, app._test_loop).result(timeout)
