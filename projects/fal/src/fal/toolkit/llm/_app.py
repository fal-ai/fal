import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Any, ClassVar, Deque, Dict, List, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from fal.app import App, endpoint
from fal.toolkit.compilation import get_gpu_type
from fal.toolkit.llm._cache import apply_engine_cache_env
from fal.toolkit.llm._preread import DEFAULT_WORKERS, preread_weights

_METRIC_PREFIXES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:num_preemptions_total",
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
    "sglang:token_usage",
    "sglang:cache_hit_rate",
)
_METRIC_RE = re.compile(r"^([^\s{]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+|nan|NaN)$")
_STREAM_TAIL_BYTES = 8192
# Engine log lines that mark a boot phase, as (substring, phase). Each phase
# records its first occurrence; a needle that never matches just leaves the
# phase out of the report.
_BOOT_MILESTONES: Dict[str, tuple] = {
    "vllm": (
        ("Loading weights took", "weights_loaded"),
        ("Model loading took", "model_loaded"),
        ("Capturing CUDA graph", "cuda_graph_begin"),
        ("Graph capturing finished", "cuda_graph_end"),
        ("GPU KV cache size", "kv_cache_allocated"),
        ("Application startup complete", "server_ready"),
    ),
    "sglang": (
        ("Load weight end", "weights_loaded"),
        ("Capture cuda graph begin", "cuda_graph_begin"),
        ("Capture cuda graph end", "cuda_graph_end"),
        ("The server is fired up", "server_ready"),
    ),
}
# Progress bars redraw with \r; log at most one redraw per interval.
_PROGRESS_LOG_INTERVAL_S = 30.0
_LINE_SPLIT = re.compile(rb"(\r\n|\n|\r)")
_MAX_LINE_BYTES = 1 << 20


def _engine_version(engine: str) -> str:
    from importlib import metadata  # noqa: PLC0415

    try:
        return metadata.version(engine)
    except Exception:
        return "unknown"


def _parse_stream_usage(tail: bytes) -> Optional[Dict[str, Any]]:
    for raw in reversed(tail.decode("utf-8", errors="replace").splitlines()):
        line = raw.strip()
        if line.startswith("data:") and '"usage"' in line:
            try:
                usage = json.loads(line[5:]).get("usage")
            except ValueError:
                return None
            return usage if isinstance(usage, dict) else None
    return None


def _has_files(*dirs: str) -> bool:
    for d in dirs:
        for _, _, files in os.walk(d):
            if files:
                return True
    return False


async def _wait_for_disconnect(request: Request) -> None:
    # Blocking receive instead of Request.is_disconnected(): the latter polls
    # with a cancelled scope and never sees the event behind http middleware.
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


class LLMApp(App):
    """Serve a vLLM or SGLang engine behind OpenAI-compatible routes.

    Subclass it, set ``model`` (and usually ``machine_type``, ``num_gpus``,
    ``engine_args``), and deploy. The engine runs as a subprocess; the app
    handles readiness, ``/v1/chat/completions``, ``/v1/completions`` and
    ``/v1/models`` (streaming passthrough, engine abort on client disconnect),
    persistent compile caches, a parallel weight pre-read, and engine-metric
    based health. Override ``engine_command`` for full control of the launch.
    """

    engine: ClassVar[str] = "vllm"
    model: ClassVar[str] = ""
    served_model_name: ClassVar[Optional[str]] = None
    engine_args: ClassVar[List[str]] = []
    engine_env: ClassVar[Dict[str, str]] = {}
    engine_port: ClassVar[int] = 8000
    engine_startup_timeout_s: ClassVar[float] = 3300.0

    # Persistent caches under /data (None disables, a string overrides the
    # auto namespace of engine, version and GPU type).
    persistent_caches: ClassVar[bool] = True
    cache_namespace: ClassVar[Optional[str]] = None
    # Parallel safetensors pre-read; 0 disables.
    preread_workers: ClassVar[int] = DEFAULT_WORKERS
    # Ask the engine for a final usage chunk on streams so token counts are
    # visible to ``record_usage`` (clients see one extra empty-choices chunk).
    report_usage: ClassVar[bool] = True
    # Publish X-Fal-Runner-Hints derived from recent prompt prefixes.
    prefix_hints: ClassVar[bool] = False
    prefix_hint_chars: ClassVar[int] = 256

    keep_alive = 300

    def engine_command(self, model_path: str) -> List[str]:
        name = self._served_name()
        if self.engine == "vllm":
            head = ["vllm", "serve", model_path]
        elif self.engine == "sglang":
            head = [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                model_path,
            ]
        else:
            raise ValueError(f"unknown engine {self.engine!r}; override engine_command")
        return [
            *head,
            "--served-model-name",
            name,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.engine_port),
            *self.engine_args,
        ]

    def record_usage(self, path: str, usage: Dict[str, Any]) -> None:
        """Called with the engine's token ``usage`` for every completed request."""
        for key in ("prompt_tokens", "completion_tokens"):
            self._usage_totals[key] = self._usage_totals.get(key, 0) + int(
                usage.get(key) or 0
            )
        self._usage_totals["requests"] = self._usage_totals.get("requests", 0) + 1

    def provide_hints(self) -> List[str]:
        if not self.prefix_hints:
            raise NotImplementedError
        return list(getattr(self, "_recent_prefixes", ()))

    def _served_name(self) -> str:
        return self.served_model_name or self.model

    def setup(self) -> None:  # type: ignore[override]
        # Async setup is not needed: blocking steps run before the event loop
        # serves requests, and the readiness wait is a plain polling loop.
        import httpx  # noqa: PLC0415

        if not self.model:
            raise ValueError("LLMApp subclasses must set `model`")
        t0 = time.monotonic()
        self._setup_start = t0
        self._boot: Dict[str, Any] = {}
        self._usage_totals: Dict[str, int] = {}
        self._recent_prefixes: Deque[str] = deque(maxlen=8)
        self._engine_log_tail: Deque[str] = deque(maxlen=50)

        if self.persistent_caches:
            namespace = self.cache_namespace or "-".join(
                (self.engine, _engine_version(self.engine), self._gpu_type())
            )
            cache_env = apply_engine_cache_env(namespace)
            # Warm means a previous runner left compiled artifacts behind; the
            # boot report then shows what the cache saved.
            self._boot["compile_cache_warm"] = _has_files(
                *(v for k, v in cache_env.items() if not k.startswith("HF_"))
            )

        model_path = self._resolve_model()
        self._boot["resolve_s"] = time.monotonic() - t0
        if self.preread_workers > 0 and os.path.isdir(model_path):
            stats = preread_weights(model_path, workers=self.preread_workers)
            self._boot["preread_s"] = stats.seconds
            print(
                f"[llm-app] pre-read {stats.bytes / 1e9:.1f} GB in "
                f"{stats.seconds:.1f}s ({stats.gb_per_s:.2f} GB/s)"
            )

        self._engine_started = time.monotonic()
        self._proc = self._start_engine(model_path)
        base_url = f"http://127.0.0.1:{self.engine_port}"
        self._wait_ready(base_url)
        self._boot["engine_ready_s"] = time.monotonic() - self._engine_started
        self._boot["total_setup_s"] = time.monotonic() - t0
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(connect=30.0, read=3600.0, write=300.0, pool=60.0),
            limits=httpx.Limits(max_connections=512, max_keepalive_connections=128),
        )
        # Copy: the log pump thread may still add late milestones.
        print(f"[llm-app] boot report: {dict(self._boot)}")

    # Runners do not have `fal` installed: it is pickled by value, so only
    # module-level imports resolve. Keep fal imports out of method bodies.
    @staticmethod
    def _gpu_type() -> str:
        return get_gpu_type()

    def _resolve_model(self) -> str:
        if os.path.isdir(self.model):
            return self.model
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415
        except ImportError:
            return self.model
        return snapshot_download(self.model)

    def _start_engine(self, model_path: str) -> subprocess.Popen:
        cmd = self.engine_command(model_path)
        env = {**os.environ, **self.engine_env}
        env.setdefault("VLLM_NO_USAGE_STATS", "1")
        print(f"[llm-app] starting: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )

        threading.Thread(
            target=self._pump_engine_logs, args=(proc,), daemon=True
        ).start()
        return proc

    def _pump_engine_logs(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        milestones = _BOOT_MILESTONES.get(self.engine, ())
        last_progress = float("-inf")

        def handle(raw: bytes, progress: bool) -> None:
            nonlocal last_progress
            line = raw[:_MAX_LINE_BYTES].decode("utf-8", errors="replace").rstrip()
            if not line:
                return
            # Milestone times are seconds since setup start.
            for needle, phase in milestones:
                key = f"{phase}_s"
                if needle in line and key not in self._boot:
                    self._boot[key] = time.monotonic() - self._setup_start
            if progress:
                now = time.monotonic()
                if now - last_progress < _PROGRESS_LOG_INTERVAL_S:
                    return
                last_progress = now
            self._engine_log_tail.append(line)
            print(f"[{self.engine}] {line}")

        buf = b""
        for chunk in iter(lambda: proc.stdout.read1(65536), b""):  # type: ignore[union-attr]
            buf += chunk
            # Hold back a trailing \r: it may be the first half of \r\n.
            held = buf.endswith(b"\r")
            parts = _LINE_SPLIT.split(buf[:-1] if held else buf)
            buf = parts.pop() + (b"\r" if held else b"")
            for raw, sep in zip(parts[::2], parts[1::2]):
                handle(raw, sep == b"\r")
            if len(buf) > _MAX_LINE_BYTES:
                handle(buf, False)
                buf = b""
        handle(buf, False)

    def _wait_ready(self, base_url: str) -> None:
        import httpx  # noqa: PLC0415

        deadline = time.monotonic() + self.engine_startup_timeout_s
        while time.monotonic() < deadline:
            code = self._proc.poll()
            if code is not None:
                tail = "\n".join(list(self._engine_log_tail)[-10:])
                raise RuntimeError(f"{self.engine} exited with code {code}\n{tail}")
            try:
                if httpx.get(f"{base_url}/health", timeout=5.0).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1.0)
        raise RuntimeError(
            f"{self.engine} not ready within {self.engine_startup_timeout_s}s"
        )

    def teardown(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()

    async def _engine_stats(self) -> Dict[str, float]:
        text = (await self._client.get("/metrics", timeout=5.0)).text
        stats: Dict[str, float] = {}
        for line in text.splitlines():
            if line.startswith(_METRIC_PREFIXES):
                m = _METRIC_RE.match(line)
                if m:
                    stats[m.group(1)] = stats.get(m.group(1), 0.0) + float(m.group(2))
        return stats

    async def health(self) -> Dict[str, Any]:  # type: ignore[override]
        if self._proc.poll() is not None:
            raise HTTPException(503, f"{self.engine} exited ({self._proc.returncode})")
        try:
            resp = await self._client.get("/health", timeout=10.0)
        except Exception as exc:
            raise HTTPException(503, f"{self.engine} unreachable: {exc}") from exc
        if resp.status_code != 200:
            raise HTTPException(503, f"{self.engine} unhealthy ({resp.status_code})")
        try:
            stats = await self._engine_stats()
        except Exception:
            stats = {}
        return {"status": "ok", "version": self.version, "engine": stats}

    @endpoint("/v1/models")
    async def models(self) -> Response:
        upstream = await self._client.get("/v1/models")
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type="application/json",
        )

    async def engine_stats(self) -> JSONResponse:
        try:
            stats = await self._engine_stats()
        except Exception:
            stats = {}
        return JSONResponse(
            {
                "model": self.model,
                "engine": self.engine,
                "stats": stats,
                "usage_totals": self._usage_totals,
                "boot": dict(self._boot),
            }
        )

    def _add_extra_routes(self, app: Any) -> None:
        super()._add_extra_routes(app)
        # @endpoint routes are POST only; OpenAI clients GET these.
        app.add_api_route("/v1/models", self.models, methods=["GET"])
        app.add_api_route("/engine-stats", self.engine_stats, methods=["GET"])

    @endpoint("/v1/chat/completions")
    async def chat_completions(self, request: Request) -> Response:
        return await self._proxy(request, "/v1/chat/completions")

    @endpoint("/v1/completions")
    async def completions(self, request: Request) -> Response:
        return await self._proxy(request, "/v1/completions")

    def _note_prefix(self, payload: Dict[str, Any]) -> None:
        if not self.prefix_hints:
            return
        text = json.dumps(payload.get("messages") or payload.get("prompt") or "")
        digest = hashlib.sha1(text[: self.prefix_hint_chars].encode()).hexdigest()
        hint = f"pfx-{digest[:16]}"
        if hint in self._recent_prefixes:
            self._recent_prefixes.remove(hint)
        self._recent_prefixes.append(hint)

    async def _proxy(self, request: Request, path: str) -> Response:
        try:
            payload: Dict[str, Any] = await request.json()
        except ValueError:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse({"error": "body must be an object"}, status_code=400)
        payload["model"] = self._served_name()
        self._note_prefix(payload)

        if payload.get("stream") is not True:
            return await self._proxy_unary(request, path, payload)

        if self.report_usage and "stream_options" not in payload:
            payload["stream_options"] = {"include_usage": True}
        upstream = await self._client.send(
            self._client.build_request("POST", path, json=payload), stream=True
        )
        if upstream.status_code != 200:
            body = await upstream.aread()
            await upstream.aclose()
            return Response(
                content=body,
                status_code=upstream.status_code,
                media_type="application/json",
            )

        async def relay():
            tail = b""
            try:
                async for chunk in upstream.aiter_raw():
                    tail = (tail + chunk)[-_STREAM_TAIL_BYTES:]
                    yield chunk
            finally:
                # Closing the upstream connection makes the engine abort the
                # request, which also covers client disconnects.
                await upstream.aclose()
                usage = _parse_stream_usage(tail)
                if usage:
                    self.record_usage(path, usage)

        return StreamingResponse(relay(), media_type="text/event-stream")

    async def _proxy_unary(
        self, request: Request, path: str, payload: Dict[str, Any]
    ) -> Response:
        task = asyncio.ensure_future(self._client.post(path, json=payload))
        watcher = asyncio.ensure_future(_wait_for_disconnect(request))
        try:
            await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            task.cancel()
            raise
        finally:
            watcher.cancel()
        if not task.done():
            # uvicorn does not cancel the handler on disconnect; cancel the
            # upstream call so the engine aborts the request.
            task.cancel()
            return Response(status_code=499)
        upstream = task.result()
        if upstream.status_code == 200:
            try:
                usage = upstream.json().get("usage")
            except ValueError:
                usage = None
            if isinstance(usage, dict):
                self.record_usage(path, usage)
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type="application/json",
        )
