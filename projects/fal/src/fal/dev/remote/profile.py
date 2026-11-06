"""Torch profiler capture for `fal dev app profile`, inside the resident app process.

The holder arms a profile for one endpoint path, then sends one request. Each
endpoint is wrapped (see ``wrap_routes``) so the first matching request after
arming runs under ``torch.profiler`` in the thread that runs the endpoint: the
profiler records CPU ops per thread, so starting it anywhere else would miss
the ops of sync endpoints, which run in a worker thread.

Artifacts go to the armed profile's folder: ``trace.json.gz`` (Chrome trace
format, opens in Perfetto) and ``ops.txt`` (``key_averages()`` table).
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import threading
import time
import traceback
from pathlib import Path
from typing import Any

MODES = ("light", "standard", "deep")
TABLE_ROWS = 50
TRACE_NAME = "trace.json.gz"
OPS_NAME = "ops.txt"


class Profiler:
    """One armed profile at a time; the first matching request claims it."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.armed: dict[str, Any] | None = None
        self.finished: dict[str, Any] = {}

    def arm(self, spec: dict[str, Any]) -> dict[str, Any]:
        if spec.get("mode") not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        with self.lock:
            self.armed = spec
        return {"armed": spec["id"]}

    def claim(self, path: str) -> dict[str, Any] | None:
        with self.lock:
            spec = self.armed
            if spec is None or spec["path"] != path:
                return None
            self.armed = None
            return spec

    def result(self, profile_id: str) -> dict[str, Any]:
        """The profile's outcome; also disarms it if no request claimed it."""
        with self.lock:
            if self.armed is not None and self.armed["id"] == profile_id:
                # The request never reached the endpoint (a 422, a wrong path).
                self.armed = None
                return {"status": "not_captured"}
            if self.finished.get("id") == profile_id:
                return self.finished
        return {"status": "unknown"}

    @contextlib.contextmanager
    def capture(self, spec: dict[str, Any]):
        folder = Path(spec["dir"])
        folder.mkdir(parents=True, exist_ok=True)
        status: dict[str, Any] = {
            "id": spec["id"],
            "path": spec["path"],
            "mode": spec["mode"],
        }
        try:
            profiler, has_cuda = _start(spec["mode"])
        except Exception:
            status.update(status="error", error=traceback.format_exc(limit=3))
            self._finish(status)
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            try:
                status.update(_stop(profiler, has_cuda, folder, spec))
                status.update(status="ok", elapsed_seconds=round(elapsed, 4))
            except Exception:
                status.update(status="error", error=traceback.format_exc(limit=3))
            self._finish(status)

    def _finish(self, status: dict[str, Any]) -> None:
        with self.lock:
            self.finished = status


def _start(mode: str):
    import torch  # noqa: PLC0415

    activity = torch.profiler.ProfilerActivity
    has_cuda = torch.cuda.is_available()
    deep = mode == "deep"
    activities = []
    if mode != "light" or not has_cuda:
        activities.append(activity.CPU)
    if has_cuda:
        activities.append(activity.CUDA)
        torch.cuda.reset_peak_memory_stats()
    profiler = torch.profiler.profile(
        activities=activities,
        record_shapes=deep,
        profile_memory=deep,
        with_stack=deep,
        with_flops=deep,
    )
    profiler.start()
    return profiler, has_cuda


def _stop(profiler, has_cuda: bool, folder: Path, spec: dict[str, Any]) -> dict:
    import torch  # noqa: PLC0415

    # stop() waits for the GPU work the request queued.
    profiler.stop()
    result: dict[str, Any] = {"cuda": has_cuda}
    if has_cuda:
        result["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
    started = time.perf_counter()
    trace = folder / TRACE_NAME
    # A .gz suffix makes torch write the trace gzipped.
    profiler.export_chrome_trace(str(trace))
    result["trace_bytes"] = trace.stat().st_size
    result["export_seconds"] = round(time.perf_counter() - started, 2)
    averages = profiler.key_averages()
    sort_by = "self_cuda_time_total" if has_cuda else "self_cpu_time_total"
    (folder / OPS_NAME).write_text(
        averages.table(sort_by=sort_by, row_limit=TABLE_ROWS) + "\n"
    )
    result["top_ops"] = averages.table(
        sort_by=sort_by, row_limit=int(spec.get("top", 15))
    )
    return result


def _streaming(endpoint: Any) -> bool:
    function = inspect.unwrap(endpoint)
    return inspect.isgeneratorfunction(function) or inspect.isasyncgenfunction(function)


def wrap_routes(routes: dict[Any, Any], profiler: Profiler) -> dict[Any, Any]:
    """The app's routes, each able to run under an armed profile."""
    wrapped = {}
    for signature, endpoint in routes.items():
        if signature.is_websocket or _streaming(endpoint):
            wrapped[signature] = endpoint
        else:
            wrapped[signature] = _wrap(signature.path, endpoint, profiler)
    return wrapped


def _wrap(path: str, endpoint: Any, profiler: Profiler) -> Any:
    if inspect.iscoroutinefunction(endpoint):

        @functools.wraps(endpoint)
        async def wrapper(*args, **kwargs):
            spec = profiler.claim(path)
            if spec is None:
                return await endpoint(*args, **kwargs)
            with profiler.capture(spec):
                return await endpoint(*args, **kwargs)

    else:

        @functools.wraps(endpoint)
        def wrapper(*args, **kwargs):
            spec = profiler.claim(path)
            if spec is None:
                return endpoint(*args, **kwargs)
            with profiler.capture(spec):
                return endpoint(*args, **kwargs)

    # FastAPI resolves string annotations against the wrapper's module, so
    # hand it the endpoint's signature with the annotations already resolved.
    with contextlib.suppress(Exception):
        wrapper.__signature__ = inspect.signature(endpoint, eval_str=True)  # type: ignore[attr-defined]
    return wrapper
