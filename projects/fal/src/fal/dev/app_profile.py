"""Torch profiler capture for `fal dev app profile`, inside the resident app process.

The holder arms a profile for one endpoint path, then sends one request. Each
endpoint is wrapped (see ``wrap_routes``) so the first matching request after
arming runs under ``torch.profiler`` in the thread that runs the endpoint: the
profiler records CPU ops per thread, so starting it anywhere else would miss
the ops of sync endpoints, which run in a worker thread.

Artifacts go to the armed profile's folder: ``trace.json.gz`` (Chrome trace
format, opens in Perfetto), ``ops.txt`` (``key_averages()`` table) and
``status.json``.

Default request bodies for endpoints with a Pydantic input model come from
``default_body``: required fields from their first example, else a value that
fits the type and constraints.
"""

from __future__ import annotations

import contextlib
import enum
import functools
import inspect
import json
import shutil
import threading
import time
import traceback
import types
from pathlib import Path
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

MODES = ("light", "standard", "deep")
TABLE_ROWS = 50
RESULTS_KEPT = 10
# An armed profile no request claimed within this long is dropped.
ARM_SECONDS = 3600
TRACE_NAME = "trace.json.gz"
OPS_NAME = "ops.txt"
STATUS_NAME = "status.json"


class Profiler:
    """One armed profile at a time; the first matching request claims it."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.armed: dict[str, Any] | None = None
        self.results: dict[str, dict[str, Any]] = {}

    def arm(self, spec: dict[str, Any]) -> dict[str, Any]:
        if spec.get("mode") not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        with self.lock:
            self.armed = {**spec, "expires": time.monotonic() + ARM_SECONDS}
        return {"armed": spec["id"]}

    def claim(self, path: str) -> dict[str, Any] | None:
        with self.lock:
            spec = self.armed
            if spec is not None and spec["expires"] < time.monotonic():
                self.armed = spec = None
            if spec is None or spec["path"] != path:
                return None
            self.armed = None
            return spec

    def result(self, profile_id: str) -> dict[str, Any]:
        with self.lock:
            if self.armed is not None and self.armed["id"] == profile_id:
                # The request never reached the endpoint (a 422, a wrong path).
                self.armed = None
                return {"status": "not_captured"}
        return self.results.get(profile_id, {"status": "unknown"})

    @contextlib.contextmanager
    def capture(self, spec: dict[str, Any]):
        folder = Path(spec["dir"])
        folder.mkdir(parents=True, exist_ok=True)
        status: dict[str, Any] = {"id": spec["id"], "path": spec["path"]}
        status["mode"] = spec["mode"]
        try:
            profiler, has_cuda = _start(spec["mode"])
        except Exception:
            status.update(status="error", error=traceback.format_exc(limit=3))
            self._finish(folder, status)
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
            self._finish(folder, status)

    def _finish(self, folder: Path, status: dict[str, Any]) -> None:
        with self.lock:
            self.results[status["id"]] = status
            while len(self.results) > RESULTS_KEPT:
                del self.results[next(iter(self.results))]
        # Never fail the app's request over the profile's own bookkeeping.
        try:
            (folder / STATUS_NAME).write_text(json.dumps(status, indent=2))
        except OSError:
            pass


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
    try:
        wrapper.__signature__ = inspect.signature(endpoint, eval_str=True)  # type: ignore[attr-defined]
    except Exception:
        pass
    return wrapper


# Default request bodies, from the endpoint's Pydantic input model.


def _model(annotation: Any) -> Any:
    from pydantic import BaseModel  # noqa: PLC0415

    origin = get_origin(annotation)
    if origin is Annotated:
        return _model(get_args(annotation)[0])
    if origin in (Union, types.UnionType):
        for option in get_args(annotation):
            candidate = _model(option)
            if candidate is not None:
                return candidate
        return None
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _number(field: Any, integer: bool) -> int | float:
    value: int | float = 1
    for metadata in field.metadata:
        if getattr(metadata, "ge", None) is not None:
            value = max(value, metadata.ge)
        if getattr(metadata, "gt", None) is not None:
            value = max(value, metadata.gt + (1 if integer else 0.1))
    return int(value) if integer else float(value)


def _length(field: Any) -> int:
    minimum, maximum = 1, None
    for metadata in field.metadata:
        if getattr(metadata, "min_length", None) is not None:
            minimum = max(minimum, int(metadata.min_length))
        if getattr(metadata, "max_length", None) is not None:
            maximum = int(metadata.max_length)
    return minimum if maximum is None else min(minimum, maximum)


def _value(annotation: Any, field: Any) -> Any:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return _value(args[0], field)
    if origin in (Union, types.UnionType):
        for option in args:
            if option is type(None):
                continue
            try:
                return _value(option, field)
            except ValueError:
                continue
        raise ValueError("no usable union member")
    if origin is Literal:
        return args[0]
    if origin in (list, set, tuple):
        if origin is tuple and args and args[-1] is not Ellipsis:
            return [_value(item, field) for item in args]
        return [_value(args[0] if args else str, field) for _ in range(_length(field))]
    if origin is dict:
        return {}
    model = _model(annotation)
    if model is not None:
        return default_body(model)
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return next(iter(annotation)).value
    if annotation is bool:
        return False
    if annotation is int:
        return _number(field, integer=True)
    if annotation is float:
        return _number(field, integer=False)
    if annotation is str:
        return "test"
    if isinstance(annotation, type) and "url" in annotation.__name__.lower():
        return "https://example.com"
    raise ValueError(f"unsupported type {annotation!r}")


def default_body(model: Any) -> dict[str, Any]:
    """A valid body: required fields from examples (or made up), defaults kept."""
    values = {}
    for name, field in model.model_fields.items():
        if not field.is_required():
            continue
        if field.examples:
            values[name] = field.examples[0]
            continue
        try:
            values[name] = _value(field.annotation, field)
        except ValueError as exc:
            raise ValueError(
                f"{model.__name__}.{name} has no example to build a request from"
            ) from exc
    return model.model_validate(values).model_dump(mode="json")


def _input_model(endpoint: Any) -> tuple[bool, Any]:
    """(takes a body, its Pydantic model or None)."""
    try:
        hints = get_type_hints(endpoint)
    except Exception:
        hints = {}
    parameters = inspect.signature(endpoint).parameters
    names = [n for n in ("input", "payload", "request") if n in parameters]
    names += [n for n in parameters if n not in names and n != "response"]
    for name in names:
        model = _model(hints.get(name, parameters[name].annotation))
        if model is not None:
            return True, model
    return bool(parameters), None


def endpoints(routes: dict[Any, Any]) -> list[dict[str, Any]]:
    """Profileable endpoints and the default body for each (or why there is none)."""
    found = []
    for signature, endpoint in routes.items():
        if signature.is_websocket or _streaming(endpoint):
            continue
        entry: dict[str, Any] = {"path": signature.path}
        try:
            takes_body, model = _input_model(endpoint)
        except Exception as exc:
            entry["error"] = f"could not read its signature: {exc}"
            found.append(entry)
            continue
        if model is not None:
            try:
                entry["body"] = default_body(model)
            except Exception as exc:
                entry["error"] = str(exc)
        elif takes_body:
            entry["error"] = "no Pydantic input model to build a request from"
        else:
            entry["body"] = {}
        found.append(entry)
    return found


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def prune(root: Path, keep: int) -> None:
    """Keep the newest ``keep`` profile folders on the runner."""
    if not root.is_dir():
        return
    folders = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=_mtime,
        reverse=True,
    )
    for folder in folders[keep:]:
        shutil.rmtree(folder, ignore_errors=True)
