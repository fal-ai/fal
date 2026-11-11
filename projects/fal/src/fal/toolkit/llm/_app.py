import asyncio
import contextvars
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import warnings
from collections import OrderedDict, deque
from datetime import timezone
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
)

import anyio
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from fal.app import DEFAULT_PLATFORM_STARTUP_TIMEOUT, App, endpoint
from fal.sdk import HealthCheck
from fal.toolkit.compilation import get_gpu_type
from fal.toolkit.llm._cache import (
    advance_snapshot_marker,
    apply_engine_cache_env,
    namespace_dir,
    read_marker,
    snapshot_intact,
    snapshot_marker_path,
    write_marker,
)
from fal.toolkit.llm._preread import DEFAULT_WORKERS, preread_weights
from fal.toolkit.llm._presets import (
    AGENT_KV_CPU_OFFLOAD,
    AGENT_PROFILE,
    B200,
    H100,
    PROFILES,
    RTX_PRO_6000,
    Machine,
    checkpoint_for,
    detect_gpu,
    engine_arg_warnings,
    engine_preset_args,
    engine_preset_env,
    load_model_config,
    merge_engine_args,
    model_facts,
    short_gpu_type,
)
from fal.toolkit.llm._upstream import UpstreamBusy, open_stream

# vLLM speculative decoding counters (cumulative since engine start).
_SPEC_ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
_SPEC_DRAFTED = "vllm:spec_decode_num_draft_tokens_total"
# Cumulative token counters that move while the engine makes progress, and the
# gauges of requests it holds (engine stall check in health). vLLM only: SGLang
# counts tokens when a request finishes, so one long stream looks stalled.
_PROGRESS_COUNTERS = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
)
_PENDING_GAUGES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
)
# Tells the gateway to probe the runner's health endpoint before deciding
# what an error response means for the runner, instead of terminating it on a
# 503 or counting a 5xx toward termination.
_HEALTH_CHECK_HEADERS = {"X-Fal-Runner-Health-Check": "true"}
# Inner waits of one health probe; together under the probe's own timeout.
_ENGINE_HEALTH_TIMEOUT_S = 5.0
_ENGINE_METRICS_TIMEOUT_S = 3.0
# Consecutive inference 5xx replies further apart than this do not add up.
_ENGINE_ERROR_WINDOW_S = 60.0
_HEALTH_CHECK_TIMEOUT_S = 10
_METRIC_PREFIXES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:num_preemptions_total",
    _SPEC_ACCEPTED,
    _SPEC_DRAFTED,
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
    "sglang:token_usage",
    "sglang:cache_hit_rate",
    *_PROGRESS_COUNTERS,
)
# Fractions, averaged across engine replicas instead of summed.
_MEAN_METRICS = frozenset(
    (
        "vllm:kv_cache_usage_perc",
        "vllm:gpu_cache_usage_perc",
        "sglang:token_usage",
        "sglang:cache_hit_rate",
    )
)
_METRIC_RE = re.compile(r"^([^\s{]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+|nan|NaN)$")
# Session ids remembered for replica routing (least recently used dropped).
_SESSION_ROUTES_MAX = 4096
# Responses API ids remembered with the replica that created them, so a
# ``previous_response_id`` continuation reaches the engine holding its state.
_RESPONSE_ROUTES_MAX = 65536
# Extra requests in flight a session's replica may hold over the least loaded
# one before the session moves.
_STICKY_SLACK = 4
_STREAM_TAIL_BYTES = 8192
_ENGINE_CONNECT_TIMEOUT_S = 30.0
# An engine error body is small; a stalled one must not hold the request.
_ERROR_BODY_TIMEOUT_S = 30.0
# Per-request deadline override (seconds from arrival) and the gateway's
# remaining time budget (seconds, relative; the gateway overwrites inbound
# values, so it can be trusted).
_DEADLINE_HEADER = "x-fal-request-deadline"
_GATEWAY_REMAINING_HEADER = "x-fal-request-timeout-remaining"
# Engine log lines that mark a boot phase, as (substring, phase). Each phase
# records its first occurrence, except phases ending in "_end", which record
# the last (vLLM captures graphs once to size memory, then again for real). A
# needle that never matches just leaves the phase out of the report.
_BOOT_MILESTONES: Dict[str, tuple] = {
    "vllm": (
        ("Initializing a V1 LLM engine", "engine_core_start"),
        ("Loading weights took", "weights_loaded"),
        ("Model loading took", "model_loaded"),
        ("torch.compile took", "compile_loaded"),
        ("Capturing CUDA graph", "cuda_graph_begin"),
        ("Graph capturing finished", "cuda_graph_end"),
        ("GPU KV cache size", "kv_cache_allocated"),
        ("Autotuning process starts", "autotune_begin"),
        ("Autotuning process ends", "autotune_end"),
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
# Head room kept below the platform startup timeout so a stage that finishes in
# time can still fail with LLMApp's own report. A single blocking stage that
# overruns the budget is only reported by the watchdog, not interrupted.
_STARTUP_MARGIN_S = 30.0
_STARTUP_MARGIN_FRACTION = 0.1
# Terminate wait when failing setup; teardown uses the termination window.
_FAIL_STOP_TIMEOUT_S = 5.0
# Time left for stopping the engine inside the runner termination window, plus
# slack for streams to unwind before teardown. A full upload-policy drain (up
# to 5s) does not fit the default window; the platform kill is the fallback.
_ENGINE_STOP_BUDGET_S = 3.0
_MAX_ENGINE_STOP_WAIT_S = 30.0
_DRAIN_SLACK_S = 1.0
# A forwarded chunk that already carries a non-null finish reason.
_FINISH_REASON_RE = re.compile(rb'"finish_reason":\s*"')
# Hint ids the runner can advertise: header-safe, at most 64 chars.
_HINT_RE = re.compile(r"[A-Za-z0-9._:-]{1,64}")
# Platform window between SIGTERM and SIGKILL when the app sets none.
_DEFAULT_TERMINATION_GRACE_S = 5.0
_SHUTTING_DOWN_BODY = {
    "error": {"message": "runner is shutting down", "type": "shutting_down"}
}
_HUB_TIMEOUT_S = 10.0
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_UPSTREAM_GAUGE = "fal_app_llm_model_upstream_newer"


def _engine_version(engine: str) -> str:
    from importlib import metadata  # noqa: PLC0415

    try:
        return metadata.version(engine)
    except Exception:
        return "unknown"


def _torch_tag() -> str:
    """Torch build (version and CUDA) as a cache-key part: kernels compiled by
    one build are not valid for another, and an engine version alone does not
    pin it across images."""
    from importlib import metadata  # noqa: PLC0415

    try:
        return metadata.version("torch")
    except Exception:
        return "unknown"


# ``"usage": {...}`` with one level of nesting, for a Responses API terminal
# event whose start fell out of the stream tail (it carries the whole output).
_USAGE_TAIL_RE = re.compile(r'"usage":\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})')
# Responses API stream events that end a response (no ``[DONE]`` follows).
_RESPONSES_TERMINAL_RE = re.compile(
    rb'"type":\s*"response\.(?:completed|incomplete|failed)"'
)


def _parse_stream_usage(tail: bytes) -> Optional[Dict[str, Any]]:
    text = tail.decode("utf-8", errors="replace")
    for raw in reversed(text.splitlines()):
        line = raw.strip()
        if line.startswith("data:") and '"usage"' in line:
            try:
                event = json.loads(line[5:])
            except ValueError:
                break
            if not isinstance(event, dict):
                break
            # Chat and completions: top level. Responses: under ``response``.
            usage = event.get("usage")
            if usage is None and isinstance(event.get("response"), dict):
                usage = event["response"].get("usage")
            return usage if isinstance(usage, dict) else None
    # The terminal Responses event can outgrow the tail and lose its start.
    match = None
    for match in _USAGE_TAIL_RE.finditer(text):
        pass
    if match:
        try:
            usage = json.loads(match.group(1))
        except ValueError:
            return None
        return usage if isinstance(usage, dict) else None
    return None


def _cached_tokens(usage: Dict[str, Any]) -> int:
    """Prompt tokens served from the prefix cache (chat or Responses shape)."""
    for key in ("prompt_tokens_details", "input_tokens_details"):
        details = usage.get(key)
        if isinstance(details, dict):
            return int(details.get("cached_tokens") or 0)
    return 0


def _hub_commit(
    model: str, revision: Optional[str], timeout: float
) -> Optional[Tuple[str, Optional[str]]]:
    """(commit sha, ISO-8601 UTC commit date or None) of a Hub revision."""
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415

        info = HfApi().model_info(model, revision=revision, timeout=timeout)
        stamp = info.last_modified
        date = None
        if stamp is not None:
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            date = stamp.astimezone(timezone.utc).isoformat()
        return (info.sha, date) if info.sha else None
    except Exception:
        return None


def _set_upstream_newer_gauge(value: float) -> None:
    try:
        from prometheus_client import REGISTRY, Gauge  # noqa: PLC0415

        try:
            gauge = Gauge(_UPSTREAM_GAUGE, "1 if the Hub has a newer model commit")
        except ValueError:  # already registered
            gauge = REGISTRY._names_to_collectors[_UPSTREAM_GAUGE]
        gauge.set(value)
    except Exception as exc:
        print(f"[llm-app] could not set {_UPSTREAM_GAUGE}: {exc!r}")


def _parse_seconds(raw: Optional[str]) -> Optional[float]:
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _deadline_budget(
    headers: Any, default: Optional[float], margin: float
) -> Optional[float]:
    """Seconds a request may take from arrival, or None for no deadline.

    The override header can shorten the class default, never extend it; the
    gateway's remaining time (minus a margin) caps both. Raises ValueError on
    a bad override.
    """
    candidates: List[float] = []
    raw = headers.get(_DEADLINE_HEADER)
    if raw is not None:
        override = _parse_seconds(raw)
        if override is None or override <= 0:
            raise ValueError(f"invalid {_DEADLINE_HEADER} header: {raw!r}")
        candidates.append(override)
    if default is not None:
        candidates.append(default)
    remaining = _parse_seconds(headers.get(_GATEWAY_REMAINING_HEADER))
    if remaining is not None:
        candidates.append(remaining - margin)
    return min(candidates, default=None)


_FD_RESERVE = 1536  # engine log pipes, files, listeners, slack


def _fit_pool_to_fd_limit(wanted: int) -> int:
    """Largest pool <= ``wanted`` that leaves fds for the inbound sockets.

    Streams and unary requests can each fill ``pool`` upstream sockets, and as
    many requests may hold (or wait with) a client socket, so the pool is capped
    at about a quarter of the descriptor limit. The soft limit is raised toward
    the hard one first; without that a low default turns overload into EMFILE
    instead of pool backpressure.
    """
    try:
        import resource  # noqa: PLC0415

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = 4 * wanted + _FD_RESERVE
        if soft != resource.RLIM_INFINITY and soft < target:
            new = target if hard == resource.RLIM_INFINITY else min(target, hard)
            if new > soft:
                resource.setrlimit(resource.RLIMIT_NOFILE, (new, hard))
                soft = new
        if soft == resource.RLIM_INFINITY:
            return wanted
        return max(64, min(wanted, (soft - _FD_RESERVE) // 4))
    except Exception:
        return min(wanted, 512)


def _event_cut(buf: bytes) -> int:
    """Length of the leading run of whole SSE events in ``buf`` (0 if none)."""
    lf, crlf = buf.rfind(b"\n\n"), buf.rfind(b"\r\n\r\n")
    return max(lf + 2 if lf >= 0 else 0, crlf + 4 if crlf >= 0 else 0)


def _sse_response_id(events: bytes) -> Optional[str]:
    """The ``response.id`` of the first Responses API event in ``events``
    that has one."""
    for line in events.splitlines():
        if not line.startswith(b"data:"):
            continue
        try:
            event = json.loads(line[5:])
        except ValueError:
            continue
        response = event.get("response") if isinstance(event, dict) else None
        if isinstance(response, dict) and isinstance(response.get("id"), str):
            return response["id"]
    return None


def _expired(deadline: Optional[float]) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _remaining(deadline: Optional[float]) -> Optional[float]:
    """Seconds until ``deadline`` (never negative), or None for no deadline."""
    return None if deadline is None else max(deadline - time.monotonic(), 0.0)


def _same_repo(a: str, b: str) -> bool:
    return a.lower() == b.lower()


def _wait_limit(
    deadline: Optional[float], idle_s: Optional[float]
) -> Tuple[Optional[float], bool]:
    """Timeout for one wait on the engine, and whether the idle limit sets it."""
    remaining = _remaining(deadline)
    if idle_s is not None and (remaining is None or idle_s < remaining):
        return idle_s, True
    return remaining, False


def _error_body(message: str, kind: str) -> Dict[str, Any]:
    """An OpenAI-shaped error body."""
    return {"error": {"message": message, "type": kind}}


def _bad_request(message: str) -> JSONResponse:
    return JSONResponse(_error_body(message, "invalid_request_error"), status_code=400)


def _shutting_down_response() -> JSONResponse:
    return JSONResponse(
        _SHUTTING_DOWN_BODY,
        status_code=503,
        headers={"Retry-After": "1", "X-Fal-Needs-Retry": "true"},
    )


class _RelayResponse(StreamingResponse):
    """Streaming response that always runs ``cleanup``, even if never iterated.

    Starlette skips the body generator when the client disconnects before the
    first iteration, so cleanup cannot live in the generator's ``finally``.
    """

    def __init__(self, content: Any, cleanup: Callable[[], Awaitable[None]], **kw):
        super().__init__(content, **kw)
        self._cleanup = cleanup

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Unwinds a suspended generator; a no-op if it finished or never ran.
            try:
                aclose = getattr(self.body_iterator, "aclose", None)
                if aclose is not None:
                    await aclose()
            finally:
                await self._cleanup()


class _Stream:
    """An upstream streaming response and its once-only cleanup."""

    def __init__(
        self,
        app: "LLMApp",
        upstream: Any,
        path: str,
        abort: asyncio.Event,
        replica: Optional["_Replica"] = None,
    ) -> None:
        self.app = app
        self.upstream = upstream
        self.path = path
        self.abort = abort
        self.replica = replica
        self._recent: Deque[bytes] = deque()
        self._recent_len = 0
        self._settled = False
        self._upstream_closed = False

    @property
    def tail(self) -> bytes:
        """The last ``_STREAM_TAIL_BYTES`` bytes relayed."""
        return b"".join(self._recent)[-_STREAM_TAIL_BYTES:]

    @tail.setter
    def tail(self, value: bytes) -> None:
        self._recent = deque([value])
        self._recent_len = len(value)

    def feed(self, chunk: bytes) -> None:
        # Keeps whole chunks instead of re-slicing one tail buffer per chunk,
        # which copied the full tail for every few-hundred-byte event.
        self._recent.append(chunk)
        self._recent_len += len(chunk)
        while len(self._recent) > 1 and (
            self._recent_len - len(self._recent[0]) >= _STREAM_TAIL_BYTES
        ):
            self._recent_len -= len(self._recent.popleft())

    async def close(self) -> None:
        # Bookkeeping runs once; the upstream close is retried by the next
        # owner if a cancelled scope (client disconnect) interrupted it.
        if not self._settled:
            self._settled = True
            self.app._inflight.discard(self.abort)
            if self.replica is not None:
                self.replica.inflight -= 1
            self.app._count_if_cancelled()
            try:
                # Inside the try: a parser bug must not skip the upstream close.
                usage = _parse_stream_usage(self.tail)
                if usage:
                    self.app.record_usage(self.path, usage)
            except Exception as exc:
                # A subclass hook must not keep the engine request alive.
                print(f"[llm-app] record_usage failed: {exc!r}")
        if not self._upstream_closed:
            # Closing the upstream connection makes the engine abort the
            # request, which also covers client disconnects and shutdown.
            # Shielded: on a client disconnect this runs in a cancelled scope,
            # and an interrupted httpx close cannot be retried (the response
            # is marked closed first), which with httpcore < 1.0 left the
            # engine connection open. The shield covers anyio scope
            # cancellation (Starlette's task group on ASGI spec 2.3), not a
            # second native task cancel.
            with anyio.CancelScope(shield=True):
                await self.upstream.aclose()
            self._upstream_closed = True


async def _wait_for_disconnect(request: Request) -> None:
    # Blocking receive instead of Request.is_disconnected(): the latter polls
    # with a cancelled scope, which can miss the event.
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


def _parse_metrics(text: str) -> Dict[str, float]:
    """Engine metrics LLMApp reports, summed over label sets."""
    stats: Dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith(_METRIC_PREFIXES):
            m = _METRIC_RE.match(line)
            if m:
                stats[m.group(1)] = stats.get(m.group(1), 0.0) + float(m.group(2))
    return stats


def _merge_stats(per_engine: List[Dict[str, float]]) -> Dict[str, float]:
    """One view of several engines: counts add up, fractions average.
    Empty unless every engine answered: a partial sum would make the
    cumulative counters dip."""
    if not all(per_engine):
        return {}
    stats: Dict[str, float] = {}
    seen: Dict[str, int] = {}
    for engine_stats in per_engine:
        for key, value in engine_stats.items():
            stats[key] = stats.get(key, 0.0) + value
            seen[key] = seen.get(key, 0) + 1
    for key in _MEAN_METRICS & stats.keys():
        stats[key] /= seen[key]
    if stats.get(_SPEC_DRAFTED) and _SPEC_ACCEPTED in stats:
        stats["spec_decode_acceptance_rate"] = (
            stats[_SPEC_ACCEPTED] / stats[_SPEC_DRAFTED]
        )
    return stats


def _with_port(cmd: List[str], port: int) -> List[str]:
    """``cmd`` with its ``--port`` value replaced; ValueError if it has none."""
    out = list(cmd)
    for i, arg in enumerate(out):
        if arg == "--port" and i + 1 < len(out):
            out[i + 1] = str(port)
            return out
        if arg.startswith("--port="):
            out[i] = f"--port={port}"
            return out
    raise ValueError("engine_replicas > 1 needs a --port flag in engine_command")


def _split_devices(visible: Optional[str], num_gpus: int, replicas: int) -> List[str]:
    """CUDA_VISIBLE_DEVICES for each replica: consecutive equal slices."""
    ids = [d.strip() for d in (visible or "").split(",") if d.strip()]
    if not ids:
        ids = [str(i) for i in range(num_gpus)]
    if len(ids) != num_gpus or num_gpus % replicas:
        raise ValueError(
            f"cannot split {len(ids)} visible GPUs (num_gpus {num_gpus}) "
            f"into {replicas} equal engine replicas"
        )
    size = num_gpus // replicas
    return [",".join(ids[i * size : (i + 1) * size]) for i in range(replicas)]


class _EngineChecks:
    """Per-engine stall and error state, and the checks that read it.

    Each _Replica runs them for its engine. LLMApp inherits them too, so a
    single-engine caller still finds them there, but the health check and
    error count use the replicas' copies, so they are not override points."""

    engine_stall_timeout_s: Optional[float]
    # (token counter total, when it last moved); None until a sample with
    # pending requests, so idle time never counts toward a stall.
    _progress: Optional[Tuple[float, float]] = None
    # Inference 5xx replies in a row, and when the last one came.
    _engine_errors = 0
    _engine_error_at = 0.0

    def _stalled_for(self, stats: Dict[str, float], now: float) -> Optional[float]:
        """Seconds the engine has held requests without its token counters
        moving, once that exceeds ``engine_stall_timeout_s``; else None."""
        tokens = [stats[k] for k in _PROGRESS_COUNTERS if k in stats]
        pending = [stats[k] for k in _PENDING_GAUGES if k in stats]
        if self.engine_stall_timeout_s is None:
            self._progress = None
            return None
        if not tokens or not pending:
            # A failed /metrics read, or an engine without these metrics: no
            # judgment, and the last sample stands.
            return None
        if sum(pending) <= 0:
            self._progress = None
            return None
        total = sum(tokens)
        last = self._progress
        if last is None or total != last[0]:
            self._progress = (total, now)
            return None
        idle = now - last[1]
        return idle if idle > self.engine_stall_timeout_s else None

    def _recent_engine_errors(self, now: float) -> int:
        if now - self._engine_error_at > _ENGINE_ERROR_WINDOW_S:
            return 0
        return self._engine_errors

    def _count_engine_reply(self, status: int) -> None:
        if status < 500:
            self._engine_errors = 0
            return
        now = time.monotonic()
        self._engine_errors = self._recent_engine_errors(now) + 1
        self._engine_error_at = now


class _Replica(_EngineChecks):
    """One engine process: its port, routing load and health state."""

    def __init__(self, app: "LLMApp", index: int, port: int) -> None:
        self.app = app
        self.index = index
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.proc: Optional[subprocess.Popen] = None
        self.inflight = 0
        self.requests = 0
        # Set by each health probe; routing skips a replica that failed one.
        self.healthy = True

    @property
    def engine_stall_timeout_s(self) -> Optional[float]:
        return self.app.engine_stall_timeout_s

    def usable(self) -> bool:
        return self.healthy and (self.proc is None or self.proc.poll() is None)


# Pinned pages cannot be reclaimed, and the engine, /dev/shm and weight staging
# share the container limit, so leave headroom. The agent default is 0.7
# (_presets.AGENT_KV_CPU_OFFLOAD).
_KV_OFFLOAD_MAX_FRACTION = 0.8


def _kill_group(proc: "subprocess.Popen[bytes]") -> None:
    # _start_engine starts the engine as a session leader, so its pid is the
    # group id, still valid for its workers after the leader itself exited.
    if getattr(proc, "_own_group", False):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    elif proc.poll() is None:
        proc.kill()


def _read_limit(path: Path) -> Optional[int]:
    try:
        value = int(path.read_text().strip())
    except (OSError, ValueError):
        # Missing, unreadable, empty or "max": no limit here.
        return None
    # v1 reports "unlimited" as a value near 2^63.
    return value if 0 < value < 1 << 50 else None


def _container_ram_bytes(
    cgroup_root: str = "/sys/fs/cgroup", proc_cgroup: str = "/proc/self/cgroup"
) -> int:
    """The tightest cgroup memory limit on this process or an ancestor, else 0.

    memory.high counts too: past it the kernel throttles and reclaims, and the
    CPU KV tier is pinned memory that cannot be reclaimed. Never /proc/meminfo:
    inside a fal runner it shows the whole host (1.5 TiB on an RTX PRO 6000
    node) while the cgroup files can read empty.
    """
    root = Path(cgroup_root)
    v2_path, v1_path = "/", "/"
    try:
        lines = Path(proc_cgroup).read_text().splitlines()
    except (OSError, ValueError):
        lines = []
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, path = fields
        if hierarchy == "0" and not controllers:
            v2_path = path
        elif "memory" in controllers.split(","):
            v1_path = path
    limits = []
    for base, path, names in (
        (root, v2_path, ("memory.max", "memory.high")),
        (root / "memory", v1_path, ("memory.limit_in_bytes",)),
    ):
        parts = [p for p in path.split("/") if p]
        # Own cgroup and every ancestor: a parent's limit binds as well.
        for depth in range(len(parts), -1, -1):
            directory = base.joinpath(*parts[:depth])
            limits += [_read_limit(directory / name) for name in names]
    return min((x for x in limits if x), default=0)


# RAM fal gives a container per GPU (isolate_controller resources.py:
# GPU-RTXPRO6000 60Gi, GPU-H100-S 60Gi, GPU-H100 112Gi, GPU-H200 112Gi,
# GPU-B200 210Gi), scaled by GPU count. Only used when no cgroup limit is
# readable, so it takes the smallest variant of each GPU. A custom_memory_gb
# or a mixed machine_type list can still give less; kv_cpu_offload_gib covers
# that. Matched by GPU key prefix; MIG slices (16 to 56Gi containers) and
# other GPUs (some at 40 to 48Gi) get the 16 GiB floor.
_FAL_RAM_GIB_PER_GPU = (
    (RTX_PRO_6000, 60),
    (H100, 60),
    ("NVIDIA_H200", 112),
    (B200, 210),
)
_MIN_RAM_GIB_PER_GPU = 16


def _fal_ram_gib_per_gpu(machine: Machine) -> int:
    key = machine.gpu_key or ""
    if "MIG" in (machine.gpu_name or "").upper():
        return _MIN_RAM_GIB_PER_GPU
    return next(
        (gib for prefix, gib in _FAL_RAM_GIB_PER_GPU if key.startswith(prefix)),
        _MIN_RAM_GIB_PER_GPU,
    )


def _kv_offload_config(cpu_bytes: int) -> Dict[str, Any]:
    return {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"cpu_bytes_to_use": cpu_bytes},
    }


class LLMApp(_EngineChecks, App):
    """Serve a vLLM or SGLang engine behind OpenAI-compatible routes.

    Subclass it, set ``model`` (and usually ``machine_type``, ``num_gpus``,
    ``engine_args``), and deploy. The engine runs as a subprocess; the app
    handles readiness, ``/v1/chat/completions``, ``/v1/completions`` and
    ``/v1/models`` (streaming passthrough, engine abort on client disconnect),
    persistent compile caches, a parallel weight pre-read, and engine-metric
    based health. Override ``engine_command`` for full control of the launch.

    A Hugging Face ``model`` is resolved local-first: once a download has
    completed, the cached snapshot is reused without contacting the Hub, so an
    unpinned ``main`` is not refreshed while the volume lives. Pin
    ``model_revision`` (or delete the marker under ``llm_app_snapshots/``
    next to the HF hub cache) to change it. An unpinned app adopts the newest
    complete copy on the volume (also one downloaded by a pinned app), and
    after boot reports whether the Hub has a newer commit (``upstream_newer``
    in the boot report, gauge ``fal_app_llm_model_upstream_newer``). When a
    recipe serves a different checkpoint (below), these markers, downloads and
    checks apply to that checkpoint's repo, so pre-stage it with
    ``fal volumes weights pull <checkpoint>``; the boot report names it as
    ``served_repo``. A pinned ``model_revision`` is never swapped.

    With ``engine_presets`` on, tensor parallelism (from ``num_gpus``),
    ``--disable-custom-all-reduce`` on PCIe-only GPUs and the engine's max
    concurrent sequences (from ``max_multiplexing``: 2x on several GPUs, 1x
    on one) are added to the launch;
    any flag already in ``engine_args`` wins. Set ``max_multiplexing`` to the
    measured cap (256 on one GPU, 128 for multi-GPU MoE) so the router feeds
    the engine enough requests; on several GPUs the engine cap is twice it.
    The presets also apply measured vLLM recipes
    on RTX PRO 6000 and H100 (see ``RECIPES``): sequence caps, FP8 KV cache
    and EAGLE3 for Qwen3-8B, and settings for gpt-oss-120b and DeepSeek V4
    Flash (run it on 2 GPUs: 1.9x the tokens per GPU-hour of 4). A recipe may
    raise the engine's cap above ``max_multiplexing``. A recipe can also
    serve a pre-quantized checkpoint (``checkpoint_for``): the default for
    Qwen3-8B on RTX PRO 6000 is ``Qwen/Qwen3-8B-FP8``. Clients keep using
    ``model`` (or ``served_model_name``). See ``engine_preset_args``.

    OpenAI compatibility: ``/v1/chat/completions``, ``/v1/completions`` and
    ``/v1/responses`` are proxied, streaming included. With vLLM the presets
    add ``--enable-prompt-tokens-details`` (vLLM), so ``usage`` reports prefix-cache
    hits as ``prompt_tokens_details.cached_tokens`` (``input_tokens_details``
    on the Responses API). Use ``/v1/responses`` for models that reason
    between tool calls (gpt-oss) and replay the response ``output`` items,
    reasoning included, in the next ``input``. ``previous_response_id`` is not
    isolated per caller, and works only while one runner serves the app: the
    engine keeps that state in memory by id. Within a runner, continuations
    are routed to the engine replica that created the response.

    For multi-turn conversations on several runners, set ``prefix_hints`` and
    have clients send a stable session id (letters, digits and ``._:-``, at
    most 64 characters; anything else is not advertised) on every turn, as the
    ``X-Fal-Runner-Hint`` header (routing) and in the body as
    ``prompt_cache_key`` or ``user`` (so the runner can advertise it).

    Subclasses overriding ``handle_exit`` or ``teardown`` must call ``super()``:
    they start the shutdown drain and stop the engine.

    Attributes:
        engine (`str`): Engine to launch, ``"vllm"`` or ``"sglang"``.
            Defaults to ``"vllm"``. Any other value needs an ``engine_command``
            override.
        model (`str`): Hugging Face repo id or local model directory to serve.
            Required. Defaults to ``""``.
        model_revision (`str | None`): Hub branch, tag or commit to pin. A
            pinned revision is never swapped for a recipe checkpoint. Defaults
            to ``None`` (unpinned ``main``, reused once downloaded).
        served_model_name (`str | None`): Model name clients send in the
            ``model`` field. Defaults to ``None``, which uses ``model``.
        engine_args (`list[str]`): Extra command-line flags for the engine.
            They win over the presets per flag. Defaults to ``[]``.
        engine_presets (`bool`): Prepend machine-matched engine flags
            (tensor parallelism, all-reduce, sequence cap, measured recipes).
            Defaults to ``True``.
        engine_profile (`str`): Preset profile: ``"throughput"``,
            ``"latency"``, ``"max_throughput"`` or ``"agent"``. Defaults to
            ``"throughput"``. ``"max_throughput"`` trades quality for speed
            and is opt-in. ``"agent"`` targets long tool-calling sessions on
            vLLM: prefix caching, the model family's tool-call parser, a
            context window up to 128k, the CPU KV tier and ``prefix_hints``.
        fp8_weights (`bool | None`): ``None`` uses the measured default
            checkpoint for the model, ``False`` stays on bf16, ``True`` also
            applies online FP8 to models without a measured checkpoint (a
            quality tradeoff). Defaults to ``None``.
        active_params_b (`float | None`): Active parameters in billions, for
            when the estimate from ``config.json`` is wrong. Defaults to
            ``None``.
        engine_env (`dict[str, str]`): Environment variables for the engine
            process. Defaults to ``{}``.
        spec_max_conc (`int | None`): Keep the presets' speculative decoding
            only while ``max_multiplexing`` is at or below this value. ``0``
            never uses it. Defaults to ``None``, which uses the recipe's limit.
        kv_cpu_offload (`float | None`): vLLM only. Fraction (0 to 0.8) of the
            container's RAM used as a CPU tier for evicted KV blocks. ``0``
            turns it off. Defaults to ``None`` (off, except ``0.7`` with
            ``engine_profile = "agent"``).
        kv_cpu_offload_gib (`float | None`): Size of the CPU KV tier in GiB.
            It turns the tier on and takes precedence over ``kv_cpu_offload``.
            Defaults to ``None``.
        engine_port (`int`): Local port of the engine. With replicas, engine
            ``i`` listens on ``engine_port + i``. Defaults to ``8000``.
        engine_replicas (`int`): Engines per runner. ``num_gpus`` is split
            into this many equal slices, one engine each. Defaults to ``1``.
        engine_startup_timeout_s (`float`): Seconds to wait for the engine to
            become ready. Defaults to ``3300.0``.
        engine_max_connections (`int`): Size of the connection pool to the
            engine. It must exceed the engine's max sequences plus queueing.
            Defaults to ``4096``.
        raw_stream_client (`bool`): Use the lightweight raw-socket client for
            streaming requests. Set ``False`` to use httpx. Defaults to
            ``True``.
        persistent_caches (`bool`): Keep engine compile caches on the
            persistent ``/data`` volume. Defaults to ``True``.
        cache_namespace (`str | None`): Overrides the automatic cache
            namespace (engine, version and GPU type). Defaults to ``None``.
        preread_workers (`int`): Threads that pre-read weight shards in
            parallel to warm the page cache. ``0`` disables it. Defaults to
            ``48``.
        report_usage (`bool`): Ask the engine for a final usage chunk on
            streams so ``record_usage`` sees token counts. Clients get one
            extra chunk with empty ``choices``. Defaults to ``True``.
        prefix_hints (`bool | None`): Advertise recent session ids so the
            gateway can route a conversation back to the runner that holds its
            KV cache. Defaults to ``None`` (on with
            ``engine_profile = "agent"``).
        prefix_hint_max (`int`): Number of recent session ids to advertise.
            Defaults to ``64``.
        request_deadline_s (`float | None`): Per-request time limit in
            seconds. Streams end with ``finish_reason`` ``"deadline"``, unary
            calls get a 504, and the engine request is aborted either way.
            Defaults to ``None`` (no limit).
        deadline_margin_s (`float`): Seconds subtracted from the gateway's
            remaining request time when enforcing the deadline. Defaults to
            ``1.0``.
        stream_idle_timeout_s (`float | None`): A stream with no bytes from
            the engine for this long ends with ``finish_reason``
            ``"idle_timeout"`` and the engine request is aborted. It also caps
            the time to first token. Defaults to ``300.0``. ``None`` disables.
        engine_stall_timeout_s (`float | None`): The health check fails when
            the engine holds requests but its token counters do not move for
            this long. vLLM only. Defaults to ``180.0``. ``None`` disables.
        engine_error_threshold (`int | None`): The health check fails after
            this many consecutive inference 5xx replies. Defaults to ``None``
            (disabled).
        shutdown_grace_s (`float`): Seconds that in-flight requests get to
            finish after SIGTERM before they are aborted (streams end with
            ``finish_reason`` ``"shutdown"``, unary calls get a retryable
            503). Capped to fit inside ``termination_grace_period_seconds``.
            Defaults to ``30.0``.
        termination_grace_period_seconds (`int | None`): Seconds between
            SIGTERM and SIGKILL. Raise it to drain longer. Defaults to ``60``.
        cancel_on_gateway_cancel (`bool`): Stop the handler when the gateway
            cancels a request, so the engine aborts it and frees its KV cache.
            Defaults to ``True``.
        keep_alive (`int`): Idle seconds before a runner scales down. Keep it
            at least twice the boot time. Defaults to ``600``.

    Example:
        ```python
        from fal.container import ContainerImage
        from fal.toolkit.llm import LLMApp


        class QwenVLLM(
            LLMApp,
            kind="container",
            image=ContainerImage.from_dockerfile_str(
                "FROM vllm/vllm-openai:v0.32.0"
            ),
            name="qwen-vllm",
        ):
            machine_type = "GPU-RTXPRO6000"
            num_gpus = 1
            max_multiplexing = 256
            model = "Qwen/Qwen2.5-7B-Instruct"
            served_model_name = "qwen2.5-7b-instruct"
            engine_args = ["--max-model-len", "8192"]
        ```
    """

    engine: ClassVar[str] = "vllm"
    model: ClassVar[str] = ""
    # Hub revision (branch, tag or commit). After the first successful download
    # the snapshot is reused without contacting the Hub, so an unpinned `main`
    # is not refreshed while the volume lives; pin a revision to change it.
    model_revision: ClassVar[Optional[str]] = None
    served_model_name: ClassVar[Optional[str]] = None
    engine_args: ClassVar[List[str]] = []
    # Prepend machine-matched engine args (TP, all-reduce, max sequences,
    # measured recipes); engine_args override per flag. The engine cap can
    # exceed max_multiplexing when a recipe raises it, which is harmless.
    engine_presets: ClassVar[bool] = True
    # "throughput" (default), "latency", "max_throughput" or "agent"; recipes
    # can differ by profile. "agent" is the one switch for long tool-calling
    # sessions on vLLM: prefix caching, the model family's tool-call parser, a
    # context window up to 128k, the CPU KV tier (kv_cpu_offload 0.7) and
    # prefix_hints, each overridable by setting it on the class or in
    # engine_args. The engine flags need vLLM and engine_presets; the CPU tier
    # needs vLLM, and prefix_hints works with any engine.
    # max_throughput trades quality for speed and is opt-in only:
    # for Qwen3-8B on RTX PRO 6000 it serves the NVFP4 checkpoint (MMLU-Pro -8.2
    # pts) for about 1.4x the throughput. Models without a max_throughput recipe
    # get the throughput one.
    engine_profile: ClassVar[str] = "throughput"
    # None: the measured default (Qwen3-8B on RTX PRO 6000 serves the
    # Qwen/Qwen3-8B-FP8 checkpoint, quality within run noise, see
    # ai-designs/2026-11-05-rtx-quant-quality.md). False: stay on bf16. True:
    # also online FP8 (--quantization fp8) for models without a measured
    # checkpoint, a quality tradeoff. engine_profile "max_throughput" picks its
    # own (NVFP4) checkpoint regardless of this setting.
    fp8_weights: ClassVar[Optional[bool]] = None
    # Active parameters in billions when the config.json estimate is wrong.
    active_params_b: ClassVar[Optional[float]] = None
    engine_env: ClassVar[Dict[str, str]] = {}
    # Keep the presets' speculative decoding (MTP, EAGLE, DSpark) only while
    # max_multiplexing, the expected concurrency, is at or below this; above it
    # the speculative flags are dropped with a warning (a user
    # --speculative-config in engine_args is kept, with a warning). It wins
    # over the limit a recipe carries (GLM-5.3 on B300: 8); 0 never uses
    # speculation. None: the recipe's limit, else always on.
    spec_max_conc: ClassVar[Optional[int]] = None
    # vLLM only. Fraction (0, 0.8] of the container's RAM to use as a CPU tier for
    # evicted KV blocks (OffloadingConnector). Long prefixes pushed out of GPU
    # memory reload in well under a second instead of being recomputed (gpt-oss
    # 120b on RTX PRO 6000, 113k tokens: 0.7 s vs 11.7 s), with no change in
    # decode throughput. Costs: vLLM turns off the hybrid KV cache manager for
    # connectors, and the pinned pages count against the container's RAM limit.
    # The size is the total across all workers (vLLM's cpu_bytes_to_use), capped
    # at 0.8. None: off. Ignored when engine_args already set --kv-transfer-config.
    # With engine_profile = "agent" (and engine_presets), None means 0.7; 0 turns
    # the tier off. The container's RAM is its cgroup limit; when none is
    # readable (fal runners can show empty cgroup files and the host's
    # /proc/meminfo), it is fal's per-GPU allotment times num_gpus, with a
    # warning. kv_cpu_offload_gib sets the tier size outright and turns it on.
    kv_cpu_offload: ClassVar[Optional[float]] = None
    kv_cpu_offload_gib: ClassVar[Optional[float]] = None
    engine_port: ClassVar[int] = 8000
    # Engines per runner. With R > 1 the runner's num_gpus are split into R
    # equal slices (CUDA_VISIBLE_DEVICES), each engine runs on its own slice
    # with tensor parallel num_gpus / R on port engine_port + i, and each
    # request goes to one engine: the one that served its session id before
    # (X-Fal-Runner-Hint, prompt_cache_key or user), else the one with the
    # fewest requests in flight. On PCIe GPUs (no NVLink) two TP4 engines can
    # beat one TP8 engine when the model fits on four. The health check fails
    # when any engine is down; until the runner is replaced, requests avoid
    # it. The CPU KV tier and max_multiplexing stay per runner: the tier is
    # split across engines, and each engine may take up to max_multiplexing.
    engine_replicas: ClassVar[int] = 1
    engine_startup_timeout_s: ClassVar[float] = 3300.0
    # Upstream connection pool to the engine: one connection per in-flight
    # request, so this must exceed the engine's max sequences plus queueing.
    engine_max_connections: ClassVar[int] = 4096
    # Streaming requests use a small raw-socket client (see _upstream.py): the
    # httpx stack was over half of the proxy's CPU per chunk. Set False to go
    # back to httpx, e.g. for an engine that needs TLS or a proxy.
    raw_stream_client: ClassVar[bool] = True

    # Persistent caches under /data (None disables, a string overrides the
    # auto namespace of engine, version and GPU type).
    persistent_caches: ClassVar[bool] = True
    cache_namespace: ClassVar[Optional[str]] = None
    # Parallel safetensors pre-read; 0 disables.
    preread_workers: ClassVar[int] = DEFAULT_WORKERS
    # Ask the engine for a final usage chunk on streams so token counts are
    # visible to ``record_usage`` (clients see one extra empty-choices chunk).
    report_usage: ClassVar[bool] = True
    # Cache-affinity routing. The client picks a session id and sends it on
    # every turn as ``X-Fal-Runner-Hint: <id>`` (or ``prompt_cache_key``, or
    # ``user``, in the JSON body). The runner then advertises the ids it
    # served recently in ``X-Fal-Runner-Hints`` and the gateway prefers a free
    # runner that lists the id, so a conversation keeps hitting the runner
    # that holds its KV cache. Requests without an id advertise nothing.
    # With engine_profile = "agent", None means on; False turns it off there.
    prefix_hints: ClassVar[Optional[bool]] = None
    # The gateway matches the id literally, so it is advertised as sent. The
    # runner no longer derives hints from the prompt (prefix_hint_chars is
    # gone): without a client id nothing is advertised.
    # Recent ids advertised: size it to max_multiplexing times a few turns.
    # Ids are advertised as sent, so every caller of a runner can read them:
    # use opaque random ids, never user identifiers. The header is about 65
    # bytes per id, so values above 100 risk a proxy's header limit.
    prefix_hint_max: ClassVar[int] = 64
    # Self-enforced per-request deadline in seconds (None: none). Streams end
    # with finish_reason "deadline", unary calls get a 504; either way the
    # engine request is aborted. It is also a cap: a per-request
    # X-Fal-Request-Deadline can only shorten it, and the gateway's remaining
    # time (less the margin) caps both.
    request_deadline_s: ClassVar[Optional[float]] = None
    deadline_margin_s: ClassVar[float] = 1.0
    # A stream that gets no bytes from the engine for this long (headers or
    # body) ends with finish_reason "idle_timeout" and the engine request is
    # aborted, so a wedged engine cannot hold a stream forever. The first wait
    # includes engine queueing and prefill, so this also caps time to first
    # token. None disables.
    stream_idle_timeout_s: ClassVar[Optional[float]] = 300.0
    # The health check fails when the engine has queued or running requests
    # but its prompt and generation token counters have not moved for this
    # long: an engine that answers /health but makes no progress. It is judged
    # across health probes, so it needs two probes at least this far apart.
    # vLLM only (SGLang updates its counters when a request finishes). None
    # disables.
    engine_stall_timeout_s: ClassVar[Optional[float]] = 180.0
    # Opt-in: the health check also fails after this many inference 5xx
    # replies in a row (no inference reply below 500 in between, and none of
    # them more than a minute after the previous one): an engine that answers
    # /health but fails every request. A caller retrying an input the engine
    # always fails on can reach it too. None (default) disables.
    engine_error_threshold: ClassVar[Optional[int]] = None

    # After SIGTERM, in-flight requests get this long to finish before they are
    # aborted (streams end with finish_reason "shutdown", unary calls get a
    # retryable 503). It is capped so the engine stop and a little slack still
    # fit inside the termination window, so subclasses raise
    # ``termination_grace_period_seconds`` to drain longer. Only that class
    # attribute is visible at runtime: a value set in pyproject or the CLI
    # overrides it on the platform but not in the app, so the budget then
    # assumes the class value.
    shutdown_grace_s: ClassVar[float] = 30.0
    # Seconds between SIGTERM and SIGKILL. The platform default (5s) leaves no
    # time to drain streams; 60 fits the 30s drain plus the engine stop.
    termination_grace_period_seconds: ClassVar[Optional[int]] = 60
    # A gateway cancel (queue cancel, request timeout) stops the handler, which
    # closes the engine connection so the engine aborts the request and frees
    # its KV cache. Disconnects are already handled per request below.
    cancel_on_gateway_cancel: ClassVar[bool] = True

    # Idle seconds before a runner scales down. LLM boots take minutes, so a
    # short keep_alive lets cold starts dominate the tail (simulated with a 143s
    # boot: p99 queue wait 111s at keep_alive 30, 35s at 120). 600 kept p99 under 1s at
    # high volume and is nearly insensitive to boot time; the cost is idle GPU
    # time at low volume. Rule of thumb: keep_alive >= 2x boot time.
    keep_alive = 600
    persist_cuda_cache = True

    # setup() adds httpx's.
    _transport_errors: Tuple[Type[BaseException], ...] = (OSError,)
    # Transport errors that mean the proxy ran out of engine connections.
    _busy_errors: Tuple[Type[BaseException], ...] = (UpstreamBusy,)
    _draining = False
    _drain_started: Optional[float] = None
    # A recipe checkpoint that failed to download (see _resolve_model).
    _skip_checkpoint: Optional[str] = None
    _abort_fired = False
    _serving_loop: Optional[asyncio.AbstractEventLoop] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Check only the class that sets these, so subclasses do not repeat it.
        sets_config = bool(
            {
                "engine_args",
                "num_gpus",
                "max_multiplexing",
                "spec_max_conc",
                "engine_replicas",
            }
            & set(vars(cls))
        )
        user_multiplexing = cls.max_multiplexing
        super().__init_subclass__(**kwargs)
        replicas = cls.engine_replicas
        if not isinstance(replicas, int) or replicas < 1:
            raise ValueError(
                f"{cls.__name__}: engine_replicas must be a positive integer, "
                f"got {replicas!r}"
            )
        if (cls.num_gpus or 1) % replicas:
            raise ValueError(
                f"{cls.__name__}: num_gpus {cls.num_gpus or 1} does not split "
                f"into {replicas} equal engine replicas"
            )
        if cls.engine_profile not in PROFILES:
            raise ValueError(
                f"{cls.__name__}: engine_profile must be one of {PROFILES}, "
                f"got {cls.engine_profile!r}"
            )
        if cls.kv_cpu_offload is not None and not (
            0 <= cls.kv_cpu_offload <= _KV_OFFLOAD_MAX_FRACTION
        ):
            raise ValueError(
                f"{cls.__name__}: kv_cpu_offload must be a fraction in "
                f"[0, {_KV_OFFLOAD_MAX_FRACTION}], got {cls.kv_cpu_offload!r}"
            )
        if cls.kv_cpu_offload_gib is not None and not (
            0 < cls.kv_cpu_offload_gib < math.inf
        ):
            raise ValueError(
                f"{cls.__name__}: kv_cpu_offload_gib must be positive, "
                f"got {cls.kv_cpu_offload_gib!r}"
            )
        if not sets_config:
            return
        # warnings (not print): this runs on the deploying machine and callers
        # can filter or assert on it.
        for message in engine_arg_warnings(
            cls.engine_args,
            (cls.num_gpus or 1) // replicas,
            user_multiplexing,
            cls.spec_max_conc,
        ):
            warnings.warn(f"{cls.__name__}: {message}", stacklevel=2)

    def engine_command(self, model_path: str) -> List[str]:
        """Build the command that launches the engine.

        Override it to run any server that speaks the OpenAI API on
        ``engine_port``. The default builds a ``vllm serve`` or
        ``sglang.launch_server`` command from ``engine``, the presets and
        ``engine_args``.

        Args:
            model_path: Local path of the resolved model weights.

        Returns:
            The engine command as an argument list.

        Raises:
            ValueError: If ``engine`` is neither ``"vllm"`` nor ``"sglang"``
                and this method is not overridden.
        """
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
        engine_args = list(self.engine_args)
        user_args = list(engine_args)
        if self.engine_presets:
            model = model_facts(
                self.model, load_model_config(model_path), self.active_params_b
            )
            engine_args = merge_engine_args(
                engine_preset_args(
                    self.engine,
                    self._machine(),
                    model,
                    profile=self.engine_profile,
                    fp8_weights=self.fp8_weights is True,
                    max_multiplexing=self.max_multiplexing,
                    user_args=engine_args,
                    skip_checkpoint=self._skip_checkpoint,
                    spec_max_conc=self.spec_max_conc,
                ),
                engine_args,
            )
        offload = (
            self._kv_offload_bytes(user_args)
            if self.engine == "vllm"
            and not any(
                a.replace("_", "-").split("=")[0] == "--kv-transfer-config"
                for a in engine_args
            )
            else None
        )
        if offload:
            # The tier is sized for the runner; replicas share its RAM.
            engine_args += [
                "--kv-transfer-config",
                json.dumps(_kv_offload_config(offload // self.engine_replicas)),
            ]
        return [
            *head,
            "--served-model-name",
            name,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.engine_port),
            *engine_args,
        ]

    def record_usage(self, path: str, usage: Dict[str, Any]) -> None:
        """Receive the token usage of a request once the engine reports it.

        Override it to log or bill usage. Call ``super()`` to keep the built-in
        totals. A stream reports usage only if the engine sends a usage chunk:
        ``report_usage`` asks for one unless the client set its own
        ``stream_options``, and ``/v1/responses`` always includes it.

        Args:
            path: Request path, for example ``/v1/chat/completions``.
            usage: The ``usage`` object from the engine's response, with
                ``prompt_tokens`` and ``completion_tokens`` (``input_tokens``
                and ``output_tokens`` on the Responses API).
        """
        # The Responses API names these input_tokens and output_tokens.
        counts = {
            "prompt_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
            "completion_tokens": usage.get("completion_tokens")
            or usage.get("output_tokens"),
            "cached_tokens": _cached_tokens(usage),
        }
        for key, value in counts.items():
            self._usage_totals[key] = self._usage_totals.get(key, 0) + int(value or 0)
        self._usage_totals["requests"] = self._usage_totals.get("requests", 0) + 1

    def _kv_offload_fraction(self) -> Optional[float]:
        if self.kv_cpu_offload is None:
            agent = self.engine_profile == AGENT_PROFILE and self.engine_presets
            return AGENT_KV_CPU_OFFLOAD if agent else None
        return self.kv_cpu_offload or None

    def _kv_offload_bytes(self, engine_args: List[str]) -> Optional[int]:
        """Size of the CPU KV tier in bytes, or None for no tier."""
        if self.kv_cpu_offload == 0:
            return None
        if self.kv_cpu_offload_gib is not None:
            size = int(self.kv_cpu_offload_gib * (1 << 30))
            limit = int(_container_ram_bytes() * _KV_OFFLOAD_MAX_FRACTION)
            if limit and size > limit:
                self._warn_once(
                    f"kv_cpu_offload_gib={self.kv_cpu_offload_gib:g} exceeds "
                    f"{_KV_OFFLOAD_MAX_FRACTION} of the container's memory "
                    f"limit; capped at {limit / (1 << 30):.1f} GiB."
                )
                return limit
            return size
        fraction = self._kv_offload_fraction()
        if not fraction:
            return None
        if self.kv_cpu_offload is None and "--no-enable-prefix-caching" in engine_args:
            # The implicit agent default needs prefix caching to be useful.
            return None
        ram = _container_ram_bytes()
        if not ram:
            per_gpu = _fal_ram_gib_per_gpu(self._machine())
            num_gpus = self.num_gpus or 1
            ram = (per_gpu << 30) * num_gpus
            self._warn_once(
                f"no container memory limit is readable; sizing the CPU KV tier "
                f"from an assumed {per_gpu} GiB per GPU x {num_gpus} "
                "(fal's allotment for this GPU). Set kv_cpu_offload_gib to size "
                "it explicitly."
            )
        return int(ram * fraction)

    def _warn_once(self, message: str) -> None:
        # engine_command runs more than once per boot (config key, start).
        warned = self.__dict__.setdefault("_offload_warnings", set())
        if message not in warned:
            warned.add(message)
            print(f"[llm-app] WARNING: {message}")

    def _prefix_hints_on(self) -> bool:
        if self.prefix_hints is None:
            return self.engine_profile == AGENT_PROFILE
        return self.prefix_hints

    def provide_hints(self) -> List[str]:
        if not self._prefix_hints_on():
            raise NotImplementedError
        return list(getattr(self, "_recent_prefixes", ()))

    def _served_name(self) -> str:
        return self.served_model_name or self.model

    def setup(self) -> None:  # type: ignore[override]
        # Async setup is not needed: blocking steps run before the event loop
        # serves requests, and the readiness wait is a plain polling loop.
        import httpx  # noqa: PLC0415

        # Engine transport failures: httpx's, and the raw client's (OSError).
        self._transport_errors = (httpx.HTTPError, OSError)
        self._busy_errors = (httpx.PoolTimeout, UpstreamBusy)
        if not self.model:
            raise ValueError("LLMApp subclasses must set `model`")
        t0 = time.monotonic()
        self._setup_start = t0
        self._boot: Dict[str, Any] = {}
        self._usage_totals: Dict[str, int] = {}
        self._inflight: Set[asyncio.Event] = set()
        self._replicas = [
            _Replica(self, i, self.engine_port + i) for i in range(self.engine_replicas)
        ]
        # Session id digest -> replica index, least recently used first.
        self._sessions: OrderedDict[bytes, int] = OrderedDict()
        # Responses API id -> index of the replica that created it.
        self._response_routes: OrderedDict[str, int] = OrderedDict()
        # The replica the current request was routed to: set in _proxy, read
        # where the engine call is made (tasks copy the context). Made here,
        # not at module level: a ContextVar cannot be pickled with the class.
        self._route_var: contextvars.ContextVar[Optional[_Replica]] = (
            contextvars.ContextVar("llm_app_route", default=None)
        )
        self._recent_prefixes: Deque[str] = deque(
            maxlen=min(100, max(1, self.prefix_hint_max))
        )
        self._engine_log_tail: Deque[str] = deque(maxlen=50)

        self._deadline = t0 + self._startup_budget_s()
        self._boot_marker: Optional[Path] = None
        self._preread_stop = threading.Event()
        self._stage = "init"
        watchdog = threading.Timer(
            max(0.0, self._deadline - time.monotonic()), self._on_budget_timer
        )
        watchdog.daemon = True
        self._watchdog = watchdog
        watchdog.start()
        try:
            self._setup_stages(httpx, t0)
        except BaseException:
            # Any setup failure, not only budget or exit, must not leave the
            # engine running; a stop error must not replace the real one.
            try:
                self._stop_engine(_FAIL_STOP_TIMEOUT_S)
            except Exception as exc:
                print(f"[llm-app] could not stop {self.engine}: {exc!r}")
            raise
        finally:
            watchdog.cancel()

    def _on_budget_timer(self) -> None:
        # Blocking stages (download) cannot be interrupted, so make
        # the overrun visible before the platform kills the runner. The
        # readiness wait checks the budget itself and reports through _fail.
        if self._stage == "engine startup":
            return
        try:
            report = dict(self._boot)
        except RuntimeError:  # resized by the setup thread mid-copy
            report = {}
        print(
            f"[llm-app] startup budget exceeded during {self._stage}; "
            f"boot report: {report}"
        )

    def _setup_stages(self, httpx: Any, t0: float) -> None:
        namespace = None
        if self.persistent_caches:
            namespace = self.cache_namespace or "-".join(
                (
                    self.engine,
                    _engine_version(self.engine),
                    _torch_tag(),
                    self._gpu_type(),
                )
            )
            apply_engine_cache_env(namespace)

        self._stage = "model resolve"
        model_path = self._resolve_model()
        self._boot["resolve_s"] = time.monotonic() - t0
        self._check_budget("model resolve")
        if namespace is not None:
            # Compiled kernels depend on model shapes, TP size and engine args,
            # so warm means a previous boot of this exact config finished. Keyed
            # after resolve: presets read the model's config.json.
            self._boot_marker = namespace_dir(namespace) / (
                f"boot-{self._config_key(model_path)}.json"
            )
            previous = read_marker(self._boot_marker)
            self._boot["compile_cache_warm"] = previous is not None
            if previous is not None:
                self._boot["previous_boot"] = previous

        self._stage = "engine startup"
        self._engine_started = time.monotonic()
        # The first replica's process, kept as the single-engine handle.
        self._proc = self._start_engine(model_path)
        # The engine spends its first seconds importing and initializing CUDA
        # before it reads weights, so the pre-read runs alongside it.
        if self.preread_workers > 0 and os.path.isdir(model_path):
            threading.Thread(
                target=self._preread, args=(model_path,), daemon=True
            ).start()
        # Engine calls use their replica's absolute URL; the base (the first
        # replica) only serves relative paths, as a subclass might send.
        base_url = f"http://127.0.0.1:{self.engine_port}"
        self._wait_ready()
        if self.engine_replicas > 1:
            self._boot["engine_replicas"] = self.engine_replicas
        self._boot["engine_ready_s"] = time.monotonic() - self._engine_started
        self._boot["total_setup_s"] = time.monotonic() - t0
        self._write_boot_marker()
        pool = _fit_pool_to_fd_limit(self.engine_max_connections)
        self._stream_slots = asyncio.Semaphore(pool)
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(connect=30.0, read=3600.0, write=300.0, pool=60.0),
            limits=httpx.Limits(
                max_connections=pool,
                max_keepalive_connections=min(pool, 512),
            ),
        )
        # Copy: the log pump thread may still add late milestones.
        print(f"[llm-app] boot report: {dict(self._boot)}")
        self._start_upstream_check()

    # Runners do not have `fal` installed: it is pickled by value, so only
    # module-level imports resolve. Keep fal imports out of method bodies.
    @staticmethod
    def _gpu_type() -> str:
        return get_gpu_type()

    @staticmethod
    def _gpu_name() -> Optional[Tuple[str, Optional[int]]]:
        return detect_gpu()

    def _machine(self) -> Machine:
        # One nvidia-smi call per boot: the short type is derived from the name.
        machine = getattr(self, "_machine_cache", None)
        if machine is None:
            gpu = self._gpu_name()
            machine = Machine(
                gpu_type=short_gpu_type(gpu[0]) if gpu else self._gpu_type(),
                gpu_name=gpu[0] if gpu else None,
                gpu_memory_mib=gpu[1] if gpu else None,
                # What one engine sees: presets size TP and recipes from it.
                num_gpus=(self.num_gpus or 1) // self.engine_replicas,
            )
            self._machine_cache = machine
        return machine

    def _model_source(self) -> str:
        """What to download and serve: a recipe's checkpoint, else ``model``."""
        # A local model directory is served as given.
        if not self.engine_presets or os.path.isdir(self.model):
            return self.model
        source = checkpoint_for(
            self.engine,
            self._machine(),
            self.model,
            profile=self.engine_profile,
            fp8_weights=self.fp8_weights,
        )
        # Hub ids are case-insensitive, as recipe matching is.
        if not source or _same_repo(source, self.model):
            return self.model
        if self.model_revision is not None:
            # A pinned model_revision names exact weights, so it is never swapped.
            print(
                f"[llm-app] model_revision is pinned: serving {self.model}, "
                f"not the recipe checkpoint {source}"
            )
            return self.model
        self._boot["model_source"] = source
        print(
            f"[llm-app] serving checkpoint {source} as {self._served_name()} "
            f"(pre-stage it with `fal volumes weights pull {source}`)"
        )
        return source

    def _resolve_model(self) -> str:
        self._skip_checkpoint = None
        source = self._model_source()
        if source == self.model:
            return self._resolve_repo(self.model)
        try:
            return self._resolve_repo(source)
        except Exception as exc:
            # A recipe's checkpoint must never stop an app that would boot on
            # its own model (offline hub, missing repo).
            print(
                f"[llm-app] checkpoint {source} unavailable ({exc!r}), "
                f"serving {self.model}"
            )
            self._boot.pop("model_source", None)
            # Its recipe's flags were tuned for those weights, not these.
            self._skip_checkpoint = source
            return self._resolve_repo(self.model)

    def _resolve_repo(self, repo: str) -> str:
        """Resolve ``repo`` local-first; its markers and upstream check follow it."""
        self._model_repo = repo
        self._served_commit = None
        self._boot["served_repo"] = repo
        if os.path.isdir(repo):
            self._boot["resolve_source"] = "path"
            return repo
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415
        except ImportError:
            self._boot["resolve_source"] = "path"
            return repo
        # model_revision pins the app's own model; a recipe checkpoint is only
        # chosen when unpinned (see _model_source).
        revision = self.model_revision if _same_repo(repo, self.model) else None
        if revision is None:
            print(
                f"[llm-app] model_revision is not set: {repo} follows the "
                "default branch. Pin model_revision for reproducible deploys."
            )
        if not self.persistent_caches:
            # The HF cache is not persistent, so a marker would mislead.
            return self._download(snapshot_download, revision, repo)
        marker = snapshot_marker_path(repo, revision)
        recorded = read_marker(marker)
        recorded = recorded or {}
        path = recorded.get("path")
        if recorded.get("model") == repo and isinstance(path, str):
            # Reuse the exact snapshot (a per-commit dir) a previous download
            # completed, without contacting the Hub. Re-resolving the revision
            # offline could land on another runner's partial download of a
            # newer commit. A pruned cache leaves dangling symlinks, so require
            # a readable config.
            if snapshot_intact(path):
                self._boot["resolve_source"] = "local"
                commit = recorded.get("commit")
                self._served_commit = commit if isinstance(commit, str) else None
                if self._served_commit is None:
                    self._served_commit = self._dir_commit(path)
                return path
            print(f"[llm-app] cached snapshot {path} incomplete, downloading")
        path = self._download(snapshot_download, revision, repo)
        # Only the default branch head may enter the unpinned marker: a pinned
        # branch or PR ref with a newer date must not change what unpinned
        # apps serve. One Hub call tells both whether this commit is main's
        # head and its date.
        main = _hub_commit(repo, None, _HUB_TIMEOUT_S)
        # The snapshot dir name is authoritative for what was downloaded.
        commit = self._dir_commit(path)
        if commit is None and revision is None and main is not None:
            commit = main[0]
        is_main_head = main is not None and main[0] == commit
        date = main[1] if is_main_head else None
        self._served_commit = commit
        try:
            if revision is not None:
                write_marker(
                    marker,
                    {
                        "model": repo,
                        "revision": revision,
                        "path": path,
                        "commit": commit,
                        "commit_date": date,
                    },
                )
            # An unpinned download is main by definition, even if the lookup
            # failed; a pinned one only when it is main's current head.
            if commit is not None and (revision is None or is_main_head):
                advance_snapshot_marker(repo, path, commit, date)
            elif revision is None:
                # Commit unknown (not a per-commit dir, Hub unreachable):
                # still record the completed download for reuse.
                write_marker(
                    marker,
                    {"model": repo, "revision": None, "path": path},
                )
        except OSError as exc:
            print(f"[llm-app] could not write snapshot marker: {exc!r}")
        return path

    @staticmethod
    def _dir_commit(path: str) -> Optional[str]:
        name = os.path.basename(os.path.normpath(path))
        return name if _COMMIT_RE.fullmatch(name) else None

    def _start_upstream_check(self) -> None:
        # Only unpinned Hub models served from a known commit can fall behind.
        if (
            self.model_revision is not None
            or self._boot.get("resolve_source") == "path"
            or not getattr(self, "_served_commit", None)
        ):
            return
        threading.Thread(target=self._check_upstream, daemon=True).start()

    def _check_upstream(self) -> None:
        # Report only: never changes the weights served or the markers.
        repo = getattr(self, "_model_repo", None) or self.model
        info = _hub_commit(repo, None, _HUB_TIMEOUT_S)
        if info is None:
            print("[llm-app] upstream model check failed, skipped")
            return
        # A sha comparison: "newer" means main differs from the served commit
        # (usually ahead; a force-push also counts).
        newer = info[0] != self._served_commit
        print(
            f"[llm-app] upstream check: serving {self._served_commit}, "
            f"main is {info[0]}, newer={newer}"
        )
        if newer:
            print(
                f"[llm-app] WARNING: {repo} main differs from the served "
                "commit. New runners pick it up once a complete copy is on the "
                "volume (fal volumes weights pull, or pin model_revision)."
            )
        self._boot["model_commit"] = self._served_commit
        self._boot["upstream_commit"] = info[0]
        self._boot["upstream_newer"] = newer
        _set_upstream_newer_gauge(1.0 if newer else 0.0)

    def _download(
        self, snapshot_download: Any, revision: Optional[str], repo: str
    ) -> str:
        path = snapshot_download(repo, revision=revision)
        self._boot["resolve_source"] = "network"
        self._served_commit = self._dir_commit(path)
        return path

    def _config_key(self, model_path: Optional[str] = None) -> str:
        # The command is built from the real model path (presets read its
        # config.json), then the path is replaced by the repo it came from:
        # snapshot dirs change per revision, while a recipe checkpoint (FP8)
        # and the model itself (bf16) compile different kernels under the same
        # flags. The port and the KV connector (its CPU tier size follows the
        # host's RAM) say nothing about the compiled kernels.
        repo = getattr(self, "_model_repo", None) or self.model
        try:
            cmd = self.engine_command(model_path or self.model)
            if model_path:
                cmd = [repo if arg == model_path else arg for arg in cmd]
        except Exception as exc:
            print(
                f"[llm-app] engine_command failed ({exc!r}), "
                "config key falls back to engine_args"
            )
            cmd = list(self.engine_args)
        stripped: List[str] = []
        skip = False
        for arg in cmd:
            if skip:
                skip = False
            elif arg in ("--port", "--kv-transfer-config"):
                skip = True
            elif not arg.startswith(("--port=", "--kv-transfer-config=")):
                stripped.append(arg)
        env = sorted(self.engine_env.items())
        blob = json.dumps(
            [repo, self.engine, stripped, env, self.num_gpus], default=str
        )
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def _write_boot_marker(self) -> None:
        if self._boot_marker is None:
            return
        try:
            write_marker(
                self._boot_marker,
                {
                    "total_setup_s": self._boot["total_setup_s"],
                    "engine_ready_s": self._boot["engine_ready_s"],
                },
            )
        except OSError as exc:
            print(f"[llm-app] could not write boot marker: {exc!r}")

    def _startup_budget_s(self) -> float:
        # The platform startup timeout only applies on a runner.
        if not os.getenv("IS_ISOLATE_AGENT"):
            return self.engine_startup_timeout_s
        platform = (
            self.startup_timeout
            if self.startup_timeout is not None
            else DEFAULT_PLATFORM_STARTUP_TIMEOUT
        )
        margin = min(_STARTUP_MARGIN_S, platform * _STARTUP_MARGIN_FRACTION)
        return min(self.engine_startup_timeout_s, platform - margin)

    def _check_budget(self, stage: str) -> None:
        if time.monotonic() > self._deadline:
            self._fail(f"startup budget exceeded after {stage}")

    def _fail(self, reason: str) -> None:
        self._boot["failed_after_s"] = time.monotonic() - self._setup_start
        report = dict(self._boot)
        print(f"[llm-app] boot report: {report}")
        try:
            self._stop_engine(_FAIL_STOP_TIMEOUT_S)
        except Exception as exc:
            print(f"[llm-app] engine stop failed: {exc!r}")
        tail = "\n".join(list(self._engine_log_tail)[-10:])
        raise RuntimeError(f"{reason}\nboot report: {report}\n{tail}")

    def _preread(self, model_path: str) -> None:
        # Only warms the page cache; the engine reads the weights regardless.
        try:
            stats = preread_weights(
                model_path, workers=self.preread_workers, stop=self._preread_stop
            )
        except OSError as exc:
            print(f"[llm-app] pre-read failed: {exc}")
            return
        if self._preread_stop.is_set():
            print("[llm-app] pre-read stopped with the engine")
            return
        self._boot["preread_s"] = stats.seconds
        print(
            f"[llm-app] pre-read {stats.bytes / 1e9:.1f} GB in "
            f"{stats.seconds:.1f}s ({stats.gb_per_s:.2f} GB/s)"
        )

    def _preset_env(self, model_path: str) -> Dict[str, str]:
        if not self.engine_presets or self.engine not in ("vllm", "sglang"):
            return {}
        model = model_facts(
            self.model, load_model_config(model_path), self.active_params_b
        )
        return engine_preset_env(
            self.engine,
            self._machine(),
            model,
            profile=self.engine_profile,
            user_args=self.engine_args,
            skip_checkpoint=self._skip_checkpoint,
            max_multiplexing=self.max_multiplexing,
        )

    def _start_engine(self, model_path: str) -> subprocess.Popen:
        """Start every engine replica; returns the first one's process."""
        cmd = self.engine_command(model_path)
        env = {**self._preset_env(model_path), **os.environ, **self.engine_env}
        env.setdefault("VLLM_NO_USAGE_STATS", "1")
        # vLLM's own 600s engine-ready limit is too short for a cold data
        # parallel boot (kernel JIT and warmup); the startup budget governs.
        remaining = self._deadline - time.monotonic()
        env.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", str(max(int(remaining), 600)))
        # setup() made the replicas; a direct call (no setup) gets one.
        replicas = getattr(self, "_replicas", None) or [
            _Replica(self, 0, self.engine_port)
        ]
        if len(replicas) == 1:
            commands = [(cmd, env)]
        else:
            devices = _split_devices(
                env.get("CUDA_VISIBLE_DEVICES"), self.num_gpus or 1, len(replicas)
            )
            commands = [
                (
                    _with_port(cmd, r.port),
                    {**env, "CUDA_VISIBLE_DEVICES": devices[r.index]},
                )
                for r in replicas
            ]
        for replica, (replica_cmd, replica_env) in zip(replicas, commands):
            if len(replicas) == 1:
                print(f"[llm-app] starting: {' '.join(replica_cmd)}")
            else:
                print(
                    f"[llm-app] starting replica {replica.index} on GPUs "
                    f"{replica_env['CUDA_VISIBLE_DEVICES']}: {' '.join(replica_cmd)}"
                )
            # Recorded before the next launch, so a failed one still stops it.
            # Own session: the engine's workers share its process group, so
            # _stop_engine can kill them along with it.
            replica.proc = subprocess.Popen(
                replica_cmd,
                env=replica_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            replica.proc._own_group = True  # type: ignore[attr-defined]
            threading.Thread(
                target=self._pump_engine_logs,
                args=(replica.proc, replica.index),
                daemon=True,
            ).start()
        return replicas[0].proc  # type: ignore[return-value]

    def _engine_label(self, replica: _Replica) -> str:
        if self.engine_replicas == 1:
            return self.engine
        return f"{self.engine} replica {replica.index}"

    def _pump_engine_logs(self, proc: subprocess.Popen, replica: int = 0) -> None:
        assert proc.stdout is not None
        # Boot milestones come from the first replica; the others boot alike.
        milestones = _BOOT_MILESTONES.get(self.engine, ()) if replica == 0 else ()
        tag = self.engine if self.engine_replicas == 1 else f"{self.engine}:{replica}"
        last_progress = float("-inf")

        def handle(raw: bytes, progress: bool) -> None:
            nonlocal last_progress
            line = raw[:_MAX_LINE_BYTES].decode("utf-8", errors="replace").rstrip()
            if not line:
                return
            # Milestone times are seconds since setup start.
            for needle, phase in milestones:
                key = f"{phase}_s"
                if needle in line and (key not in self._boot or key.endswith("_end_s")):
                    self._boot[key] = time.monotonic() - self._setup_start
            if progress:
                now = time.monotonic()
                if now - last_progress < _PROGRESS_LOG_INTERVAL_S:
                    return
                last_progress = now
            # Tagged when several engines share the tail a startup failure quotes.
            self._engine_log_tail.append(
                line if self.engine_replicas == 1 else f"[{tag}] {line}"
            )
            print(f"[{tag}] {line}")

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

    def _ready_path(self) -> str:
        # SGLang's /health answers 200 as soon as its HTTP server is up, before
        # warmup finishes; /health_generate runs a one-token generation and
        # fails until the engine can serve. vLLM's /health only passes when
        # the engine is up.
        return "/health_generate" if self.engine == "sglang" else "/health"

    def _wait_ready(self) -> None:
        """Wait until every replica can serve; any exit fails setup."""
        import httpx  # noqa: PLC0415

        waiting = list(self._replicas)
        while time.monotonic() < self._deadline:
            for replica in self._replicas:
                code = replica.proc.poll()  # type: ignore[union-attr]
                if code is not None:
                    self._fail(f"{self._engine_label(replica)} exited with code {code}")
            for replica in list(waiting):
                try:
                    # /health_generate runs a generation, so it gets longer than
                    # /health, capped so one probe cannot overrun the budget.
                    # The runtime health probe stays on the cheap /health.
                    cap = 30.0 if self.engine == "sglang" else 5.0
                    left = self._deadline - time.monotonic()
                    resp = httpx.get(
                        f"{replica.url}{self._ready_path()}",
                        timeout=max(1.0, min(cap, left)),
                    )
                except httpx.HTTPError:
                    continue
                if resp.status_code == 200:
                    waiting.remove(replica)
            if not waiting:
                return
            time.sleep(1.0)
        self._fail(
            f"{self._engine_label(waiting[0])} not ready within the startup budget"
        )

    def _termination_window(self) -> float:
        return float(
            self.termination_grace_period_seconds or _DEFAULT_TERMINATION_GRACE_S
        )

    def _drain_budget(self) -> float:
        return max(
            0.0,
            min(
                self.shutdown_grace_s,
                self._termination_window() - _ENGINE_STOP_BUDGET_S - _DRAIN_SLACK_S,
            ),
        )

    def _engine_stop_wait(self) -> float:
        # Without a drain start there is no deadline to respect.
        if self._drain_started is None:
            return _MAX_ENGINE_STOP_WAIT_S
        remaining = self._termination_window() - (
            time.monotonic() - self._drain_started
        )
        return min(_MAX_ENGINE_STOP_WAIT_S, max(_ENGINE_STOP_BUDGET_S, remaining))

    def handle_exit(self) -> None:
        # Runs inside a signal handler on the event loop thread, so the timer
        # is armed through call_soon_threadsafe instead of touching the loop's
        # timer heap directly.
        if self._draining:
            return
        self._draining = True
        self._drain_started = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = self._serving_loop
        try:
            if loop is None:
                raise RuntimeError("no event loop")
            loop.call_soon_threadsafe(
                lambda: loop.call_later(self._drain_budget(), self._abort_inflight)
            )
        except RuntimeError as exc:
            print(f"[llm-app] handle_exit: {exc}, in-flight requests not aborted")

    def _abort_inflight(self) -> None:
        self._abort_fired = True
        for abort in list(getattr(self, "_inflight", ())):
            abort.set()

    def _count_shutdown_abort(self) -> None:
        self._usage_totals["shutdown_aborts"] = (
            self._usage_totals.get("shutdown_aborts", 0) + 1
        )

    def teardown(self) -> None:
        self._stop_engine(self._engine_stop_wait())

    def _stop_engine(self, timeout: float) -> None:
        # A running pre-read would otherwise hold interpreter exit.
        stop = getattr(self, "_preread_stop", None)
        if stop is not None:
            stop.set()
        # Every replica is signalled first, so their waits overlap.
        procs = [
            r.proc
            for r in getattr(self, "_replicas", ())
            if r.proc is not None and r.proc.poll() is None
        ]
        # The parent gets SIGTERM alone so it can shut its workers down; the
        # group is SIGKILLed after, which reaps workers it left behind.
        for proc in procs:
            proc.terminate()
        end = time.monotonic() + timeout
        stuck = []
        for proc in procs:
            try:
                proc.wait(timeout=max(0.0, end - time.monotonic()))
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                stuck.append(proc)
            else:
                _kill_group(proc)
        # A parent that already exited may have left workers in its group.
        for r in getattr(self, "_replicas", ()):
            if r.proc is not None and r.proc not in procs:
                _kill_group(r.proc)
        for proc in stuck:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"[llm-app] {self.engine} still running after kill")

    async def _replica_stats(
        self, replica: _Replica, timeout: float = 5.0
    ) -> Dict[str, float]:
        resp = await self._client.get(f"{replica.url}/metrics", timeout=timeout)
        return _parse_metrics(resp.text)

    async def _engine_stats(self, timeout: float = 5.0) -> List[Dict[str, float]]:
        """Each replica's metrics; an empty dict for one that did not answer."""

        async def read(replica: _Replica) -> Dict[str, float]:
            try:
                return await self._replica_stats(replica, timeout)
            except Exception:
                return {}

        return list(await asyncio.gather(*(read(r) for r in self._replicas)))

    async def _check_replica(
        self, replica: _Replica
    ) -> Tuple[Dict[str, float], Optional[str]]:
        """(metrics, why the replica fails the health check or None)."""
        label = self._engine_label(replica)
        code = replica.proc.poll()  # type: ignore[union-attr]
        if code is not None:
            return {}, f"{label} exited ({code})"
        try:
            resp = await self._client.get(
                f"{replica.url}/health", timeout=_ENGINE_HEALTH_TIMEOUT_S
            )
        except Exception as exc:
            return {}, f"{label} unreachable: {exc}"
        if resp.status_code != 200:
            return {}, f"{label} unhealthy ({resp.status_code})"
        try:
            stats = await self._replica_stats(replica, _ENGINE_METRICS_TIMEOUT_S)
        except Exception:
            # /health answered, so a slow /metrics alone is not a failure.
            stats = {}
        errors = replica._recent_engine_errors(time.monotonic())
        if self.engine_error_threshold is not None and (
            errors >= self.engine_error_threshold
        ):
            return stats, f"{label} failed {errors} requests in a row"
        stalled = replica._stalled_for(stats, time.monotonic())
        if stalled is not None:
            return stats, f"{label} made no progress for {stalled:.0f}s"
        return stats, None

    async def health(self) -> Dict[str, Any]:  # type: ignore[override]
        """Report engine liveness, never request outcomes.

        The check passes when every engine process is up, its own ``/health``
        answers 200 and, when it holds requests, its token counters keep
        moving (see ``engine_stall_timeout_s``). With replicas, one failing
        engine fails the check (the runner is replaced); until then, routing
        avoids it. Overrides must call ``super()``.

        Returns:
            A dict with ``status``, ``version`` and engine stats.

        Raises:
            HTTPException: With status 503 while the runner shuts down or when
                an engine is unhealthy.
        """
        self._serving_loop = asyncio.get_running_loop()
        if self._draining:
            raise HTTPException(503, "runner is shutting down")
        checks = await asyncio.gather(*(self._check_replica(r) for r in self._replicas))
        for replica, (_, failure) in zip(self._replicas, checks):
            replica.healthy = failure is None
        failure = next((f for _, f in checks if f is not None), None)
        if failure is not None:
            raise HTTPException(503, failure)
        body: Dict[str, Any] = {
            "status": "ok",
            "version": self.version,
            "engine": _merge_stats([stats for stats, _ in checks]),
        }
        if self.engine_replicas > 1:
            body["replicas"] = self._replica_report([s for s, _ in checks])
        return body

    def _replica_report(
        self, per_replica: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        return [
            {
                "index": r.index,
                "port": r.port,
                "healthy": r.usable(),
                "inflight": r.inflight,
                "requests": r.requests,
                "stats": _merge_stats([stats]),
            }
            for r, stats in zip(self._replicas, per_replica)
        ]

    @endpoint(
        "/engine-health",
        health_check=HealthCheck(timeout_seconds=_HEALTH_CHECK_TIMEOUT_S),
    )
    async def engine_health(self) -> Dict[str, Any]:
        """The platform health check (on-demand after an error response, and
        periodic). Override ``health`` to change what it checks: an app has
        one health check, so a subclass must not declare another."""
        return await self.health()

    def _probe_header(self) -> Dict[str, str]:
        # A draining runner fails the probe, which would end its grace early.
        return {} if self._draining else _HEALTH_CHECK_HEADERS

    def _engine_reply(
        self, content: bytes, status: int, inference: bool = True
    ) -> Response:
        """An engine reply passed through as is. On a 5xx the gateway probes
        the health check before acting on the runner: one failed request does
        not mean the engine is down, and replacing the runner drops every
        stream on it. Only inference replies count toward
        ``engine_error_threshold``, for the replica that sent them."""
        if inference:
            self._route()._count_engine_reply(status)
        return Response(
            content=content,
            status_code=status,
            media_type="application/json",
            headers=self._probe_header() if status >= 500 else None,
        )

    def _engine_unavailable(self, replica: _Replica, exc: BaseException) -> Response:
        """The engine could not be reached or dropped the connection. Counts
        toward ``engine_error_threshold``, and the gateway probes the health
        check before acting on the runner, as for an engine 5xx. Running out of
        connection slots is proxy load, so it is neither counted nor probed."""
        if not isinstance(exc, self._busy_errors):
            replica._count_engine_reply(502)
        return self._unreachable(replica, exc)

    def _unreachable(self, replica: _Replica, exc: BaseException) -> Response:
        if isinstance(exc, self._busy_errors):
            print(f"[llm-app] {self._engine_label(replica)} busy: {exc!r}")
            # A bare 503 makes the gateway stop the runner; this one is fine.
            return JSONResponse(
                _error_body("engine busy", "engine_unavailable"),
                status_code=503,
                headers={"Retry-After": "1", "X-Fal-Stop-Runner": "false"},
            )
        print(f"[llm-app] {self._engine_label(replica)} unreachable: {exc!r}")
        return JSONResponse(
            _error_body("engine unavailable", "engine_unavailable"),
            status_code=503,
            headers={"Retry-After": "1", **self._probe_header()},
        )

    @endpoint("/v1/models")
    async def models(self) -> Response:
        if self._draining:
            return _shutting_down_response()
        replica = self._pick_replica(None)
        try:
            upstream = await self._client.get(f"{replica.url}/v1/models")
        except self._transport_errors as exc:
            return self._unreachable(replica, exc)
        return self._engine_reply(
            upstream.content, upstream.status_code, inference=False
        )

    async def engine_stats(self) -> JSONResponse:
        per_replica = await self._engine_stats()
        body: Dict[str, Any] = {
            "model": self.model,
            "engine": self.engine,
            "stats": _merge_stats(per_replica),
            "usage_totals": self._usage_totals,
            "boot": dict(self._boot),
        }
        if self.engine_replicas > 1:
            body["replicas"] = self._replica_report(per_replica)
        return JSONResponse(body)

    def _route(self) -> _Replica:
        """The replica the current request was routed to. Outside ``_proxy``
        (a subclass calling the engine directly) that is the first one."""
        return self._route_var.get() or self._replicas[0]

    def _pick_replica(self, session: Optional[str]) -> _Replica:
        """The replica that served ``session`` before, if it is still usable
        and not far busier than the rest; else the usable one with the fewest
        requests in flight."""
        replicas = self._replicas
        if len(replicas) == 1:
            return replicas[0]
        # With none usable the runner is failing its health check; any reply
        # (likely a 5xx) beats queueing here until it is replaced.
        usable = [r for r in replicas if r.usable()] or replicas
        choice = min(usable, key=lambda r: (r.inflight, r.index))
        key = None
        if session is not None:
            # A digest bounds the memory a long id can take.
            key = hashlib.blake2b(session.encode(), digest_size=16).digest()
            index = self._sessions.get(key)
            if index is not None and replicas[index] in usable:
                sticky = replicas[index]
                # A hot session id (say a constant ``user``) must not pile
                # every request onto one engine: past the least load plus the
                # larger of that load and the slack, it moves and loses its
                # cached prefix.
                slack = max(_STICKY_SLACK, choice.inflight)
                if sticky.inflight <= choice.inflight + slack:
                    self._sessions.move_to_end(key)
                    return sticky
        if key is not None:
            self._sessions[key] = choice.index
            self._sessions.move_to_end(key)
            while len(self._sessions) > _SESSION_ROUTES_MAX:
                self._sessions.popitem(last=False)
        return choice

    def _continuation_replica(
        self, path: str, payload: Dict[str, Any]
    ) -> Optional[_Replica]:
        """The replica that created ``previous_response_id``, if known. It is
        used even when busy or unhealthy: the engine keeps the conversation in
        memory, so any other replica would answer that the id is not found."""
        if path != "/v1/responses" or len(self._replicas) == 1:
            return None
        previous = payload.get("previous_response_id")
        if not isinstance(previous, str):
            return None
        index = self._response_routes.get(previous)
        if index is None:
            return None
        replica = self._replicas[index]
        if replica.proc is not None and replica.proc.poll() is not None:
            # The engine exited and its in-memory state went with it.
            del self._response_routes[previous]
            return None
        self._response_routes.move_to_end(previous)
        return replica

    def _note_response(self, response_id: Any, replica: _Replica) -> None:
        if not isinstance(response_id, str) or len(self._replicas) == 1:
            return
        self._response_routes[response_id] = replica.index
        self._response_routes.move_to_end(response_id)
        while len(self._response_routes) > _RESPONSE_ROUTES_MAX:
            self._response_routes.popitem(last=False)

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

    @endpoint("/v1/responses")
    async def responses(self, request: Request) -> Response:
        return await self._proxy(request, "/v1/responses")

    @staticmethod
    def _session_id(payload: Dict[str, Any], headers: Any) -> Optional[str]:
        return next(
            (
                v.strip()
                for v in (
                    headers.get("x-fal-runner-hint"),
                    payload.get("prompt_cache_key"),
                    payload.get("user"),
                )
                if isinstance(v, str) and v.strip()
            ),
            None,
        )

    def _note_prefix(self, payload: Dict[str, Any], headers: Any) -> None:
        if not self._prefix_hints_on():
            return
        raw = self._session_id(payload, headers)
        # The gateway matches the id literally, so a value the header cannot
        # carry (commas, spaces, non-ASCII, over 64 chars) is not advertised
        # at all; a hash of it could never match what the client sends.
        if raw is None or not _HINT_RE.fullmatch(raw):
            return
        if raw in self._recent_prefixes:
            self._recent_prefixes.remove(raw)
        self._recent_prefixes.append(raw)

    def _finish_chunk(self, path: str, reason: str) -> bytes:
        """Final SSE event with ``finish_reason`` ``reason`` (a
        ``response.incomplete`` event for the Responses API), then [DONE] for
        chat and completions."""
        if path == "/v1/responses":
            event = {
                "type": "response.incomplete",
                "response": {
                    "id": f"{reason}-{os.urandom(8).hex()}",
                    "object": "response",
                    "created_at": int(time.time()),
                    "status": "incomplete",
                    "model": self._served_name(),
                    "output": [],
                    "incomplete_details": {"reason": reason},
                },
            }
            return f"event: response.incomplete\ndata: {json.dumps(event)}\n\n".encode()
        chat = path == "/v1/chat/completions"
        choice: Dict[str, Any] = {"index": 0, "finish_reason": reason}
        if chat:
            choice["delta"] = {}
        else:
            choice["text"] = ""
        event = {
            "id": f"{reason}-{os.urandom(8).hex()}",
            "object": "chat.completion.chunk" if chat else "text_completion",
            "created": int(time.time()),
            "model": self._served_name(),
            "choices": [choice],
        }
        return f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode()

    async def _proxy(self, request: Request, path: str) -> Response:
        arrival = time.monotonic()
        # Kept for handle_exit, which can run where no loop is current.
        self._serving_loop = asyncio.get_running_loop()
        if self._draining:
            return _shutting_down_response()
        try:
            payload: Dict[str, Any] = await request.json()
        except ValueError:
            return _bad_request("invalid JSON body")
        if self._abort_fired:
            # The abort sweep ran while the body was being read, so this
            # request's event would never be set.
            return _shutting_down_response()
        if not isinstance(payload, dict):
            return _bad_request("body must be an object")
        stream_flag = payload.get("stream")
        if stream_flag is not None and not isinstance(stream_flag, bool):
            # The engine would coerce "true" or 1 and stream, while the proxy
            # would wait for one JSON reply.
            return _bad_request("`stream` must be a boolean")
        try:
            budget = _deadline_budget(
                request.headers, self.request_deadline_s, self.deadline_margin_s
            )
        except ValueError as exc:
            return _bad_request(str(exc))
        deadline = None if budget is None else arrival + budget
        payload["model"] = self._served_name()
        self._note_prefix(payload, request.headers)
        # Routing is independent of prefix_hints: those only advertise ids to
        # the gateway, while a replica's prefix cache pays off either way.
        replica = self._continuation_replica(path, payload) or self._pick_replica(
            self._session_id(payload, request.headers)
        )
        self._route_var.set(replica)

        abort = asyncio.Event()
        self._inflight.add(abort)
        replica.inflight += 1
        replica.requests += 1
        relaying = False
        try:
            if payload.get("stream") is not True:
                return await self._proxy_unary(
                    request, path, payload, abort, deadline, budget
                )

            if (
                self.report_usage
                and path != "/v1/responses"  # usage rides response.completed
                and "stream_options" not in payload
            ):
                payload["stream_options"] = {"include_usage": True}
            send = asyncio.ensure_future(self._open_upstream(path, payload))
            aborted = asyncio.ensure_future(abort.wait())
            limit, idle_bound = _wait_limit(deadline, self.stream_idle_timeout_s)
            try:
                await asyncio.wait(
                    {send, aborted},
                    timeout=limit,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except asyncio.CancelledError:
                send.cancel()
                self._count_if_cancelled()
                if send.done() and not send.cancelled() and not send.exception():
                    # Headers arrived in the same tick: close what we got.
                    await send.result().aclose()
                raise
            finally:
                aborted.cancel()
            if not send.done():
                # Still waiting for engine headers: cancelling closes the
                # connection, which aborts the engine request.
                send.cancel()
                return self._stream_cut_short(path, abort, deadline, idle_bound)
            try:
                upstream = send.result()
            except self._transport_errors as exc:
                return self._engine_unavailable(replica, exc)
            if upstream.status_code != 200:
                return await self._engine_error_reply(upstream, path, abort, deadline)
            relaying = True
            # The error count resets when the stream ends cleanly (_relay), so
            # an engine that answers and then dies mid-stream still adds up.
            stream = _Stream(self, upstream, path, abort, replica)
            return _RelayResponse(
                self._relay(stream, path, payload.get("n") in (None, 1), deadline),
                stream.close,
                media_type="text/event-stream",
            )
        finally:
            if not relaying:
                self._inflight.discard(abort)
                replica.inflight -= 1

    def _stream_cut_short(
        self,
        path: str,
        abort: asyncio.Event,
        deadline: Optional[float],
        idle_bound: bool = False,
    ) -> Response:
        """Reply for a stream that ended before relaying began: shutdown,
        idle timeout or deadline."""
        if abort.is_set() and not _expired(deadline):
            self._count_shutdown_abort()
            return _shutting_down_response()
        # An expired deadline wins here too, as in _relay.
        if idle_bound and not _expired(deadline):
            reason = "idle_timeout"
            self._count_idle_abort(path)
        else:
            reason = "deadline"
            self._count_deadline_abort()
        # Same shape as a mid-stream expiry (200 SSE), so stream clients
        # handle a deadline in one place.
        return StreamingResponse(
            iter([self._finish_chunk(path, reason)]),
            media_type="text/event-stream",
        )

    async def _engine_error_reply(
        self,
        upstream: Any,
        path: str,
        abort: asyncio.Event,
        deadline: Optional[float],
    ) -> Response:
        """Relay a non-200 stream reply, reading its body within the request
        deadline and until shutdown, as the stream itself would be."""
        read = asyncio.ensure_future(upstream.aread())
        aborted = asyncio.ensure_future(abort.wait())
        remaining = _remaining(deadline)
        limit = (
            _ERROR_BODY_TIMEOUT_S
            if remaining is None
            else min(remaining, _ERROR_BODY_TIMEOUT_S)
        )
        try:
            await asyncio.wait(
                {read, aborted}, timeout=limit, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            self._count_if_cancelled()
            raise
        finally:
            aborted.cancel()
            if not read.done():
                read.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.wait({read})
                await upstream.aclose()
        if not read.cancelled():
            body = read.result()
            return self._engine_reply(body, upstream.status_code)
        if abort.is_set() or _expired(deadline):
            # The engine already answered with this status, so it counts
            # toward engine_error_threshold even though the body is dropped.
            self._route()._count_engine_reply(upstream.status_code)
            return self._stream_cut_short(path, abort, deadline)
        return self._engine_reply(
            json.dumps(
                _error_body("engine error body timed out", "engine_error")
            ).encode(),
            upstream.status_code,
        )

    async def _open_upstream(self, path: str, payload: Dict[str, Any]) -> Any:
        """Start a streaming request on the routed replica; the result has the
        httpx.Response surface the relay uses (status_code, aread, aiter_raw,
        aclose)."""
        replica = self._route()
        if not self.raw_stream_client:
            return await self._client.send(
                self._client.build_request(
                    "POST", f"{replica.url}{path}", json=payload
                ),
                stream=True,
            )
        return await open_stream(
            "127.0.0.1",
            replica.port,
            path,
            json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode(),
            _ENGINE_CONNECT_TIMEOUT_S,
            self._stream_slots,
        )

    async def _relay(
        self,
        stream: _Stream,
        path: str,
        single_choice: bool = True,
        deadline: Optional[float] = None,
    ):
        # Only whole SSE events are forwarded, so the shutdown or deadline chunk
        # never lands inside a partial event. Abort and expiry are seen between
        # chunks: a client that stops reading blocks the write, and only the
        # platform kill at the end of the termination window ends that request.
        partial = b""
        done_sent = False
        finished = False
        # Responses ids are noted from the first event that carries one.
        creator = stream.replica
        want_id = path == "/v1/responses" and len(self._replicas) > 1
        chunks = stream.upstream.aiter_raw().__aiter__()
        aborted = asyncio.ensure_future(stream.abort.wait())
        pending: Optional[asyncio.Future] = None
        try:
            while True:
                idle_bound = False
                if not aborted.done() and not _expired(deadline):
                    pending = asyncio.ensure_future(chunks.__anext__())
                    limit, idle_bound = _wait_limit(
                        deadline, self.stream_idle_timeout_s
                    )
                    await asyncio.wait(
                        {pending, aborted},
                        timeout=limit,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                if pending is None or not pending.done():
                    # No read starts after an abort or expiry, so chunks
                    # already buffered upstream cannot starve either. An
                    # expired deadline wins: the caller's budget is spent, so
                    # a retryable shutdown would only re-run it. Without an
                    # abort this is the wait's timeout, which may fire a
                    # clock tick early.
                    shutdown = aborted.done() and not _expired(deadline)
                    idle = idle_bound and not aborted.done() and not _expired(deadline)
                    if shutdown:
                        reason = "shutdown"
                    elif idle:
                        reason = "idle_timeout"
                    else:
                        reason = "deadline"
                    if not done_sent:
                        if shutdown:
                            self._count_shutdown_abort()
                        elif idle:
                            self._count_idle_abort(path)
                        else:
                            self._count_deadline_abort()
                        # After a real finish reason only the terminator is
                        # missing; a second reason would invite a retry of a
                        # completed generation. With n > 1 other choices may
                        # still be open, so they get the reason.
                        yield (
                            b"data: [DONE]\n\n"
                            if finished and single_choice
                            else self._finish_chunk(path, reason)
                        )
                    return
                # A read that finished in the same round as an abort or expiry
                # is forwarded first (it may carry the finish event or usage);
                # the next round then ends the stream.
                read, pending = pending, None
                try:
                    chunk = read.result()
                except StopAsyncIteration:
                    if stream.replica is not None:
                        stream.replica._count_engine_reply(200)
                    if partial:
                        yield partial
                    return
                except self._transport_errors as exc:
                    # Headers were sent, so the client learns of it in-band:
                    # the partial event is dropped, then an "error" finish.
                    if stream.replica is not None:
                        stream.replica._count_engine_reply(502)
                    print(f"[llm-app] {path} stream from {self.engine} broke: {exc!r}")
                    if not done_sent:
                        # As above: after a real finish only [DONE] is missing.
                        yield (
                            b"data: [DONE]\n\n"
                            if finished and single_choice
                            else self._finish_chunk(path, "error")
                        )
                    return
                stream.feed(chunk)
                partial = partial + chunk if partial else chunk
                cut = _event_cut(partial)
                if not cut and len(partial) > _MAX_LINE_BYTES:
                    # Not SSE framed after all: stop holding it back.
                    cut = len(partial)
                if cut:
                    out, partial = partial[:cut], partial[cut:]
                    # Anchored: "data: [DONE]" may also appear inside content.
                    done_sent = done_sent or (
                        out.startswith(b"data: [DONE]") or b"\ndata: [DONE]" in out
                    )
                    if want_id and creator is not None and b'"response":' in out:
                        response_id = _sse_response_id(out)
                        if response_id is not None:
                            self._note_response(response_id, creator)
                            want_id = False
                    if path == "/v1/responses":
                        # No [DONE] here: the terminal event ends the stream.
                        if _RESPONSES_TERMINAL_RE.search(out):
                            done_sent = finished = True
                    elif not finished:
                        finished = bool(_FINISH_REASON_RE.search(out))
                    yield out
        finally:
            aborted.cancel()
            if pending is not None:
                if not pending.done():
                    pending.cancel()
                elif not pending.cancelled():
                    pending.exception()  # retrieved, so asyncio does not warn
            await stream.close()

    def _count_deadline_abort(self) -> None:
        self._usage_totals["deadline_aborts"] = (
            self._usage_totals.get("deadline_aborts", 0) + 1
        )

    def _count_if_cancelled(self) -> None:
        # Counts gateway cancels (POST <route>/cancel), and client disconnects
        # when cancel_on_disconnect is set; see fal._request_cancel.
        context = self.current_request
        if context is not None and context.cancelled:
            self._usage_totals["cancel_aborts"] = (
                self._usage_totals.get("cancel_aborts", 0) + 1
            )

    def _count_idle_abort(self, path: str) -> None:
        self._usage_totals["idle_aborts"] = self._usage_totals.get("idle_aborts", 0) + 1
        print(
            f"[llm-app] {path} stream got nothing from {self.engine} for "
            f"{self.stream_idle_timeout_s:g}s; aborted the engine request"
        )

    async def _proxy_unary(
        self,
        request: Request,
        path: str,
        payload: Dict[str, Any],
        abort: asyncio.Event,
        deadline: Optional[float],
        budget: Optional[float],
    ) -> Response:
        task = asyncio.ensure_future(
            self._client.post(f"{self._route().url}{path}", json=payload)
        )
        watcher = asyncio.ensure_future(_wait_for_disconnect(request))
        aborted = asyncio.ensure_future(abort.wait())
        try:
            await asyncio.wait(
                {task, watcher, aborted},
                timeout=_remaining(deadline),
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            # A gateway cancel lands here: dropping the upstream call closes
            # its connection, which makes the engine abort the request.
            task.cancel()
            self._count_if_cancelled()
            raise
        finally:
            watcher.cancel()
            aborted.cancel()
        if not task.done():
            # uvicorn does not cancel the handler on disconnect; cancel the
            # upstream call so the engine aborts the request.
            task.cancel()
            if watcher.done():
                return Response(status_code=499)
            # An expired deadline wins over shutdown, as in _relay.
            if abort.is_set() and not _expired(deadline):
                self._count_shutdown_abort()
                return _shutting_down_response()
            self._count_deadline_abort()
            message = f"request deadline of {max(budget or 0, 0):g}s exceeded"
            # The gateway marks a 503/504 without this header as retryable,
            # but a retry would hit the same deadline.
            return JSONResponse(
                {"error": {"message": message, "type": "deadline_exceeded"}},
                status_code=504,
                headers={"X-Fal-Needs-Retry": "false", **self._probe_header()},
            )
        try:
            upstream = task.result()
        except self._transport_errors as exc:
            return self._engine_unavailable(self._route(), exc)
        if upstream.status_code == 200:
            try:
                reply = upstream.json()
            except ValueError:
                reply = None
            if not isinstance(reply, dict):
                reply = {}
            if path == "/v1/responses":
                self._note_response(reply.get("id"), self._route())
            usage = reply.get("usage")
            if isinstance(usage, dict):
                try:
                    self.record_usage(path, usage)
                except Exception as exc:
                    # The request already succeeded; a hook bug must not fail it.
                    print(f"[llm-app] record_usage failed: {exc!r}")
        return self._engine_reply(upstream.content, upstream.status_code)
