"""What the box does with a request for the app process: forward, time, profile.

Runs inside the holder, which is shipped by value: keep imports at module level
and limited to the standard library and ``fal``.
"""

import base64
import contextlib
import http.client
import json
import socket
import time
import urllib.parse
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from fal.dev.remote import profile as app_profile
from fal.dev.remote.process import CHILD_ERRORS, CHILD_PORT, child_call
from fal.dev.remote.protocol import (
    BODY_TAIL_CHARS,
    CHILD_PREFIX,
    MAX_PROFILE_WARMUPS,
    MAX_REPEAT,
    MAX_WARMUPS,
    REQUEST_TIMEOUT_SECONDS,
    STREAM_HEADERS,
    number,
)

MAX_RESPONSE_BYTES = 20 * 1024 * 1024
# About 5.3 MiB once base64-encoded in the JSON response.
ARTIFACT_CHUNK_BYTES = 4 * 1024 * 1024
ARTIFACTS = (app_profile.TRACE_NAME, app_profile.OPS_NAME)
# A batch of runs ends within this, inside the box's request timeout.
RUNS_BUDGET_SECONDS = 3300
HTTP_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")
STREAM_CHUNK_BYTES = 64 * 1024


def _timeout(payload: dict) -> float:
    return number(payload, "timeout", float(REQUEST_TIMEOUT_SECONDS), 1.0, 1e6)


def _remaining(deadline: float, timeout: float) -> float | None:
    """The socket timeout for the next request, or None once the budget is spent."""
    left = deadline - time.monotonic()
    return min(timeout, left) if left > 0 else None


def _target(request: dict) -> tuple[str, str]:
    """The checked method and path of a request for the app process."""
    method = str(request.get("method", "POST")).upper()
    if method not in HTTP_METHODS:
        raise ValueError(f"unsupported method {method!r}")
    path = str(request.get("path", "/"))
    # The app process decodes the path, so check the decoded form.
    if not path.startswith("/") or urllib.parse.unquote(path).startswith(CHILD_PREFIX):
        raise ValueError(f"invalid path {path!r}")
    return method, path


def stream(request: dict) -> tuple[int, str, Iterator[bytes], Callable[[], None]]:
    """Open one request to the app process and stream its response back.

    The body arrives base64-encoded, so any client request (an OpenAI SDK's
    streamed chat completion, say) fits in the box's JSON envelope. Returns
    the status, the content type, the body chunks as the app sends them, and
    a close function for when the caller goes away mid-stream.
    """
    method, path = _target(request)
    body = base64.b64decode(str(request.get("body_base64") or ""), validate=True)
    headers = {
        str(key): str(value)
        for key, value in (request.get("headers") or {}).items()
        if str(key).lower() in STREAM_HEADERS
    }
    connection = http.client.HTTPConnection(
        "127.0.0.1", CHILD_PORT, timeout=_timeout(request)
    )
    try:
        connection.request(method, path, body=body or None, headers=headers)
        # The response takes the socket over; keep it to stop a read with.
        sock = connection.sock
        response = connection.getresponse()
    except BaseException:
        connection.close()
        raise

    def close() -> None:
        # close() alone leaves a read blocked in another thread until the app
        # sends more; shutting the socket down ends that read at once.
        with contextlib.suppress(OSError):
            if sock is not None:
                sock.shutdown(socket.SHUT_RDWR)
        connection.close()

    def chunks() -> Iterator[bytes]:
        try:
            while chunk := response.read1(STREAM_CHUNK_BYTES):
                yield chunk
        finally:
            close()

    content_type = response.headers.get("content-type", "application/octet-stream")
    return response.status, content_type, chunks(), close


def forward(request: dict, timeout: float) -> dict:
    """Send one request to the app process and wrap its response."""
    method, path = _target(request)
    body = request.get("body")
    data = json.dumps(body).encode() if body is not None else None
    started = time.perf_counter()
    connection = http.client.HTTPConnection("127.0.0.1", CHILD_PORT, timeout=timeout)
    try:
        connection.request(
            method, path, body=data, headers={"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        content = response.read(MAX_RESPONSE_BYTES + 1)
        content_type = response.headers.get("content-type", "")
        status = response.status
    finally:
        connection.close()
    if len(content) > MAX_RESPONSE_BYTES:
        raise ValueError("the app's response is over the 20 MiB proxy limit")
    result: dict[str, Any] = {
        "status_code": status,
        "content_type": content_type,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    try:
        result["body"] = content.decode()
    except UnicodeDecodeError:
        result["body_base64"] = base64.b64encode(content).decode()
    return result


def forward_runs(payload: dict) -> dict:
    """Send the request ``warmups`` + ``repeat`` times, timing each one.

    Runs back to back on the box, so the times leave out the trip to fal.run.
    Only the last response's body is returned; a failed run ends the batch and
    returns its response instead. After the first run, each run's socket
    timeout is what is left of ``RUNS_BUDGET_SECONDS`` (a soft cap: a response
    that keeps trickling in can run past it). A run that does not fit, or fails
    to reach the app, ends the batch early with the runs so far (``stopped``
    says why).
    """
    warmups = number(payload, "warmups", 0, 0, MAX_WARMUPS)
    repeat = number(payload, "repeat", 1, 1, MAX_REPEAT)
    timeout = _timeout(payload)
    warmup_runs: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    result: dict[str, Any] = {}
    stopped = ""
    deadline = time.monotonic() + RUNS_BUDGET_SECONDS
    for index in range(warmups + repeat):
        run_timeout = timeout
        if index:
            left = _remaining(deadline, timeout)
            if left is None:
                stopped = "the batch reached its time budget"
                break
            run_timeout = left
        try:
            result = forward(payload, run_timeout)
        except CHILD_ERRORS as exc:
            if not index:
                raise
            stopped = f"run {index + 1} did not complete: {exc}"
            break
        run = {
            "status_code": result["status_code"],
            "elapsed_seconds": result["elapsed_seconds"],
        }
        (warmup_runs if index < warmups else runs).append(run)
        if result["status_code"] >= 400:
            break
    return {**result, "warmup_runs": warmup_runs, "runs": runs, "stopped": stopped}


def profile(payload: dict, root: Path) -> dict:
    """Warm up, then send one request to ``path`` under the torch profiler.

    Warmups and the profiled request share ``RUNS_BUDGET_SECONDS``, so the
    call answers inside the box's request timeout.
    """
    path = str(payload.get("path", "/"))
    mode = str(payload.get("mode", "standard"))
    warmups = number(payload, "warmups", 1, 0, MAX_PROFILE_WARMUPS)
    top = number(payload, "top", 15, 1, 100)
    timeout = _timeout(payload)
    deadline = time.monotonic() + RUNS_BUDGET_SECONDS
    body = payload.get("body")
    request = {"path": path, "body": body, "method": "POST"}

    def send() -> dict:
        run_timeout = _remaining(deadline, timeout)
        if run_timeout is None:
            raise ValueError("warmups used up the time budget; pass fewer --warmups")
        return forward(request, run_timeout)

    for _ in range(warmups):
        warm = send()
        if warm["status_code"] >= 400:
            failed = {k: warm[k] for k in ("status_code", "elapsed_seconds")}
            failed["body"] = warm.get("body", "")[-BODY_TAIL_CHARS:]
            return {"status": "request_failed", "request": failed}
    profile_id = uuid.uuid4().hex
    spec = {
        "id": profile_id,
        "path": path,
        "mode": mode,
        "dir": str(root / profile_id),
        "top": top,
    }
    status, armed = child_call("profile/arm", spec, timeout=30)
    if status != 200:
        raise ValueError(armed.get("detail", f"could not arm the profile ({status})"))
    try:
        response = send()
    except BaseException:
        # Disarm the profile, without hiding why the request failed.
        with contextlib.suppress(*CHILD_ERRORS):
            child_call("profile/result", {"id": profile_id}, timeout=30)
        raise
    # Also disarms the profile when the request never reached the endpoint.
    _, result = child_call("profile/result", {"id": profile_id}, timeout=30)
    result["request"] = {
        "status_code": response["status_code"],
        "elapsed_seconds": response["elapsed_seconds"],
    }
    if response["status_code"] >= 400:
        result["request"]["body"] = response.get("body", "")[-BODY_TAIL_CHARS:]
    result["id"] = profile_id
    return result


def artifact(payload: dict, root: Path) -> dict:
    """One chunk of a profile's trace or ops table."""
    profile_id = str(payload.get("id", ""))
    name = str(payload.get("name", ""))
    if not profile_id.isalnum() or name not in ARTIFACTS:
        raise ValueError("unknown profile artifact")
    path = root / profile_id / name
    if not path.is_file():
        raise ValueError(f"{name} of profile {profile_id} is gone from the box")
    size = path.stat().st_size
    offset = number(payload, "offset", 0, 0, size)
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(ARTIFACT_CHUNK_BYTES)
    return {"size": size, "offset": offset, "data": base64.b64encode(data).decode()}
