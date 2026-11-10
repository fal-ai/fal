"""The `fal dev app` box: a subclass of the user's app that holds its runner.

``make_holder(target)`` subclasses the user's ``fal.App`` so the box inherits
its machine type, requirements, image and secrets, but it skips the target's
``setup()`` and serves only ``/__fal_dev/*`` routes. Those routes write synced
source files, start and restart the resident app process (``process``), and
proxy requests to it (``proxy``).

This module is shipped to the runner by value (cloudpickle). ``fal`` is not
installed there, so the holder's process can use the fal code pickled with it
but cannot import anything new: keep imports at module level, and outside
``fal`` limited to the standard library and FastAPI. The app process gets
``fal`` from a copy of the local SDK sent with ``start`` (plus its dependencies
when the runner's environment lacks them).

Errors are 4xx, never 5xx: fal replaces a runner whose app answers 503.
"""

import asyncio
import hmac
import http.client
import json
import os
import tarfile
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

import fastapi
from fastapi import Body
from fastapi.responses import StreamingResponse

import fal
from fal.app import App
from fal.dev.errors import DevError
from fal.dev.remote import proxy
from fal.dev.remote.objects import Store
from fal.dev.remote.process import CHILD_ERRORS, ChildProcess, LogBuffer, child_call
from fal.dev.remote.protocol import (
    ACTIVITY_FILE,
    APP_PHASE_HEADER,
    BOX_STATE_ROOT,
    NOT_RELOADED,
    REQUEST_TIMEOUT_SECONDS,
    ROUTE_PREFIX,
    STOPPED_PHASES,
    TOKEN_FIELD,
    number,
    redact,
)
from fal.dev.remote.tree import SourceTree

STATE_ROOT = Path(BOX_STATE_ROOT)
# A streamed request whose app answered 5xx is passed on as 424 with the
# app's status here: fal replaces a runner whose app answers 503.
UPSTREAM_STATUS_HEADER = "X-Fal-Dev-App-Status"
# A dev box is one developer's; this many parallel streams is plenty.
MAX_BOX_MULTIPLEXING = 32
# Everything a route handler may raise for a request it cannot carry out.
REQUEST_ERRORS = (
    ValueError,
    TypeError,
    EOFError,
    OSError,
    tarfile.TarError,
    http.client.HTTPException,
    DevError,
)


class _StreamThreads:
    # Made on first use, on the box: the holder is shipped by value.
    executor: ThreadPoolExecutor | None = None


def _stream_threads() -> ThreadPoolExecutor:
    """Threads for proxied streams, apart from the control routes' threads.

    A stream holds one while it waits for the app, so slow streams never
    starve `status` or `logs`.
    """
    if _StreamThreads.executor is None:
        _StreamThreads.executor = ThreadPoolExecutor(
            MAX_BOX_MULTIPLEXING, thread_name_prefix="fal-dev-stream"
        )
    return _StreamThreads.executor


def _close_opened(opening: Future) -> None:
    """Close a stream whose request was over before it opened."""
    if not opening.cancelled() and opening.exception() is None:
        opening.result()[3]()


class _EndingResponse(StreamingResponse):
    """A streaming response that always runs ``end`` once it is over.

    The body's own finally does not run when the response fails before its
    first chunk (the client left at once), and Starlette then skips
    background tasks.
    """

    def __init__(self, end: Callable[[], None], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._end = end

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._end()


class Activity:
    """Box requests in flight and when the last one started or ended.

    Published to a file for `fal dev app dash`: the platform scales the box to
    zero `keep_alive` seconds after its last request, and the dashboard reads
    the file over a runner exec, which is not a request.
    """

    def __init__(self, path: Path, keep_alive: int) -> None:
        self.path = path
        self.keep_alive = keep_alive
        self.inflight = 0
        self._lock = threading.Lock()
        self.change(0)

    def change(self, delta: int) -> None:
        """Count a request starting (+1) or ending (-1), and publish it."""
        with self._lock:
            self.inflight = max(0, self.inflight + delta)
            state = {
                "last_request": time.time(),
                "inflight": self.inflight,
                "keep_alive": self.keep_alive,
            }
            tmp = self.path.with_suffix(".tmp")
            try:
                tmp.write_text(json.dumps(state))
                os.replace(tmp, self.path)
            except OSError:
                pass


class DevAppHolder:
    """Mixed in ahead of the user's app class by ``make_holder``."""

    _fal_dev_token: ClassVar[str] = ""
    _fal_dev_session: ClassVar[str] = "default"
    # Set by setup(); None in a holder that never ran it.
    _activity: Activity | None = None

    def setup(self) -> None:
        self._log_buffer = LogBuffer()
        self._child = ChildProcess(STATE_ROOT / self._fal_dev_session, self._log_buffer)
        self._child.src.mkdir(parents=True, exist_ok=True)
        self._tree = SourceTree(self._child.src, Store.open(), self._log_buffer.emit)
        # Files `place` copied since the last `files` call, which reloads them.
        self._placed: set[str] = set()
        self._placed_lock = threading.Lock()
        # One profile at a time: arming replaces any armed profile.
        self._profile_lock = threading.Lock()
        # A class attribute of the box class (box_host_options).
        keep_alive = getattr(type(self), "keep_alive", 0)
        self._activity = Activity(self._child.session_dir / ACTIVITY_FILE, keep_alive)
        print(f"[fal dev] box ready, session {self._fal_dev_session}", flush=True)

    def teardown(self) -> None:
        self._child.stop()

    # The target's exit handler, hints and health check belong to its setup()
    # state, which the box never builds.
    def handle_exit(self) -> None:
        return None

    def provide_hints(self) -> list[str]:
        return []

    @classmethod
    def get_health_check_config(cls):
        return None

    def health(self):
        return App.health(self)

    def _add_extra_routes(self, app: Any) -> None:
        # The target's extra routes (an LLMApp's /v1/models) need its setup().
        App._add_extra_routes(self, app)  # type: ignore[arg-type]

    def collect_routes(self) -> dict[Any, Any]:
        return {
            signature: endpoint
            for signature, endpoint in App.collect_routes(self).items()  # type: ignore[arg-type]
            if signature.path.startswith(ROUTE_PREFIX)
        }

    def _touch(self, change: int) -> None:
        if self._activity is not None:
            self._activity.change(change)

    def _check_token(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            raise fastapi.HTTPException(422, "expected a JSON object")
        token = str(payload.get(TOKEN_FIELD, "")).encode()
        expected = self._fal_dev_token.encode()
        if not expected or not hmac.compare_digest(token, expected):
            raise fastapi.HTTPException(403, "bad or missing dev token")

    def _bad_request(self, exc: Exception) -> fastapi.HTTPException:
        """A 400 with the error's message; unexpected errors are also logged."""
        # Never a 5xx: fal replaces a runner whose app answers 503.
        if not isinstance(exc, REQUEST_ERRORS):
            self._log_buffer.emit(f"[fal dev] {redact(traceback.format_exc())}")
        return fastapi.HTTPException(400, redact(str(exc)))

    async def _run(
        self, payload: Any, handler: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> dict[str, Any]:
        """Check the box token, then run ``handler(payload)`` in a thread."""
        self._check_token(payload)
        self._touch(1)
        try:
            return await asyncio.to_thread(handler, payload)
        except Exception as exc:
            raise self._bad_request(exc) from exc
        finally:
            self._touch(-1)

    async def _stream(self, payload: Any) -> StreamingResponse:
        """Pass one request through to the app and stream its response back."""
        self._check_token(payload)
        phase = self._child.phase
        if phase != "ready":
            # The header tells `fal dev app proxy` it can boot the app again
            # (a box scaled to zero comes back idle) and retry.
            raise fastapi.HTTPException(
                400,
                f"the app is not ready (phase: {phase})",
                headers={APP_PHASE_HEADER: phase},
            )
        self._touch(1)
        opening = None
        try:
            opening = _stream_threads().submit(proxy.stream, payload)
            status, content_type, chunks, close = await asyncio.wrap_future(opening)
        except BaseException as exc:
            # Cancellation too: the request is over either way, and a stream
            # that opens after it is closed again.
            self._touch(-1)
            if opening is not None:
                opening.add_done_callback(_close_opened)
            if isinstance(exc, Exception):
                raise self._bad_request(exc) from exc
            raise
        once = threading.Lock()

        def end() -> None:
            # Called when the stream ends and when the response is done; only
            # the first call counts. close() also stops a read in progress.
            if once.acquire(blocking=False):
                close()
                self._touch(-1)

        async def relay():
            # Async, so a client that goes away (or a cancelled request) runs
            # the finally at once instead of when the generator is collected.
            loop = asyncio.get_running_loop()
            try:
                while chunk := await loop.run_in_executor(
                    _stream_threads(), next, chunks, b""
                ):
                    yield chunk
            finally:
                end()

        headers = {}
        if status >= 500:
            headers[UPSTREAM_STATUS_HEADER] = str(status)
            status = 424
        try:
            return _EndingResponse(
                end,
                relay(),
                status_code=status,
                media_type=content_type,
                headers=headers,
            )
        except BaseException:
            end()
            raise

    def _require_ready(self) -> None:
        phase = self._child.phase
        if phase != "ready":
            raise ValueError(f"the app is not ready (phase: {phase})")

    def _status(self, payload: dict[str, Any]) -> dict[str, Any]:
        status = self._child.status()
        status["session"] = self._fal_dev_session
        return status

    def _start(self, payload: dict[str, Any]) -> dict[str, Any]:
        # `start` checks these again, but only after the SDK swap has stopped
        # the running app.
        for key in ("module", "class_name"):
            value = payload.get(key)
            if value is not None and (
                not isinstance(value, str) or value.startswith("-")
            ):
                raise ValueError(f"{key} must be a name, not {value!r}")
        self._child.check_start(payload.get("layout"), payload.get("session_env"))
        if payload.get("sdk_archive"):
            self._child.replace_sdk(payload["sdk_archive"])
        started = self._child.start(
            payload.get("module"),
            payload.get("class_name"),
            payload.get("sdk_requirements"),
            layout=payload.get("layout"),
            session_env=payload.get("session_env"),
        )
        if not started:
            raise ValueError("no app has been started on this box")
        return self._child.status()

    def _place(self, payload: dict[str, Any]) -> dict[str, Any]:
        placed = self._tree.place(payload.get("files") or {})
        with self._placed_lock:
            self._placed.update(placed)
        return {"placed": placed}

    def _files(self, payload: dict[str, Any]) -> dict[str, Any]:
        written, deleted = self._tree.sync(
            archive=payload.get("archive"),
            deleted=list(payload.get("deleted") or []),
            all_files=payload.get("all_files"),
        )
        result: dict[str, Any] = {"written": written, "deleted": deleted}
        if payload.get("reload"):
            with self._placed_lock:
                placed = sorted(self._placed)
            result["reload"] = self._child.reload(written + deleted + placed)
            # Kept for the next reload unless this one took them.
            if result["reload"].get("action") not in NOT_RELOADED:
                with self._placed_lock:
                    self._placed.difference_update(placed)
        return result

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_ready()
        return proxy.forward_runs(payload)

    def _profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_ready()
        if not self._profile_lock.acquire(blocking=False):
            raise ValueError("another profile is running on this box")
        try:
            return proxy.profile(payload, self._child.session_dir / "profiles")
        finally:
            self._profile_lock.release()

    def _artifact(self, payload: dict[str, Any]) -> dict[str, Any]:
        return proxy.artifact(payload, self._child.session_dir / "profiles")

    def _deps(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Change or show the dependency overlay; restart the app to apply it."""
        child = self._child
        overlay = child.overlay
        action = payload.get("action")
        before = overlay.version
        result: dict[str, Any] = {}
        if action == "install":
            index_url = payload.get("index_url")
            result = overlay.install(
                [str(spec) for spec in payload.get("packages") or []],
                no_deps=bool(payload.get("no_deps")),
                pre=bool(payload.get("pre")),
                index_url=str(index_url) if index_url else None,
                extra_index_urls=[
                    str(url) for url in payload.get("extra_index_urls") or []
                ],
                force=bool(payload.get("force")),
            )
        elif action == "uninstall":
            names = [str(name) for name in payload.get("packages") or []]
            result["removed"] = overlay.uninstall(names)
        elif action == "reset":
            result["removed"] = overlay.reset()
        elif action != "status":
            raise ValueError(f"unknown deps action {action!r}")
        result["packages"] = overlay.packages()
        # What a new runner needs to install the same overlay: URLs as given,
        # credentials included, so the CLI keeps it and never prints it.
        result["replay"] = overlay.state()
        # A stopped app stays stopped; it picks the overlay up when it starts.
        result["restarted"] = bool(
            overlay.version != before
            and payload.get("restart")
            and child.phase not in STOPPED_PHASES
            and child.start()
        )
        result["restart_needed"] = child.restart_needed()
        names = sorted(
            {package["name"] for package in result["packages"]}
            | {str(name) for name in payload.get("names") or []}
        )
        if action == "status" and child.phase == "ready" and names:
            try:
                _, result["app_versions"] = child_call("versions", {"names": names}, 30)
            except CHILD_ERRORS as exc:
                result["app_versions"] = {"error": str(exc)}
        return result

    def _logs(self, payload: dict[str, Any]) -> dict[str, Any]:
        after = None
        if payload.get("after") is not None:
            after = number(payload, "after", 0, 0, 2**62)
        return self._log_buffer.since(after, number(payload, "lines", 200, 0, 2**31))

    @fal.endpoint(f"{ROUTE_PREFIX}status")
    async def dev_status(self, payload: Any = Body(...)) -> dict[str, Any]:
        return await self._run(payload, self._status)

    @fal.endpoint(f"{ROUTE_PREFIX}start")
    async def dev_start(self, payload: Any = Body(...)) -> dict[str, Any]:
        return await self._run(payload, self._start)

    @fal.endpoint(f"{ROUTE_PREFIX}place")
    async def dev_place(self, payload: Any = Body(...)) -> dict[str, Any]:
        return await self._run(payload, self._place)

    @fal.endpoint(f"{ROUTE_PREFIX}files")
    async def dev_files(self, payload: Any = Body(...)) -> dict[str, Any]:
        return await self._run(payload, self._files)

    @fal.endpoint(f"{ROUTE_PREFIX}request")
    async def dev_request(self, payload: Any = Body(...)) -> dict[str, Any]:
        return await self._run(payload, self._request)

    @fal.endpoint(f"{ROUTE_PREFIX}proxy")
    async def dev_proxy(self, payload: Any = Body(...)) -> StreamingResponse:
        return await self._stream(payload)

    @fal.endpoint(f"{ROUTE_PREFIX}profile")
    async def dev_profile(self, payload: Any = Body(...)) -> dict[str, Any]:
        return await self._run(payload, self._profile)

    @fal.endpoint(f"{ROUTE_PREFIX}artifact")
    async def dev_artifact(self, payload: Any = Body(...)) -> dict[str, Any]:
        return await self._run(payload, self._artifact)

    @fal.endpoint(f"{ROUTE_PREFIX}deps")
    async def dev_deps(self, payload: Any = Body(...)) -> dict[str, Any]:
        return await self._run(payload, self._deps)

    @fal.endpoint(f"{ROUTE_PREFIX}logs")
    async def dev_logs(self, payload: Any = Body(...)) -> dict[str, Any]:
        return await self._run(payload, self._logs)


def box_host_options(
    keep_alive: int, machine_type: str | None = None, max_multiplexing: int = 1
) -> dict:
    """Host options every box gets, whatever the app or pyproject.toml set.

    An app that takes many requests at once (an LLM server) keeps that on the
    box, so `proxy` can carry a client's parallel requests, up to
    ``MAX_BOX_MULTIPLEXING``: each open stream holds one of the holder's
    stream threads while it waits for the app.
    """
    options: dict[str, Any] = {
        "keep_alive": keep_alive,
        "min_concurrency": 0,
        "max_concurrency": 1,
        "concurrency_buffer": 0,
        "concurrency_buffer_perc": 0,
        "max_multiplexing": min(MAX_BOX_MULTIPLEXING, max(8, max_multiplexing)),
        "request_timeout": REQUEST_TIMEOUT_SECONDS,
        # The box serves only its own routes, not the app's health check.
        "health_check_config": None,
    }
    if machine_type:
        options["machine_type"] = machine_type
    return options


def make_holder(
    target: type[App],
    *,
    name: str,
    token: str,
    keep_alive: int,
    machine_type: str | None = None,
) -> type[App]:
    """A box class that inherits ``target``'s environment but not its setup()."""
    namespace: dict[str, Any] = {
        "__module__": __name__,
        "__qualname__": f"FalDev{target.__name__}",
        "_fal_dev_token": token,
        "_fal_dev_session": name,
        # App.__init_subclass__ would otherwise keep the target's name.
        "app_name": None,
        "app_auth": "private",
        # The dev sync sends the app's files; the box uploads none.
        "app_files": [],
        "app_files_context_dir": None,
    }
    # Class attributes win over the target's scaling, header kwargs included.
    options = box_host_options(keep_alive, machine_type, target.max_multiplexing or 1)
    options.pop("health_check_config")
    namespace.update(options)
    return type(
        f"FalDev{target.__name__}",
        (DevAppHolder, target),
        namespace,
        name=name,
    )
