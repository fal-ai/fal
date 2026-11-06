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
import tarfile
import threading
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import fastapi
from fastapi import Body

import fal
from fal.app import App
from fal.dev.errors import DevError
from fal.dev.remote import proxy
from fal.dev.remote.objects import Store
from fal.dev.remote.process import CHILD_ERRORS, ChildProcess, LogBuffer, child_call
from fal.dev.remote.protocol import (
    NOT_RELOADED,
    REQUEST_TIMEOUT_SECONDS,
    ROUTE_PREFIX,
    STOPPED_PHASES,
    TOKEN_FIELD,
    number,
    redact,
)
from fal.dev.remote.tree import SourceTree

STATE_ROOT = Path("/tmp/fal-dev-app")
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


class DevAppHolder:
    """Mixed in ahead of the user's app class by ``make_holder``."""

    _fal_dev_token: ClassVar[str] = ""
    _fal_dev_session: ClassVar[str] = "default"

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

    def collect_routes(self) -> dict[Any, Any]:
        return {
            signature: endpoint
            for signature, endpoint in App.collect_routes(self).items()  # type: ignore[arg-type]
            if signature.path.startswith(ROUTE_PREFIX)
        }

    async def _run(
        self, payload: Any, handler: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> dict[str, Any]:
        """Check the box token, then run ``handler(payload)`` in a thread.

        Any failure is a 400 with its message; unexpected ones are also logged.
        """
        if not isinstance(payload, dict):
            raise fastapi.HTTPException(422, "expected a JSON object")
        token = str(payload.get(TOKEN_FIELD, "")).encode()
        expected = self._fal_dev_token.encode()
        if not expected or not hmac.compare_digest(token, expected):
            raise fastapi.HTTPException(403, "bad or missing dev token")
        try:
            return await asyncio.to_thread(handler, payload)
        except Exception as exc:
            # Never a 5xx: fal replaces a runner whose app answers 503.
            if not isinstance(exc, REQUEST_ERRORS):
                self._log_buffer.emit(f"[fal dev] {redact(traceback.format_exc())}")
            raise fastapi.HTTPException(400, redact(str(exc))) from exc

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


def box_host_options(keep_alive: int, machine_type: str | None = None) -> dict:
    """Host options every box gets, whatever the app or pyproject.toml set."""
    options: dict[str, Any] = {
        "keep_alive": keep_alive,
        "min_concurrency": 0,
        "max_concurrency": 1,
        "concurrency_buffer": 0,
        "concurrency_buffer_perc": 0,
        "max_multiplexing": 8,
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
    options = box_host_options(keep_alive, machine_type)
    options.pop("health_check_config")
    namespace.update(options)
    return type(
        f"FalDev{target.__name__}",
        (DevAppHolder, target),
        namespace,
        name=name,
    )
