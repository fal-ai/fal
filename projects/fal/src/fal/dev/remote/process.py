"""The resident app process of a `fal dev app` box: launched, watched and restarted.

Runs inside the holder, which is shipped by value: keep imports at module level
and limited to the standard library and ``fal``.
"""

import collections
import http.client
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fal.dev.remote.deps import Overlay
from fal.dev.remote.protocol import (
    CHILD_PORT,
    CHILD_PREFIX,
    SESSION_ENV,
    STARTING_PHASES,
    check_session_env,
    redact,
)
from fal.dev.remote.tree import extract, relative

LOG_LINES = 2000
CHILD_ERRORS = (OSError, ValueError, http.client.HTTPException)


def child_call(action: str, payload: dict, timeout: float) -> tuple[int, dict]:
    """POST to one of the app process's control routes."""
    connection = http.client.HTTPConnection("127.0.0.1", CHILD_PORT, timeout=timeout)
    try:
        connection.request(
            "POST",
            f"{CHILD_PREFIX}{action}",
            body=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        return response.status, json.loads(response.read() or b"{}")
    finally:
        connection.close()


class LogBuffer:
    """The box's recent log lines, numbered so a client can ask for the new ones."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.lines: collections.deque[str] = collections.deque(maxlen=LOG_LINES)
        self.total = 0

    def emit(self, line: str) -> None:
        with self.lock:
            self.lines.append(line)
            self.total += 1
        print(line, flush=True)

    def since(self, after: int | None, lines: int) -> dict[str, Any]:
        """Lines after number ``after``, or the last ``lines`` without it."""
        with self.lock:
            buffered = list(self.lines)
            total = self.total
        if after is None or after > total:
            selected = buffered[-lines:] if lines > 0 else []
        else:
            selected = buffered[max(0, len(buffered) - (total - after)) :]
        return {"lines": selected, "total": total}


def sdk_import_error(env: dict[str, str]) -> str | None:
    """Why the app process cannot import the shipped SDK, or None if it can."""
    probe = subprocess.run(
        [sys.executable, "-c", "import fal.app, fal.dev.remote.child, uvicorn"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return None if probe.returncode == 0 else (probe.stderr or "")[-2000:]


def install_sdk_deps(
    requirements: list[str], target: Path, log: Callable[[str], None]
) -> None:
    """Install the shipped SDK's dependencies into ``target``; logs the tail.

    Unlike the overlay's installer this must work on a runner without uv (it
    falls back to pip) and keeps the environment as it is, ``FAL_*`` included.
    """
    if uv := shutil.which("uv"):
        command = [uv, "pip", "install", "--python", sys.executable]
    else:
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
        ]
    result = subprocess.run(
        [*command, "--target", str(target), *requirements],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in (result.stdout + result.stderr).splitlines()[-20:]:
        log(f"[fal dev] {redact(line)}")


class ChildProcess:
    """Starts, watches and restarts the resident app process on this runner.

    ``start`` and ``stop`` are serialized, and each launch has a generation
    number: a launch still installing dependencies when it is superseded never
    spawns its process.
    """

    def __init__(self, session_dir: Path, logs: LogBuffer) -> None:
        self.session_dir = session_dir
        self.logs = logs
        self.src = session_dir / "src"
        self.sdk = session_dir / "sdk"
        self.sdk_deps = session_dir / "sdk-deps"
        self.overlay = Overlay(session_dir / "overlay")
        # Overlay version the running app process started with.
        self.launched_overlay: int | None = None
        # Guards the fields below; `control` serializes start and stop.
        self.lock = threading.Lock()
        self.control = threading.Lock()
        self.install_lock = threading.Lock()
        self.generation = 0
        self.proc: subprocess.Popen | None = None
        self.phase = "idle"
        self.detail = ""
        self.module = ""
        self.class_name = ""
        self.sdk_requirements: list[str] = []
        # More names for the box name in the app's environment.
        self.session_env: list[str] = []
        # sys.path entries and working directory, relative to src.
        self.paths: list[str] = ["."]
        self.workdir = "."
        self.started_at: float | None = None
        self.ready_at: float | None = None
        self.exit_code: int | None = None
        self.pending_reload = False

    def _env(self) -> dict[str, str]:
        # The shipped SDK goes first so `fal` matches the CLI that deployed the
        # holder, then the overlay, then the app's own environment; the SDK's
        # dependencies go last so the app's own versions win.
        paths = [str(self.sdk), str(self.overlay.site)]
        paths += [p for p in sys.path if p]
        if self.sdk_deps.exists():
            paths.append(str(self.sdk_deps))
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(paths)
        env["PYTHONUNBUFFERED"] = "1"
        env["FAL_DEV_APP"] = "1"
        for name in (SESSION_ENV, *self.session_env):
            env[name] = self.session_dir.name
        env["FAL_DEV_OVERLAY"] = str(self.overlay.site)
        return env

    def _install_sdk_deps(self, generation: int) -> None:
        error = sdk_import_error(self._env())
        if error is None:
            return
        with self.lock:
            if generation != self.generation:
                return
            self.phase, self.detail = "installing", "fal SDK dependencies"
        self.logs.emit(
            f"[fal dev] installing fal SDK dependencies: {error.strip()[-300:]}"
        )
        install_sdk_deps(self.sdk_requirements, self.sdk_deps, self.logs.emit)
        error = sdk_import_error(self._env())
        if error is not None:
            raise RuntimeError(f"fal is still not importable on the runner:\n{error}")

    def _pump(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            self.logs.emit(raw.rstrip("\n"))
        code = proc.wait()
        with self.lock:
            if self.proc is proc:
                self.exit_code = code
                self.phase = "exited"
                self.detail = f"exit code {code}"
                self.proc = None

    def _child_answers(self) -> bool:
        try:
            status, _ = child_call("status", {}, timeout=2)
        except CHILD_ERRORS:
            return False
        return status == 200

    def _wait_ready(self, proc: subprocess.Popen) -> None:
        while not self._child_answers():
            if proc.poll() is not None:
                return
            time.sleep(0.5)
        while True:
            with self.lock:
                if self.proc is not proc:
                    return
                if not self.pending_reload:
                    self.phase, self.detail = "ready", ""
                    self.ready_at = time.time()
                    return
                self.pending_reload = False
                generation = self.generation
            # Files changed while setup() ran; the app imported older code.
            # Reload before reporting ready so no request sees it.
            try:
                _, result = child_call("reload", {"files": []}, timeout=300)
            except CHILD_ERRORS as exc:
                result = {"action": "error", "reason": str(exc)}
            self.logs.emit(f"[fal dev] applied edits made during setup: {result}")
            if result.get("action") == "restart":
                threading.Thread(
                    target=self.start, kwargs={"generation": generation}, daemon=True
                ).start()
                return

    def _launch(self, generation: int) -> None:
        try:
            self._spawn(generation)
        except Exception as exc:
            with self.lock:
                if generation == self.generation:
                    self.phase, self.detail = "failed", str(exc)
            self.logs.emit(f"[fal dev] {exc}")

    def _spawn(self, generation: int) -> None:
        with self.install_lock:
            self._install_sdk_deps(generation)
        command = [
            sys.executable,
            "-u",
            "-m",
            "fal.dev.remote.child",
            "--root",
            str(self.src),
            "--module",
            self.module,
            "--cls",
            self.class_name,
            "--port",
            str(CHILD_PORT),
            # One argument each, so a path starting with "-" is not an option.
            f"--cwd={self.workdir}",
            *(f"--path={path}" for path in self.paths),
        ]
        with self.lock:
            if generation != self.generation:
                return
            self.phase, self.detail = "setup", "running setup()"
            self.started_at = time.time()
            self.ready_at = None
            self.launched_overlay = self.overlay.version
            self.exit_code = None
            proc = subprocess.Popen(
                command,
                cwd=self.src,
                env=self._env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                start_new_session=True,
            )
            self.proc = proc
        threading.Thread(target=self._pump, args=(proc,), daemon=True).start()
        self._wait_ready(proc)

    def _supersede(self, phase: str) -> subprocess.Popen | None:
        """Cancel any launch in flight and take the process, to be killed.

        Leaves ``phase`` behind, unless nothing ever ran here (``idle``).
        """
        with self.lock:
            self.generation += 1
            proc, self.proc = self.proc, None
            if proc is not None or self.phase != "idle":
                self.phase, self.detail = phase, ""
            return proc

    @staticmethod
    def _kill(proc: subprocess.Popen | None, timeout: float = 20) -> None:
        if proc is None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=timeout)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

    def _stop(self) -> None:
        self._kill(self._supersede("stopping"))
        with self.lock:
            if self.phase == "stopping":
                self.phase = "stopped"

    def stop(self) -> None:
        with self.control:
            self._stop()

    def replace_sdk(self, archive_b64: str) -> None:
        """Stop the app and unpack a new copy of the SDK for the next start."""
        fresh = Path(tempfile.mkdtemp(dir=self.session_dir, prefix="sdk.new-"))
        try:
            # A bad archive fails here, with the old SDK and the app untouched.
            extract(archive_b64, fresh)
            with self.control:
                self._stop()
                shutil.rmtree(self.sdk, ignore_errors=True)
                os.replace(fresh, self.sdk)
        finally:
            shutil.rmtree(fresh, ignore_errors=True)

    @staticmethod
    def check_start(
        layout: Any, session_env: Any
    ) -> tuple[list[str] | None, str | None, list[str] | None]:
        """Validate the optional parts of a start: (paths, workdir, session_env)."""
        paths = workdir = names = None
        if layout is not None:
            if not isinstance(layout, dict):
                raise ValueError("layout must be an object")
            raw = layout.get("paths") or ["."]
            if not isinstance(raw, list) or not all(isinstance(p, str) for p in raw):
                raise ValueError("layout paths must be a list of strings")
            paths = [str(relative(p)) for p in raw]
            workdir = str(relative(layout.get("workdir") or "."))
        if session_env is not None:
            if not isinstance(session_env, list):
                raise ValueError("session_env must be a list")
            names = check_session_env([str(name) for name in session_env])
        return paths, workdir, names

    def start(
        self,
        module: str | None = None,
        class_name: str | None = None,
        sdk_requirements: list[str] | None = None,
        generation: int | None = None,
        layout: dict[str, Any] | None = None,
        session_env: list[str] | None = None,
    ) -> bool:
        """(Re)start the app process; False when no app was ever started here.

        With ``generation``, only restart if nothing started or stopped the app
        since that generation was read.
        """
        # Checked before any state changes.
        paths, workdir, names = self.check_start(layout, session_env)
        with self.control:
            if not (module or self.module):
                return False
            if generation is not None and generation != self.generation:
                return False
            # Phase "starting" right away, so pollers never see a stopped app.
            self._kill(self._supersede("starting"))
            with self.lock:
                self.module = module or self.module
                self.class_name = class_name or self.class_name
                if sdk_requirements is not None:
                    self.sdk_requirements = sdk_requirements
                if paths is not None and workdir is not None:
                    self.paths, self.workdir = paths, workdir
                if names is not None:
                    self.session_env = names
                self.pending_reload = False
                self.phase, self.detail = "starting", ""
                generation = self.generation
            threading.Thread(
                target=self._launch, args=(generation,), daemon=True
            ).start()
            return True

    def reload(self, files: list[str]) -> dict[str, Any]:
        """Ask the app to reload after a sync, or note that it cannot yet."""
        with self.lock:
            phase = self.phase
            if phase in STARTING_PHASES:
                self.pending_reload = True
                return {"action": "pending", "reason": f"the app is in {phase}"}
            generation = self.generation
        if phase != "ready":
            if self.start():
                return {"action": "restart", "reason": f"the app was {phase}"}
            return {"action": "not_running", "reason": f"the app is {phase}"}
        try:
            _, result = child_call("reload", {"files": files}, timeout=300)
        except CHILD_ERRORS as exc:
            return {"action": "error", "reason": f"app not reachable: {exc}"}
        if result.get("action") == "restart":
            self.start(generation=generation)
        return result

    def restart_needed(self) -> bool:
        """The overlay changed since the running app process started."""
        return (
            self.proc is not None
            and self.launched_overlay is not None
            and self.launched_overlay != self.overlay.version
        )

    def status(self) -> dict[str, Any]:
        with self.lock:
            result: dict[str, Any] = {
                "phase": self.phase,
                "detail": self.detail,
                "module": self.module,
                "class_name": self.class_name,
                "pid": self.proc.pid if self.proc else None,
                "started_at": self.started_at,
                "ready_at": self.ready_at,
                "exit_code": self.exit_code,
                "overlay_restart_needed": self.restart_needed(),
            }
        if result["phase"] == "ready":
            try:
                _, result["app"] = child_call("status", {}, timeout=5)
            except CHILD_ERRORS as exc:
                result["app"] = {"error": str(exc)}
        return result
