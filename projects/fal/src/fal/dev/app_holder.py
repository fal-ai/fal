"""The `fal dev app` box: a subclass of the user's app that holds its runner.

``make_holder(target)`` subclasses the user's ``fal.App`` so the box inherits
its machine type, requirements, image and secrets, but it skips the target's
``setup()`` and serves only ``/__fal_dev/*`` routes. Those routes write synced
source files, start and restart the resident app process (``app_child``), and
proxy requests to it.

This module is shipped to the runner by value (cloudpickle). ``fal`` is not
installed there, so the holder's process can use the fal code pickled with it
but cannot import anything new: keep imports at module level, and outside
``fal`` limited to the standard library and FastAPI. The child process gets
``fal`` from a copy of the local SDK sent with ``start`` (plus its dependencies
when the runner's environment lacks them).

Errors are 4xx, never 5xx: fal replaces a runner whose app answers 503.
"""

import asyncio
import base64
import collections
import hmac
import http.client
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.parse
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

import fastapi
from fastapi import Body

import fal
from fal.app import App
from fal.dev import app_profile

CHILD_PORT = 8181
STATE_ROOT = Path("/tmp/fal-dev-app")
MAX_RESPONSE_BYTES = 20 * 1024 * 1024
MAX_WARMUPS = 50
MAX_REPEAT = 100
MAX_PROFILE_WARMUPS = 10
# Profile folders kept on the runner; older ones are deleted.
PROFILES_KEPT = 3
ARTIFACT_CHUNK_BYTES = 8 * 1024 * 1024
ARTIFACTS = (app_profile.TRACE_NAME, app_profile.OPS_NAME, app_profile.STATUS_NAME)
# A batch of runs ends within this, inside the box's 3600 s request timeout.
RUNS_BUDGET_SECONDS = 3300
LOG_LINES = 2000
ROUTE_PREFIX = "/__fal_dev/"
CHILD_PREFIX = "/__fal_dev_child/"
TOKEN_FIELD = "token"
# Phases in which the child is being brought up; edits made then are applied
# by a reload once it is ready.
STARTING_PHASES = ("starting", "installing", "setup")
# Phases in which no app process is running or coming up.
STOPPED_PHASES = ("idle", "exited", "failed", "stopped")
CHILD_ERRORS = (OSError, ValueError, http.client.HTTPException)
HTTP_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")
REQUEST_ERRORS = (ValueError, TypeError, EOFError, OSError, tarfile.TarError)


def _rel(raw: str) -> PurePosixPath:
    rel = PurePosixPath(raw)
    if not raw or rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"refusing path {raw!r}")
    return rel


def _extract(archive_b64: str, dest: Path) -> list[str]:
    """Unpack a tar.gz into dest, refusing paths that escape it."""
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(archive_b64))) as archive:
        for member in archive.getmembers():
            rel = _rel(member.name)
            if not member.isfile():
                raise ValueError(f"refusing archive entry {member.name!r}")
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            assert source is not None
            tmp = target.with_name(f".{target.name}.fal-dev-tmp")
            tmp.write_bytes(source.read())
            tmp.chmod(0o755 if member.mode & 0o111 else 0o644)
            # The running process never sees a half-written file.
            os.replace(tmp, target)
            written.append(str(rel))
    return written


def _remove(rel: str, dest: Path) -> bool:
    """Delete one file, then any folders it leaves empty."""
    target = dest / _rel(rel)
    if not target.is_file():
        return False
    target.unlink()
    parent = target.parent
    while parent != dest and not any(parent.iterdir()):
        parent.rmdir()
        parent = parent.parent
    return True


def _delete(paths: list[str], dest: Path) -> list[str]:
    return [raw for raw in paths if _remove(raw, dest)]


def _prune(keep: list[str], synced: set[str], dest: Path) -> list[str]:
    """Delete previously synced files that are not in keep (a full sync).

    Files the app wrote itself are left alone.
    """
    wanted = {str(_rel(raw)) for raw in keep}
    return [rel for rel in sorted(synced - wanted) if _remove(rel, dest)]


def _child_call(action: str, payload: dict, timeout: float) -> tuple[int, dict]:
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


class _ChildProcess:
    """Starts, watches and restarts the resident app process on this runner.

    ``start``, ``restart`` and ``stop`` are serialized, and each launch has a
    generation number: a launch still installing dependencies when it is
    superseded never spawns its process.
    """

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.src = session_dir / "src"
        self.sdk = session_dir / "sdk"
        self.deps = session_dir / "sdk-deps"
        self.lock = threading.RLock()
        self.control = threading.Lock()
        self.install_lock = threading.Lock()
        self.profile_lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.generation = 0
        self.proc: subprocess.Popen | None = None
        self.phase = "idle"
        self.detail = ""
        self.module = ""
        self.class_name = ""
        self.requirements: list[str] = []
        self.started_at: float | None = None
        self.ready_at: float | None = None
        self.exit_code: int | None = None
        self.restarts = 0
        self.pending_reload = False
        # Files written by syncs, so a full sync only prunes what it put there.
        self.synced: set[str] = set()
        self.log: collections.deque[str] = collections.deque(maxlen=LOG_LINES)
        self.log_total = 0

    def _emit(self, line: str) -> None:
        with self.log_lock:
            self.log.append(line)
            self.log_total += 1
        print(line, flush=True)

    def log_since(self, after: int | None, lines: int) -> dict[str, Any]:
        """Log lines after line number ``after`` (or the last ``lines``)."""
        with self.log_lock:
            buffered = list(self.log)
            total = self.log_total
        if after is None or after > total:
            selected = buffered[-lines:] if lines > 0 else []
        else:
            selected = buffered[max(0, len(buffered) - (total - after)) :]
        return {"lines": selected, "total": total}

    def _pythonpath(self) -> str:
        # The shipped SDK goes first so `fal` matches the CLI that deployed the
        # holder; its dependencies go last so the app's own versions win.
        paths = [str(self.sdk), *[p for p in sys.path if p]]
        if self.deps.exists():
            paths.append(str(self.deps))
        return os.pathsep.join(paths)

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = self._pythonpath()
        env["PYTHONUNBUFFERED"] = "1"
        env["FAL_DEV_APP"] = "1"
        return env

    def _fal_importable(self) -> tuple[bool, str]:
        probe = subprocess.run(
            [sys.executable, "-c", "import fal.app, fal.dev.app_child, uvicorn"],
            env=self._env(),
            capture_output=True,
            text=True,
            check=False,
        )
        return probe.returncode == 0, (probe.stderr or "")[-2000:]

    def _install_sdk_deps(self, generation: int) -> None:
        ok, error = self._fal_importable()
        if ok:
            return
        with self.lock:
            if generation != self.generation:
                return
            self.phase, self.detail = "installing", "fal SDK dependencies"
        self._emit(f"[fal dev] installing fal SDK dependencies: {error.strip()[-300:]}")
        requirements = self.session_dir / "sdk-requirements.txt"
        requirements.write_text("\n".join(self.requirements) + "\n")
        uv = shutil.which("uv")
        if uv:
            command = [uv, "pip", "install", "--python", sys.executable]
        else:
            command = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
            ]
        command += ["--target", str(self.deps), "-r", str(requirements)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        for line in (result.stdout + result.stderr).splitlines()[-20:]:
            self._emit(f"[fal dev] {line}")
        ok, error = self._fal_importable()
        if not ok:
            raise RuntimeError(f"fal is still not importable on the runner:\n{error}")

    def _pump(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            self._emit(raw.rstrip("\n"))
        code = proc.wait()
        with self.lock:
            if self.proc is proc:
                self.exit_code = code
                self.phase = "exited"
                self.detail = f"exit code {code}"
                self.proc = None

    def _wait_ready(self, proc: subprocess.Popen) -> None:
        while proc.poll() is None:
            try:
                status, _ = _child_call("status", {}, timeout=2)
            except CHILD_ERRORS:
                time.sleep(0.5)
                continue
            if status != 200:
                time.sleep(0.5)
                continue
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
                    _, result = _child_call("reload", {"files": []}, timeout=300)
                except CHILD_ERRORS as exc:
                    result = {"action": "error", "reason": str(exc)}
                self._emit(f"[fal dev] applied edits made during setup: {result}")
                if result.get("action") == "restart":
                    threading.Thread(
                        target=self.start,
                        kwargs={"generation": generation},
                        daemon=True,
                    ).start()
                    return

    def _launch(self, generation: int) -> None:
        try:
            self._launch_once(generation)
        except Exception as exc:
            with self.lock:
                if generation == self.generation:
                    self.phase, self.detail = "failed", str(exc)
            self._emit(f"[fal dev] {exc}")

    def _launch_once(self, generation: int) -> None:
        with self.install_lock:
            self._install_sdk_deps(generation)
        command = [
            sys.executable,
            "-u",
            "-m",
            "fal.dev.app_child",
            "--root",
            str(self.src),
            "--module",
            self.module,
            "--cls",
            self.class_name,
            "--port",
            str(CHILD_PORT),
        ]
        with self.lock:
            if generation != self.generation:
                return
            self.phase, self.detail = "setup", "running setup()"
            self.started_at = time.time()
            self.ready_at = None
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

    def _stop(self, timeout: float = 20, final: str = "stopped") -> None:
        with self.lock:
            self.generation += 1
            proc, self.proc = self.proc, None
            if proc is None:
                if self.phase != "idle":
                    self.phase, self.detail = final, ""
                return
            self.phase, self.detail = "stopping", ""
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=timeout)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        with self.lock:
            if self.phase == "stopping":
                self.phase, self.detail = final, ""

    def stop(self) -> None:
        with self.control:
            self._stop()

    def replace_sdk(self, archive_b64: str) -> None:
        """Stop the app and unpack a new copy of the SDK for the next start."""
        with self.control:
            self._stop()
            shutil.rmtree(self.sdk, ignore_errors=True)
            _extract(archive_b64, self.sdk)

    def start(
        self,
        module: str | None = None,
        class_name: str | None = None,
        requirements: list[str] | None = None,
        generation: int | None = None,
    ) -> bool:
        """(Re)start the app process; False when no app was ever started here.

        With ``generation``, only restart if nothing started or stopped the app
        since that generation was read.
        """
        with self.control:
            if not (module or self.module):
                return False
            if generation is not None and generation != self.generation:
                return False
            # "starting" right away, so pollers never see a "stopped" app.
            self._stop(final="starting")
            with self.lock:
                if self.started_at is not None:
                    self.restarts += 1
                self.module = module or self.module
                self.class_name = class_name or self.class_name
                if requirements is not None:
                    self.requirements = requirements
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
            _, result = _child_call("reload", {"files": files}, timeout=300)
        except CHILD_ERRORS as exc:
            return {"action": "error", "reason": f"app not reachable: {exc}"}
        if result.get("action") == "restart":
            self.start(generation=generation)
        return result

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
                "restarts": self.restarts,
            }
        if result["phase"] == "ready":
            try:
                _, result["app"] = _child_call("status", {}, timeout=5)
            except CHILD_ERRORS as exc:
                result["app"] = {"error": str(exc)}
        return result


def _forward(payload: dict) -> dict:
    """Send one request to the resident app and wrap its response."""
    method = str(payload.get("method", "POST")).upper()
    if method not in HTTP_METHODS:
        raise ValueError(f"unsupported method {method!r}")
    path = str(payload.get("path", "/"))
    # The child decodes the path, so check the decoded form.
    decoded = urllib.parse.unquote(path)
    if not path.startswith("/") or decoded.startswith(CHILD_PREFIX):
        raise ValueError(f"invalid path {path!r}")
    body = payload.get("body")
    data = json.dumps(body).encode() if body is not None else None
    timeout = _timeout(payload)
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


def _forward_runs(payload: dict) -> dict:
    """Send the request ``warmups`` + ``repeat`` times, timing each one.

    Runs back to back on the box, so the times leave out the trip to fal.run.
    Only the last response's body is returned; a failed run ends the batch and
    returns its response instead. After the first run, each run's socket
    timeout is what is left of ``RUNS_BUDGET_SECONDS`` (a soft cap: a response
    that keeps trickling in can run past it). A run that does not fit, or fails
    to reach the app, ends the batch early with the runs so far (``stopped``
    says why).
    """
    warmups = _count(payload, "warmups", 0, 0, MAX_WARMUPS)
    repeat = _count(payload, "repeat", 1, 1, MAX_REPEAT)
    timeout = _timeout(payload)
    warmup_runs: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    result: dict[str, Any] = {}
    stopped = ""
    started = time.monotonic()
    for index in range(warmups + repeat):
        if index:
            left = RUNS_BUDGET_SECONDS - (time.monotonic() - started)
            if left <= 0:
                stopped = "the batch reached its time budget"
                break
            try:
                result = _forward({**payload, "timeout": min(timeout, left)})
            except CHILD_ERRORS as exc:
                stopped = f"run {index + 1} did not complete: {exc}"
                break
        else:
            result = _forward(payload)
        run = {
            "status_code": result["status_code"],
            "elapsed_seconds": result["elapsed_seconds"],
        }
        (warmup_runs if index < warmups else runs).append(run)
        if result["status_code"] >= 400:
            break
    return {
        **result,
        "warmup_runs": warmup_runs,
        "runs": runs,
        "stopped": stopped,
    }


def _count(payload: dict, key: str, default: int, minimum: int, maximum: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _timeout(payload: dict) -> float:
    try:
        return float(payload.get("timeout", 3600))
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout must be a number") from exc


def _endpoint_body(path: str) -> Any:
    """The default body the app's input model gives for ``path``."""
    _, found = _child_call("endpoints", {}, timeout=60)
    for entry in found.get("endpoints", []):
        if entry["path"] == path:
            if "error" in entry:
                raise ValueError(f"{path}: {entry['error']}; pass --data")
            return entry["body"]
    raise ValueError(f"{path} is not an endpoint that can be profiled")


def _profile(payload: dict, root: Path) -> dict:
    """Warm up, then send one request to ``path`` under the torch profiler.

    Warmups and the profiled request share ``RUNS_BUDGET_SECONDS``, so the
    call answers inside the box's request timeout.
    """
    path = str(payload.get("path", "/"))
    mode = str(payload.get("mode", "standard"))
    warmups = _count(payload, "warmups", 1, 0, MAX_PROFILE_WARMUPS)
    top = _count(payload, "top", 15, 1, 100)
    timeout = _timeout(payload)
    deadline = time.monotonic() + RUNS_BUDGET_SECONDS
    body = payload.get("body")
    if body is None:
        body = _endpoint_body(path)
    request = {"path": path, "body": body, "method": "POST"}

    def send() -> dict:
        left = deadline - time.monotonic()
        if left <= 0:
            raise ValueError("warmups used up the time budget; pass fewer --warmups")
        return _forward({**request, "timeout": min(timeout, left)})

    for _ in range(warmups):
        warm = send()
        if warm["status_code"] >= 400:
            failed = {k: warm[k] for k in ("status_code", "elapsed_seconds")}
            failed["body"] = warm.get("body", "")[-2000:]
            return {"status": "request_failed", "request": failed, "body": body}
    app_profile.prune(root, keep=PROFILES_KEPT - 1)
    profile_id = uuid.uuid4().hex
    spec = {
        "id": profile_id,
        "path": path,
        "mode": mode,
        "dir": str(root / profile_id),
        "top": top,
    }
    status, armed = _child_call("profile/arm", spec, timeout=30)
    if status != 200:
        raise ValueError(armed.get("detail", f"could not arm the profile ({status})"))
    try:
        response = send()
    finally:
        # Also disarms the profile when the request never reached the endpoint.
        _, result = _child_call("profile/result", {"id": profile_id}, timeout=30)
    result["request"] = {
        "status_code": response["status_code"],
        "elapsed_seconds": response["elapsed_seconds"],
    }
    if response["status_code"] >= 400:
        result["request"]["body"] = response.get("body", "")[-2000:]
    result["body"] = body
    result["id"] = profile_id
    return result


def _artifact(payload: dict, root: Path) -> dict:
    profile_id = str(payload.get("id", ""))
    name = str(payload.get("name", ""))
    if not profile_id.isalnum() or name not in ARTIFACTS:
        raise ValueError("unknown profile artifact")
    path = root / profile_id / name
    if not path.is_file():
        raise ValueError(f"{name} of profile {profile_id} is gone from the box")
    offset = _count(payload, "offset", 0, 0, path.stat().st_size)
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(ARTIFACT_CHUNK_BYTES)
    return {
        "size": path.stat().st_size,
        "offset": offset,
        "data": base64.b64encode(data).decode(),
    }


def _int(payload: dict, key: str, default: int | None) -> int | None:
    value = payload.get(key, default)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise fastapi.HTTPException(
            status_code=422, detail=f"{key} must be an integer"
        ) from exc


class DevAppHolder:
    """Mixed in ahead of the user's app class by ``make_holder``."""

    _fal_dev_token: ClassVar[str] = ""
    _fal_dev_session: ClassVar[str] = "default"

    def setup(self) -> None:
        self._child = _ChildProcess(STATE_ROOT / self._fal_dev_session)
        self._child.src.mkdir(parents=True, exist_ok=True)
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

    def _check(self, payload: Any) -> dict:
        if not isinstance(payload, dict):
            raise fastapi.HTTPException(
                status_code=422, detail="expected a JSON object"
            )
        token = str(payload.get(TOKEN_FIELD, "")).encode()
        expected = self._fal_dev_token.encode()
        if not expected or not hmac.compare_digest(token, expected):
            raise fastapi.HTTPException(
                status_code=403, detail="bad or missing dev token"
            )
        return payload

    @fal.endpoint(f"{ROUTE_PREFIX}status")
    async def dev_status(self, payload: Any = Body(...)) -> dict[str, Any]:
        self._check(payload)
        status = await asyncio.to_thread(self._child.status)
        status["session"] = self._fal_dev_session
        status["python"] = sys.version.split()[0]
        return status

    @fal.endpoint(f"{ROUTE_PREFIX}start")
    async def dev_start(self, payload: Any = Body(...)) -> dict[str, Any]:
        payload = self._check(payload)
        child = self._child

        def start() -> dict[str, Any]:
            if payload.get("sdk_archive"):
                child.replace_sdk(payload["sdk_archive"])
            started = child.start(
                payload.get("module"),
                payload.get("class_name"),
                payload.get("sdk_requirements"),
            )
            if not started:
                raise ValueError("no app has been started on this box")
            return child.status()

        try:
            return await asyncio.to_thread(start)
        except REQUEST_ERRORS as exc:
            raise fastapi.HTTPException(status_code=409, detail=str(exc)) from exc

    @fal.endpoint(f"{ROUTE_PREFIX}files")
    async def dev_files(self, payload: Any = Body(...)) -> dict[str, Any]:
        payload = self._check(payload)
        child = self._child

        def sync() -> dict[str, Any]:
            # Deletes first: a path can turn from a file into a folder or back.
            deleted = _delete(list(payload.get("deleted") or []), child.src)
            if payload.get("full"):
                keep = list(payload.get("all_files") or [])
                deleted += _prune(keep, child.synced, child.src)
                child.synced = set(keep)
            child.synced.difference_update(deleted)
            written = []
            if payload.get("archive"):
                written = _extract(payload["archive"], child.src)
            child.synced.update(written)
            result: dict[str, Any] = {"written": written, "deleted": deleted}
            if payload.get("reload"):
                result["reload"] = child.reload(written + deleted)
            return result

        try:
            return await asyncio.to_thread(sync)
        except REQUEST_ERRORS as exc:
            raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc

    @fal.endpoint(f"{ROUTE_PREFIX}request")
    async def dev_request(self, payload: Any = Body(...)) -> dict[str, Any]:
        payload = self._check(payload)
        phase = self._child.phase
        if phase != "ready":
            raise fastapi.HTTPException(
                status_code=409, detail=f"the app is not ready (phase: {phase})"
            )
        try:
            return await asyncio.to_thread(_forward_runs, payload)
        except CHILD_ERRORS as exc:
            raise fastapi.HTTPException(status_code=409, detail=str(exc)) from exc

    def _ready(self) -> None:
        phase = self._child.phase
        if phase != "ready":
            raise fastapi.HTTPException(
                status_code=409, detail=f"the app is not ready (phase: {phase})"
            )

    @fal.endpoint(f"{ROUTE_PREFIX}endpoints")
    async def dev_endpoints(self, payload: Any = Body(...)) -> dict[str, Any]:
        self._check(payload)
        self._ready()
        try:
            _, found = await asyncio.to_thread(_child_call, "endpoints", {}, 60)
        except CHILD_ERRORS as exc:
            raise fastapi.HTTPException(status_code=409, detail=str(exc)) from exc
        return found

    @fal.endpoint(f"{ROUTE_PREFIX}profile")
    async def dev_profile(self, payload: Any = Body(...)) -> dict[str, Any]:
        payload = self._check(payload)
        self._ready()
        root = self._child.session_dir / "profiles"

        def profile() -> dict[str, Any]:
            # One profile at a time: arming replaces any armed profile.
            if not self._child.profile_lock.acquire(blocking=False):
                raise ValueError("another profile is running on this box")
            try:
                return _profile(payload, root)
            finally:
                self._child.profile_lock.release()

        try:
            return await asyncio.to_thread(profile)
        except CHILD_ERRORS as exc:
            raise fastapi.HTTPException(status_code=409, detail=str(exc)) from exc

    @fal.endpoint(f"{ROUTE_PREFIX}artifact")
    async def dev_artifact(self, payload: Any = Body(...)) -> dict[str, Any]:
        payload = self._check(payload)
        root = self._child.session_dir / "profiles"
        try:
            return await asyncio.to_thread(_artifact, payload, root)
        except CHILD_ERRORS as exc:
            raise fastapi.HTTPException(status_code=404, detail=str(exc)) from exc

    @fal.endpoint(f"{ROUTE_PREFIX}logs")
    async def dev_logs(self, payload: Any = Body(...)) -> dict[str, Any]:
        payload = self._check(payload)
        return self._child.log_since(
            _int(payload, "after", None), _int(payload, "lines", 200) or 0
        )


def make_holder(
    target: type[App],
    *,
    app_name: str,
    session: str,
    token: str,
    keep_alive: int,
    machine_type: str | None = None,
) -> type[App]:
    """A box class that inherits ``target``'s environment but not its setup()."""
    namespace: dict[str, Any] = {
        "__module__": __name__,
        "__qualname__": f"FalDev{target.__name__}",
        "_fal_dev_token": token,
        "_fal_dev_session": session,
        # App.__init_subclass__ would otherwise keep the target's name.
        "app_name": None,
        "app_auth": "private",
        # Class attributes win over the target's scaling, header kwargs included.
        "keep_alive": keep_alive,
        "min_concurrency": 0,
        "max_concurrency": 1,
        "concurrency_buffer": 0,
        "concurrency_buffer_perc": 0,
        "max_multiplexing": 8,
        "request_timeout": 3600,
    }
    if machine_type:
        namespace["machine_type"] = machine_type
    return type(
        f"FalDev{target.__name__}",
        (DevAppHolder, target),
        namespace,
        name=app_name,
    )
