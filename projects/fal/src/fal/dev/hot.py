"""`fal dev hot`: keep a model loaded on the box and reload the code around it.

A project folder holds load.py (`load() -> state`, run once per server process)
and infer.py (`infer(state, **params)`, reloaded on every save). The server
(hot_server.py) runs on the box; this module syncs files to it and sends it
requests through the box's /hot route on fal.run.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from fal.dev import config, files, state
from fal.dev.box import Box, log
from fal.dev.errors import DevError
from fal.dev.loading import STATES, Loading
from fal.dev.transport import post_json

OUTPUT_DIR = "hot-out"
MAX_FILE_BYTES = 1_000_000
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{20,}")
# Log lines from the box that mean the server waits for a fix instead of loading.
STUCK_MARKERS = (
    "load.load() failed",
    "installing requirements.txt failed",
    "server exited with code",
)
RETRY_MARKERS = ("[hot] loading the model", "[hot] restarting", "[hot] FULL RESTART")
READY_MARKER = "[hot] serving"
# tqdm progress bars (weights, pipeline parts, denoising steps) in the box log.
PROGRESS = re.compile(r"\d+%\|.*\||\d+(?:\.\d+)?(?:it/s|s/it)\]")


class HotTokenRejected(DevError):
    """The box no longer accepts our token: it restarted on a new runner."""


@dataclass
class RunResult:
    """One /hot/infer answer: an image or JSON on 200, else the error."""

    status: int
    headers: dict
    body: bytes
    seconds: float
    restarted: str | None = None  # why the weights reloaded first, if they did
    path: Path | None = None  # where an image result was saved

    @property
    def ok(self) -> bool:
        return self.status == 200

    @property
    def is_image(self) -> bool:
        return self.headers.get("content-type", "").startswith("image/")

    @property
    def model_loaded(self) -> bool:
        """422 is the code failing while the server (and the weights) stay up."""
        return self.status == 422

    @property
    def error(self) -> dict:
        """{"message", "traceback"} for a failed request."""
        try:
            info = json.loads(self.body)
            return {"message": info.get("error"), "traceback": info.get("traceback")}
        except (ValueError, AttributeError):
            return {"message": repr(self.body[:500]), "traceback": None}


def remote_dir(local: str | None, remote: str | None) -> str:
    """The project folder on the box: --dir, else /work/<local folder name>."""
    if remote:
        return remote
    if local:
        return f"{config.LOCAL_WORK}/{Path(local).resolve().name}"
    raise DevError("pass a local project folder or --dir <folder on the box>")


def new_log_lines(printed: list[str], lines: list[str]) -> list[str]:
    """The lines of a log tail not printed yet, given the tail printed last time."""
    for overlap in range(min(len(printed), len(lines)), 0, -1):
        if printed[-overlap:] == lines[:overlap]:
            return lines[overlap:]
    return lines


def _last(lines: list[str], markers: tuple[str, ...]) -> int:
    return max(
        (i for i, line in enumerate(lines) if any(m in line for m in markers)),
        default=-1,
    )


def attempt_lines(lines: list[str]) -> list[str]:
    """The log from the newest load attempt on: earlier runs were shown already."""
    return lines[max(0, _last(lines, RETRY_MARKERS)) :]


def waiting_for_fix(lines: list[str]) -> bool:
    """True if the box log ends in a failure the server waits on, not a new attempt."""
    return _last(lines, STUCK_MARKERS) > _last(lines, RETRY_MARKERS)


def restarting(lines: list[str]) -> bool:
    """True if the box log shows a restart under way (not a stop or a failure)."""
    retried = _last(lines, RETRY_MARKERS)
    return retried > max(_last(lines, STUCK_MARKERS), _last(lines, (READY_MARKER,)))


def project_paths(
    local: str | None, remote: str | None, remembered: dict
) -> tuple[str | None, str]:
    """The local project folder (absolute) and its folder on the box.

    Without a local folder, the one last used here. The remembered folder on the
    box only goes with the remembered local project.
    """
    local = str(Path(local).resolve()) if local else remembered.get("hot_local")
    if not remote and local and local == remembered.get("hot_local"):
        remote = remembered.get("hot_dir")
    return local, remote_dir(local, remote)


def scan(root: Path, withheld: list[Path] | None = None) -> dict[str, float]:
    """Modification time of every small file under root that is worth syncing.

    Secret-like files, which are never synced, are appended to ``withheld``.
    """
    found = {}
    skip_dirs = files.SKIP_DIRS | {OUTPUT_DIR}
    for rel in files.walk(root, skip_dirs=skip_dirs, hidden=False, withheld=withheld):
        try:
            stat = (root / rel).stat()
        except OSError:
            continue
        if stat.st_size <= MAX_FILE_BYTES:
            found[rel.as_posix()] = stat.st_mtime
    return found


def parse_params(items: list[str] | None, prompt: str | None) -> dict:
    """infer() parameters from key=value items (JSON values, else strings)."""
    params = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise DevError(f"--param takes key=value, got {item!r}")
        try:
            params[key] = json.loads(value)
        except ValueError:
            params[key] = value
    if prompt:
        params["prompt"] = prompt
    return params


def _owner(box: Box) -> str:
    """The account the box's app belongs to, for its fal.run URL."""
    from fal.api.deploy import _get_user

    if box.account.team:
        return box.account.team
    return _get_user(box.client._create_rest_client()).username


def read_token(box: Box, runner_id: str) -> str:
    """This runner's /hot token, read once over runner exec (admin only) and cached."""
    cache = state.STATE_DIR / f"hot-{box.app}.json"
    try:
        cached = json.loads(cache.read_text())
        if cached["runner_id"] == runner_id:
            return cached["token"]
    except (OSError, ValueError, KeyError):
        pass
    # The trailing sleep keeps runner exec from dropping the output at exit.
    command = f"cat {config.HOT_TOKEN_FILE}; sleep 1"
    code, out = box.run(runner_id, ["sh", "-c", command])
    token = out.strip().splitlines()[-1] if out.strip() else ""
    if code != 0 and "No such file" not in out:
        raise DevError(f"could not read the box's hot token: {out.strip() or code}")
    if not TOKEN_PATTERN.fullmatch(token):
        raise DevError(
            "this box has no hot reload support yet; redeploy it with: fal dev deploy"
        )
    with contextlib.suppress(OSError):  # the cache is only a speedup
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.touch(mode=0o600)
        cache.chmod(0o600)
        cache.write_text(json.dumps({"runner_id": runner_id, "token": token}))
    return token


class HotBox:
    """Sends requests to the box's /hot routes through fal.run, signed in as you."""

    # Called with (saved image path, response headers) after each image result.
    on_image: Callable[[Path, dict], None] | None = None

    def __init__(self, box: Box) -> None:
        from fal import flags

        runner_id = box.require_runner()
        self.base = f"https://{flags.FAL_RUN_HOST}/{_owner(box)}/{box.app}/hot"
        self.box = box
        self.token = read_token(box, runner_id)

    def post(
        self, action: str, payload: dict | None = None, timeout: int = 60
    ) -> tuple[int, dict, bytes]:
        """POST to a /hot route; status 0 when there is no response (network, auth)."""
        try:
            response = post_json(
                f"{self.base}/{action}",
                payload or {},
                {**self.box.auth_headers(), config.HOT_TOKEN_HEADER: self.token},
                timeout,
            )
        except Exception as exc:
            return 0, {}, json.dumps({"error": str(exc) or repr(exc)}).encode()
        headers = {k.lower(): v for k, v in response.headers.items()}
        return response.status_code, headers, response.content

    def json(self, action: str, payload: dict | None = None) -> tuple[int, dict]:
        status, _, body = self.post(action, payload)
        try:
            return status, json.loads(body)
        except ValueError:
            return status, {"error": body.decode(errors="replace")[:500]}

    def wait_ready(
        self,
        timeout: int = 1800,
        headline: str = "starting the hot server: loading weights",
    ) -> dict:
        """Poll until the server has loaded the model, printing its log as it goes."""
        with Loading(
            "model load",
            title="Hot server",
            headline=headline,
            expected=STATES["weights"][1],
        ) as wait:
            info = self._wait_ready(timeout, wait)
            wait.done()
            return info

    def _wait_ready(self, timeout: int, wait) -> dict:
        deadline = time.time() + timeout
        printed: list[str] | None = None
        network_errors = 0
        while time.time() < deadline:
            status, info = self.json("status")
            if status == 0:
                # No response (network); the server keeps loading meanwhile.
                network_errors += 1
                if network_errors > 5:
                    raise DevError(
                        f"no answer from the box for 30s: {info.get('error')}",
                        title="lost contact with the box",
                        fixes=[
                            ("fal dev status", "is the box still up?"),
                            ("fal dev hot status", "the server keeps loading"),
                        ],
                    )
                time.sleep(5)
                continue
            network_errors = 0
            lines = info.get("log") or []
            stuck = status == 409 and waiting_for_fix(lines)
            if printed is None:
                # Earlier requests and attempts were shown by earlier commands.
                printed = lines[: len(lines) - len(attempt_lines(lines))]
            # Progress bars only when the load failed.
            for line in new_log_lines(printed, lines):
                if stuck or not PROGRESS.search(line):
                    wait.say(f"  {line}")
            printed = lines
            if stuck:
                raise DevError(
                    "the hot server stopped on an error; its log is above",
                    title="the hot server stopped",
                    fixes=[("fal dev hot start", "after you fix load.py")],
                )
            if status == 200 and info.get("ready") and not info.get("restarting"):
                return info
            # 409: the server is not up yet; 502-504: the gateway has no answer.
            if status not in (200, 409, 502, 503, 504):
                raise DevError(f"hot server error ({status}): {info.get('error')}")
            time.sleep(2)
        raise DevError(
            f"the hot server was not ready within {timeout}s",
            title="the model did not load",
            fixes=[("fal dev hot logs", "see what it is doing")],
        )

    def push(self, root: Path, remote: str, rels: list[str]) -> list[str]:
        contents = {}
        for rel in rels:
            try:
                if (root / rel).is_symlink():
                    continue  # became a link since the scan
                contents[rel] = (root / rel).read_text()
            except (OSError, UnicodeDecodeError):
                continue  # deleted since the scan, or binary
        if not contents:
            return []
        status, info = self.json("files", {"dir": remote, "files": contents})
        if status == 403:
            raise HotTokenRejected(
                "the box restarted since this command began",
                title="the box restarted",
                fixes=[
                    ("fal dev up", "if it is not running"),
                    ("fal dev hot start", "load the model again"),
                ],
            )
        if status != 200:
            raise DevError(f"sync failed ({status}): {info.get('error')}")
        return info.get("written", [])

    def run(
        self,
        params: dict,
        output: str | None = None,
        on_restart: Callable[[str], None] | None = None,
    ) -> RunResult:
        """Send one request, riding out a restart; save an image result."""
        start = time.time()
        status, headers, body = self.post("infer", params, timeout=900)
        reason = None
        if status == 409:
            try:
                info = json.loads(body)
                # The server says so, or has already exited to restart (port closed).
                reason = info.get("restarting") or (
                    restarting(info.get("log") or []) and "a restart is under way"
                )
            except (ValueError, AttributeError):
                pass
        if reason:
            if on_restart:
                on_restart(reason)
            else:
                log(f"FULL RESTART ({reason}): the weights reload ...")
            info = self.wait_ready(headline=STATES["hot-restart"][0])
            if not on_restart:
                log(f"model loaded in {info['load_seconds']}s")
            start = time.time()
            status, headers, body = self.post("infer", params, timeout=900)
        result = RunResult(status, headers, body, time.time() - start, reason or None)
        if result.ok and result.is_image:
            name = f"{time.strftime('%H%M%S')}.png"
            result.path = Path(output) if output else Path(OUTPUT_DIR) / name
            result.path.parent.mkdir(parents=True, exist_ok=True)
            result.path.write_bytes(body)
        return result

    def request(
        self, params: dict, output: str | None = None, edited_at: float | None = None
    ) -> bool:
        """Send one request; save an image result and print timings. True on success."""
        result = self.run(params, output)
        if not result.ok:
            error = result.error
            if error.get("traceback"):
                log(error["traceback"])
            log(f"error ({result.status}): {error['message']}")
            return False
        headers = result.headers
        timing = (
            f"code v{headers.get('x-hot-code-version')} | "
            f"infer {headers.get('x-hot-infer-ms')} ms | "
            f"round trip {result.seconds:.2f}s"
        )
        if edited_at:
            timing += f" | edit to result {time.time() - edited_at:.2f}s"
        if result.path:
            print(f"{timing} | {result.path}")
            if self.on_image:
                self.on_image(result.path, headers)
        else:
            print(timing)
            print(result.body.decode(errors="replace"))
        return True

    def watch(
        self,
        root: Path,
        remote: str,
        run_params: dict | None,
        output: str | None,
        view: WatchView,
    ) -> None:
        """Push each saved file; with run_params, send a request after each push."""
        known = scan(root)
        pending: set[str] = set()  # failed pushes, sent again with the next save
        view.watching(root, run_params is not None)
        while True:
            time.sleep(0.2)
            current = scan(root)
            changed = sorted(p for p in current if current[p] != known.get(p))
            if not changed:
                continue
            known = current
            edited_at = max(current[p] for p in changed)
            try:
                written = self.push(root, remote, sorted(pending | set(changed)))
            except HotTokenRejected:
                raise  # not a failed push: the box needs a new command
            except DevError as exc:
                view.sync_failed(changed, str(exc))
                pending.update(changed)
                continue
            pending.clear()
            if not written:
                continue  # saved without changes
            view.synced(written, time.time() - edited_at)
            if run_params is not None:
                try:
                    result = self.run(run_params, output, on_restart=view.restarting)
                    view.ran(result, edited_at)
                except Exception as exc:  # keep watching; the next save retries
                    view.lost(str(exc) or repr(exc))


class WatchView(Protocol):
    """What `hot sync --watch` reports as you save (see fal.dev.cli.watch_view)."""

    def watching(self, root: Path, run: bool) -> None: ...
    def synced(self, files: list[str], after_save: float) -> None: ...
    def sync_failed(self, files: list[str], error: str) -> None: ...
    def restarting(self, reason: str) -> None: ...
    def ran(self, result: RunResult, edited_at: float) -> None: ...
    def lost(self, error: str) -> None: ...
