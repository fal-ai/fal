"""Talking to a running `fal dev app` box: sync, boot, wait, request, download."""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import re
import time
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path
from typing import Any

from rich.markup import escape

from fal.console import console
from fal.dev.app import sync
from fal.dev.app.session import Session, call
from fal.dev.errors import DevError
from fal.dev.remote.protocol import (
    MAX_FILE_BYTES,
    MAX_PLACE_FILES,
    REQUEST_TIMEOUT_SECONDS,
    STOPPED_PHASES,
    redact,
)

POLL_SECONDS = 1.0
# Calls that copy files or wait for a runner, and `start`'s own call.
SLOW_TIMEOUT = 600
START_TIMEOUT = 300
# How long setup() may take.
SETUP_TIMEOUT = 3600
# A little past the box's own request timeout, so its answer arrives.
REQUEST_CALL_TIMEOUT = REQUEST_TIMEOUT_SECONDS + 60
# The box copies at most this much per `place` call, inside SLOW_TIMEOUT.
PLACE_BATCH_BYTES = 512 * 1024 * 1024
LOG_LINE_WIDTH = 400
# An incremental sync asks the object store only when it is big enough to be
# worth the extra round trip; an edit is uploaded directly.
PLACE_MIN_FILES = 20
PLACE_MIN_BYTES = 256 * 1024


# uv's text for a package index that refused the box's credentials. Only uv
# prints this, so other 401s (a model hub, the fal API) do not match.
AUTH_REFUSED = re.compile(r"HTTP status client error \((401|403)\b")
INDEX_AUTH_HINT = (
    "If this is fal's private package index, the team needs the "
    "`fal_wheels_access` grant; run with `--team <team that has it>`."
)


def index_hint(message: str) -> str:
    """The message, with what to do about it if a package index said no."""
    return f"{message}\n{INDEX_AUTH_HINT}" if AUTH_REFUSED.search(message) else message


@contextlib.contextmanager
def explain_index_refusal() -> Iterator[None]:
    """Add what to do about it to an error from a package index that said no."""
    try:
        yield
    except Exception as exc:
        if AUTH_REFUSED.search(str(exc)):
            raise DevError(index_hint(str(exc))) from exc
        raise


@dataclasses.dataclass
class SyncResult:
    changed: list[str] = dataclasses.field(default_factory=list)
    deleted: list[str] = dataclasses.field(default_factory=list)
    # Of the changed files, how many the box copied from the team's object store.
    placed: int = 0
    # The box's answer to the reload that ended the sync (empty if none was asked).
    reload: dict[str, Any] = dataclasses.field(default_factory=dict)


def print_log_lines(lines: list[str]) -> None:
    for line in lines:
        cut = f"{line[:LOG_LINE_WIDTH]} ..." if len(line) > LOG_LINE_WIDTH else line
        print(cut, flush=True)


def logs(
    session: Session, *, lines: int | None = None, after: int | None = None
) -> dict[str, Any]:
    """Log lines from the box: the last ``lines``, or those after ``after``."""
    payload = {"lines": lines} if after is None else {"after": after}
    return call(session, "logs", payload)


def status(session: Session) -> dict[str, Any]:
    return call(session, "status")


def log_position(session: Session) -> int:
    """The number of log lines the box has so far, to read what comes after."""
    return call(session, "logs", {"lines": 0}, timeout=SLOW_TIMEOUT)["total"]


def wait_ready(
    session: Session, *, after: int | None = None, timeout: float = SETUP_TIMEOUT
) -> dict[str, Any]:
    """Stream the app's log until it is ready, or raise if it stops."""
    deadline = time.monotonic() + timeout
    if after is None:
        after = log_position(session)
    while time.monotonic() < deadline:
        new = call(session, "logs", {"after": after})
        print_log_lines(new["lines"])
        after = new["total"]
        state = call(session, "status")
        phase = state["phase"]
        if phase == "ready":
            return state
        if phase in STOPPED_PHASES:
            raise DevError(
                f"The app did not start ({phase} {state['detail']}). "
                "Fix the code and run `fal dev app restart`."
            )
        time.sleep(POLL_SECONDS)
    raise DevError("Timed out waiting for setup() to finish.")


def collect(
    session: Session,
    skipped: list[str] | None = None,
    withheld: list[str] | None = None,
) -> dict[str, Path]:
    """The session's files to sync, by path relative to its root."""
    return sync.collect(
        Path(session.root),
        skipped,
        ignore=session.ignore,
        ignore_base=session.ignore_base,
        withheld=withheld,
    )


def tree_signature(session: Session) -> dict[str, tuple[int, int]]:
    """Cheap change detection for `sync --watch`: (mtime, size) per file."""
    return sync.stat_signature(collect(session))


def _worth_placing(files: dict[str, Path], names: list[str]) -> bool:
    if len(names) > PLACE_MIN_FILES:
        return True
    total = 0
    for rel in names:
        try:
            total += files[rel].stat().st_size
        except OSError:
            continue
    return total > PLACE_MIN_BYTES


def place_from_store(
    session: Session,
    files: dict[str, Path],
    manifest: dict[str, str],
    names: list[str],
) -> set[str]:
    """Have the box copy the named files its team's object store already holds.

    Returns the paths placed; the rest must be uploaded.
    """
    entries: dict[str, list[Any]] = {}
    for rel in names:
        try:
            entries[rel] = [manifest[rel], bool(files[rel].stat().st_mode & 0o111)]
        except OSError:
            continue  # gone since collect; the sync sorts it out
    placed: set[str] = set()
    groups = sync.batches(files, sorted(entries), PLACE_BATCH_BYTES, MAX_PLACE_FILES)
    for group in filter(None, groups):
        try:
            result = call(
                session,
                "place",
                {"files": {rel: entries[rel] for rel in group}},
                timeout=SLOW_TIMEOUT,
            )
        except DevError as exc:
            # Whatever was not placed is uploaded instead.
            console.print(f"[yellow]Object store place failed[/]: {escape(str(exc))}")
            continue
        placed.update(result["placed"])
    return placed


def sync_files(
    session: Session, *, full: bool = False, reload: bool = True
) -> SyncResult:
    """Send the box what changed since the last sync (everything if ``full``).

    Files the team's object store already holds are copied on the box; the
    rest are uploaded. With ``reload``, the box then reloads the app.
    """
    skipped: list[str] = []
    withheld: list[str] = []
    files = collect(session, skipped, withheld)
    if skipped:
        console.print(
            f"[yellow]Not synced (over {MAX_FILE_BYTES >> 20} MiB)[/]: "
            f"{escape(', '.join(skipped[:8]))}"
        )
    if withheld and full:
        console.print(
            "[yellow]Not synced (key or credential files)[/]: "
            f"{escape(', '.join(withheld[:8]))}"
        )
    manifest = sync.manifest(files)
    changed, deleted = sync.diff({} if full else session.manifest, manifest)
    if not changed and not deleted and not full:
        return SyncResult()
    placed: set[str] = set()
    if full or _worth_placing(files, changed):
        placed = place_from_store(session, files, manifest, changed)
    uploads = [rel for rel in changed if rel not in placed]
    # Large trees go in several requests. Deletes and pruning ride on the
    # first, before any writes; the reload on the last.
    groups = sync.batches(files, uploads)
    answer: dict[str, Any] = {}
    for index, group in enumerate(groups):
        payload: dict[str, Any] = {"reload": reload and index == len(groups) - 1}
        if group:
            payload["archive"] = sync.pack(files, group)
        if index == 0:
            payload["deleted"] = deleted
            if full:
                payload["all_files"] = sorted(files)
        answer = call(session, "files", payload, timeout=SLOW_TIMEOUT)
    reloaded = answer.get("reload") or {}
    if reloaded.get("action") not in ("rebuild", "error", "not_running"):
        # Otherwise keep the old manifest, so the next sync sends the files
        # again and repeats the warning.
        session.manifest = manifest
        session.save()
    return SyncResult(changed, deleted, len(placed), reloaded)


def _sdk_payload() -> dict[str, Any]:
    """The local fal package, which the runner does not have installed, and the
    dependencies of it to install there if missing."""
    package = Path(__file__).resolve().parents[2]
    sdk = {
        (Path("fal") / path.relative_to(package)).as_posix(): path
        for path in package.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    try:
        requires = metadata.requires("fal") or []
    except metadata.PackageNotFoundError:
        requires = []
    return {
        "sdk_archive": sync.pack(sdk, sorted(sdk)),
        "sdk_requirements": [r for r in requires if "extra ==" not in r],
    }


def boot(session: Session) -> dict[str, Any]:
    """Send the whole source tree and the SDK, then run setup() on the box."""
    started = time.perf_counter()
    result = sync_files(session, full=True, reload=False)
    elapsed = time.perf_counter() - started
    copied = (
        f", {result.placed} copied from the team's object store on the box"
        if result.placed
        else ""
    )
    uploaded = len(result.changed) - result.placed
    console.print(f"Source: {uploaded} file(s) uploaded{copied} ({elapsed:.1f}s).")
    replay_overlay(session)
    after = log_position(session)
    call(
        session,
        "start",
        {
            "module": session.module,
            "class_name": session.class_name,
            **_sdk_payload(),
            "layout": {"paths": session.paths, "workdir": session.workdir},
            "session_env": session.session_env,
        },
        timeout=START_TIMEOUT,
    )
    console.print(f"Running {session.class_name}.setup() on the box ...")
    return wait_ready(session, after=after)


def ensure_booted(session: Session) -> bool:
    """Boot again when the box was scaled to zero or replaced since `start`."""
    # A cold box first waits for a runner, which can take minutes.
    state = call(session, "status", timeout=SLOW_TIMEOUT)
    if state["phase"] != "idle" or state["module"]:
        return False
    console.print(
        "[yellow]The box is on a new runner[/] (idle shutdown or replacement): "
        "syncing the source and running setup() again."
    )
    boot(session)
    return True


def restart(session: Session, *, sync_first: bool) -> None:
    """Restart the app process, so setup() runs again."""
    if ensure_booted(session):
        return
    if sync_first:
        sync_files(session, reload=False)
    after = log_position(session)
    call(session, "start", {}, timeout=START_TIMEOUT)
    wait_ready(session, after=after)


def deps_call(session: Session, payload: dict[str, Any]) -> dict[str, Any]:
    """Call the box's deps route; a change is recorded in the session.

    The recorded overlay is what a new runner installs, so only a change the
    box has made replaces it, never a status read.
    """
    result = call(session, "deps", payload, timeout=REQUEST_TIMEOUT_SECONDS)
    if payload["action"] != "status":
        session.overlay = result["replay"]
        session.save()
    return result


def replay_overlay(session: Session) -> None:
    """Install the session's dependency overlay on a fresh runner, as recorded.

    If it no longer installs, it is kept as recorded and tried again on the
    next boot, and the box starts without it.
    """
    wanted = session.overlay
    if not wanted.get("packages"):
        return
    if call(session, "deps", {"action": "status"})["replay"] == wanted:
        return
    console.print(f"Installing {len(wanted['packages'])} overlay package(s) ...")
    payload = {
        "packages": wanted["packages"],
        "pre": wanted.get("pre"),
        "index_url": wanted.get("index_url"),
        "extra_index_urls": wanted.get("extra_index_urls", []),
        "no_deps": True,
        "force": True,
    }
    try:
        # Start from a clean overlay: this runner may hold a partial earlier replay.
        call(session, "deps", {"action": "reset"}, timeout=REQUEST_TIMEOUT_SECONDS)
        call(
            session,
            "deps",
            {"action": "install", **payload},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except DevError as exc:
        console.print(
            "[red]Could not install the dependency overlay[/] (it is kept for the "
            f"next boot): {escape(index_hint(redact(str(exc))))}"
        )


def send_request(
    session: Session, path: str, body: Any, *, warmups: int = 0, repeat: int = 1
) -> dict[str, Any]:
    """Send ``path`` to the app on the box, ``warmups`` + ``repeat`` times."""
    return call(
        session,
        "request",
        {
            "path": path,
            "body": body,
            "method": "POST",
            "warmups": warmups,
            "repeat": repeat,
        },
        timeout=REQUEST_CALL_TIMEOUT,
    )


def run_profile(
    session: Session, path: str, body: Any, *, mode: str, warmups: int, top: int
) -> dict[str, Any]:
    """Run one request to ``path`` under the profiler on the box."""
    return call(
        session,
        "profile",
        {"path": path, "body": body, "mode": mode, "warmups": warmups, "top": top},
        timeout=REQUEST_CALL_TIMEOUT,
    )


def download(session: Session, profile_id: str, name: str, target: Path) -> int:
    """Copy one profile file from the box in chunks; returns its size."""
    tmp = target.with_name(f".{target.name}.part")
    offset, size = 0, None
    try:
        with tmp.open("wb") as handle:
            while size is None or offset < size:
                chunk = call(
                    session,
                    "artifact",
                    {"id": profile_id, "name": name, "offset": offset},
                    timeout=SLOW_TIMEOUT,
                )
                size = chunk["size"]
                data = base64.b64decode(chunk["data"])
                if not data and offset < size:
                    raise DevError(f"{name} stopped at {offset} of {size} bytes")
                handle.write(data)
                offset += len(data)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return size
