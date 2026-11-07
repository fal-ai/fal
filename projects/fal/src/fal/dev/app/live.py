"""What the live dashboard shows beyond the box itself: a box that is still
starting, and the hot reload of `sync`.

Two small files per box live next to its session file, with suffixes the
session glob (``*.json``) never matches:

- ``<name>.phase``: written by `start` while the box is deploying, getting a
  runner, or running setup(); removed once it is ready or start failed.
- ``<name>.sync``: written by `sync` after every sync, with the pid of a
  running `sync --watch` so the dash can say whether one is watching.

Both are advisory: a missing or unreadable file means "unknown", never an error.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from typing import Any

from fal.dev.app import session as sessions

DEPLOYING, PROVISIONING, SETUP = "deploying", "provisioning", "setup"
PHASES = (DEPLOYING, PROVISIONING, SETUP)
# A start that died without cleaning up must not show as starting forever.
PHASE_MAX_AGE = 30 * 60


def _path(name: str, kind: str):
    return sessions.SESSIONS_DIR / f"{name}.{kind}"


def _write(name: str, kind: str, data: dict[str, Any]) -> None:
    try:
        sessions.SESSIONS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = _path(name, kind)
        tmp = path.with_suffix(f".{kind}.tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)
    except OSError:
        pass


def _read(name: str, kind: str) -> dict[str, Any]:
    try:
        data = json.loads(_path(name, kind).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _alive(pid: Any, *, unknown: bool = True) -> bool:
    """Is the process running? ``unknown`` is the answer where it cannot be asked."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        # os.kill terminates the process there.
        return unknown
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # exists, owned by someone else
    return True


def set_phase(name: str, phase: str) -> None:
    previous = _read(name, "phase")
    owner = previous.get("pid")
    if owner not in (None, os.getpid()) and _alive(owner, unknown=False):
        # Another `start` owns the note; a second one (which may be refused)
        # must neither take it over nor clear it.
        return
    since = previous.get("since") if previous.get("phase") == phase else None
    _write(
        name,
        "phase",
        {"phase": phase, "since": since or time.time(), "pid": os.getpid()},
    )


def clear_phase(name: str) -> None:
    """Remove this process's phase note; another start's note is left alone
    (a refused second `start` must not erase the first one's)."""
    owner = _read(name, "phase").get("pid")
    if owner in (None, os.getpid()):
        _path(name, "phase").unlink(missing_ok=True)


@contextlib.contextmanager
def clearing_on_error(name: str):
    """Drop the phase note if the block raises (a failed deploy)."""
    try:
        yield
    except BaseException:
        clear_phase(name)
        raise


def starting() -> list[dict[str, Any]]:
    """Boxes being started right now: ``{name, phase, since}``, newest first."""
    found = []
    now = time.time()
    try:
        paths = list(sessions.SESSIONS_DIR.glob("*.phase"))
    except OSError:
        return []
    for path in paths:
        data = _read(path.stem, "phase")
        if data.get("phase") not in PHASES or not isinstance(
            data.get("since"), (int, float)
        ):
            continue
        if now - data["since"] > PHASE_MAX_AGE or not _alive(data.get("pid")):
            continue
        found.append(
            {"name": path.stem, "phase": data["phase"], "since": data["since"]}
        )
    return sorted(found, key=lambda item: -item["since"])


def record_sync(
    name: str, *, files: int, action: str | None, ms: float | None, seconds: float
) -> None:
    """Remember the last sync that sent something (or a reload that failed)."""
    data = _read(name, "sync")
    data["last"] = {
        "at": time.time(),
        "files": files,
        "action": action,
        "ms": ms,
        "seconds": seconds,
    }
    _write(name, "sync", data)


def set_watching(name: str, watching: bool) -> None:
    data = _read(name, "sync")
    if watching:
        data["watch_pid"] = os.getpid()
    else:
        # Only the watcher that wrote the pid may clear it.
        if data.get("watch_pid") != os.getpid():
            return
        data.pop("watch_pid", None)
    _write(name, "sync", data)


def clear_sync(name: str) -> None:
    """Forget the last sync: a box started under this name is a new box."""
    _path(name, "sync").unlink(missing_ok=True)


def read_sync(name: str) -> dict[str, Any]:
    """``{"watching": bool, "last": {...} | None}``."""
    data = _read(name, "sync")
    last = data.get("last")
    return {
        "watching": _alive(data.get("watch_pid"), unknown=False),
        "last": last if isinstance(last, dict) else None,
    }
