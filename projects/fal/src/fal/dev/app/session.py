"""Local state and transport for `fal dev app` sessions.

A session is one dev box for one app. Its state (box URL, token, the manifest of
the last sync) lives in ``~/.fal/dev/apps/<name>.json``. The box is a private
app deployed under the account it was started as; requests go to its
``/__fal_dev/*`` routes on fal.run.
"""

from __future__ import annotations

import dataclasses
import getpass
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx

from fal.dev.account import Account
from fal.dev.errors import DevError
from fal.dev.remote.protocol import ROUTE_PREFIX, TOKEN_FIELD
from fal.dev.state import STATE_DIR
from fal.dev.transport import BoxUnreachable, post_json

SESSIONS_DIR = STATE_DIR / "apps"
# Every box is named with this prefix, so `start` and `stop` can only ever
# replace or delete dev boxes, never a real app.
NAME_PREFIX = "dev-"
GATEWAY_ERRORS = (502, 503, 504)


class NoSession(DevError):
    """No dev app session exists (yet): a waiting state for the dash."""


@dataclasses.dataclass
class Session:
    name: str
    file_path: str
    root: str
    module: str
    class_name: str
    run_url: str
    token: str
    account: Account
    host: str | None = None
    environment: str | None = None
    machine_type: str | None = None
    # Where the app runs inside the synced tree, as on a deployed runner:
    # sys.path entries and the working directory, relative to ``root``.
    paths: list[str] = dataclasses.field(default_factory=lambda: ["."])
    workdir: str = "."
    # The app's ``app_files_ignore`` patterns, matched against paths relative
    # to ``ignore_base`` (its app files context folder) like `fal deploy` does.
    ignore: list[str] = dataclasses.field(default_factory=list)
    ignore_base: str = "."
    # Extra environment variable names the app process gets the box name in.
    session_env: list[str] = dataclasses.field(default_factory=list)
    manifest: dict[str, str] = dataclasses.field(default_factory=dict)
    # Files the last sync withheld as key or credential files, to announce a change.
    withheld: list[str] = dataclasses.field(default_factory=list)
    # The box's dependency overlay, installed again on a new runner.
    overlay: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.account, dict):
            self.account = Account(**self.account)

    @property
    def path(self) -> Path:
        return SESSIONS_DIR / f"{self.name}.json"

    def save(self) -> None:
        # The file holds the box token: create it private, never chmod after.
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(dataclasses.asdict(self), handle, indent=2)
        tmp.replace(self.path)

    def delete(self) -> None:
        self.path.unlink(missing_ok=True)
        # The phase and sync notes of a box that is gone (see app/live.py).
        for kind in ("phase", "sync"):
            self.path.with_suffix(f".{kind}").unlink(missing_ok=True)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-") or "app"


def session_name(name: str | None, app_name: str) -> str:
    """A box name: the given one, or app and user, always under NAME_PREFIX."""
    name = slug(name) if name else f"{slug(app_name)}-{slug(getpass.getuser())}"
    if not name.startswith(NAME_PREFIX):
        name = NAME_PREFIX + name
    return name[:60].strip("-")


# Old session files already announced, so a dash polling every second says it once.
_WARNED: set[Path] = set()


def _load() -> tuple[list[Session], list[Path]]:
    """(sessions, files that could not be read as one)."""
    known = {f.name for f in dataclasses.fields(Session)}
    sessions, ignored = [], []
    for path in sorted(SESSIONS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            # Fields a newer `fal dev` added are dropped, not fatal.
            sessions.append(Session(**{k: v for k, v in data.items() if k in known}))
        except (TypeError, AttributeError):
            ignored.append(path)
            if path in _WARNED:
                continue
            _WARNED.add(path)
            print(
                f"Ignoring {path}: written by an older `fal dev`. Delete it, and "
                f"stop its box with `fal apps delete {path.stem}`.",
                file=sys.stderr,
            )
        except ValueError:
            continue
    return sessions, ignored


def load_sessions() -> list[Session]:
    return _load()[0]


def find_session(name: str | None) -> Session:
    """The named session, or the only one, or the one whose root holds the cwd.

    Sessions are kept per user (SESSIONS_DIR), not per folder: any folder finds
    the only box.
    """
    sessions, ignored = _load()
    where = f"in {SESSIONS_DIR}"
    if ignored:
        names = ", ".join(p.name for p in ignored)
        where += f" (unreadable, from another `fal dev` version: {names})"
    if name:
        wanted = {name, session_name(name, "")}
        for session in sessions:
            if session.name in wanted:
                return session
        raise NoSession(f"No dev app session named {name!r} {where}.")
    if len(sessions) == 1:
        return sessions[0]
    cwd = Path.cwd().resolve()
    matching = [s for s in sessions if cwd.is_relative_to(Path(s.root))]
    if len(matching) == 1:
        return matching[0]
    if not sessions:
        raise NoSession(
            f"No dev app session {where}. Start one with `fal dev app start`."
        )
    names = ", ".join(s.name for s in sessions)
    raise DevError(f"Several sessions ({names}); pick one with --name.")


def call(
    session: Session,
    action: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 60,
) -> dict[str, Any]:
    """POST to the box's ``/__fal_dev/<action>`` route."""
    from fal.sdk import get_credentials

    try:
        response = post_json(
            f"{session.run_url}{ROUTE_PREFIX}{action}",
            {**(payload or {}), TOKEN_FIELD: session.token},
            get_credentials(team=session.account.fal_team).to_headers(),
            timeout=timeout,
            step=f"calling the dev box ({action})",
        )
    except httpx.HTTPError as exc:
        raise BoxUnreachable(f"Could not reach the dev box: {exc}") from exc
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        failed = f"{action} failed ({response.status_code}): {detail}"
        # A gateway that gave up, not the box's answer.
        if response.status_code in GATEWAY_ERRORS:
            raise BoxUnreachable(failed)
        raise DevError(failed)
    return response.json()
