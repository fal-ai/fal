"""Local state and transport for `fal dev app` sessions.

A session is one dev box for one app. Its state (box URL, token, the manifest of
the last sync) lives in ``~/.fal/dev/apps/<name>.json``. The box is a private
app deployed under the user's team; requests go to its ``/__fal_dev/*`` routes
on fal.run.
"""

from __future__ import annotations

import dataclasses
import getpass
import json
import os
import re
import secrets
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from fal.dev.app_sync import SKIP_DIRS

SESSIONS_DIR = Path("~/.fal/dev/apps").expanduser()
# Every box is named with this prefix, so `start` and `stop` can only ever
# replace or delete dev boxes, never a real app.
NAME_PREFIX = "dev-"


class DevAppError(Exception):
    pass


@dataclasses.dataclass
class Session:
    name: str
    app_name: str
    file_path: str
    root: str
    module: str
    class_name: str
    run_url: str
    token: str
    team: str | None = None
    host: str | None = None
    environment: str | None = None
    machine_type: str | None = None
    created_at: float = dataclasses.field(default_factory=time.time)
    manifest: dict[str, str] = dataclasses.field(default_factory=dict)
    # Dependency overlay layers, oldest first, replayed on a new runner.
    overlay: list[dict[str, Any]] = dataclasses.field(default_factory=list)

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


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-") or "app"


def session_name(name: str | None, app_name: str) -> str:
    """A box name: the given one, or app and user, always under NAME_PREFIX."""
    name = slug(name) if name else f"{slug(app_name)}-{slug(getpass.getuser())}"
    if not name.startswith(NAME_PREFIX):
        name = NAME_PREFIX + name
    return name[:60].strip("-")


def new_token() -> str:
    return secrets.token_urlsafe(32)


def load_sessions() -> list[Session]:
    sessions = []
    for path in sorted(SESSIONS_DIR.glob("*.json")):
        try:
            sessions.append(Session(**json.loads(path.read_text())))
        except (ValueError, TypeError):
            continue
    return sessions


def find_session(name: str | None) -> Session:
    """The named session, or the only one, or the one whose root holds the cwd."""
    sessions = load_sessions()
    if name:
        wanted = {name, session_name(name, "")}
        for session in sessions:
            if session.name in wanted:
                return session
        raise DevAppError(f"No dev app session named {name!r}.")
    if len(sessions) == 1:
        return sessions[0]
    cwd = Path.cwd().resolve()
    matching = [s for s in sessions if cwd.is_relative_to(Path(s.root))]
    if len(matching) == 1:
        return matching[0]
    if not sessions:
        raise DevAppError("No dev app session. Start one with `fal dev app start`.")
    names = ", ".join(s.name for s in sessions)
    raise DevAppError(f"Several sessions ({names}); pick one with --session.")


def _headers(team: str | None) -> dict[str, str]:
    from fal.sdk import get_credentials

    return get_credentials(team=team).to_headers()


def call(
    session: Session,
    action: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 60,
) -> dict[str, Any]:
    """POST to the box's ``/__fal_dev/<action>`` route."""
    from fal.dev.app_holder import ROUTE_PREFIX, TOKEN_FIELD

    # The token travels in the body; keep fal from storing request payloads.
    headers = {**_headers(session.team), "X-Fal-Store-IO": "0"}
    try:
        response = httpx.post(
            f"{session.run_url}{ROUTE_PREFIX}{action}",
            json={**(payload or {}), TOKEN_FIELD: session.token},
            headers=headers,
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        raise DevAppError(f"Could not reach the dev box: {exc}") from exc
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise DevAppError(f"{action} failed ({response.status_code}): {detail}")
    return response.json()


def deploy_box(
    file_path: str,
    class_name: str | None,
    *,
    root: Path,
    app_name: str,
    session: str,
    token: str,
    keep_alive: int,
    machine_type: str | None,
    team: str | None,
    host: str | None,
    environment: str | None,
) -> tuple[str, str]:
    """Deploy the holder for the app in ``file_path``; returns (class name, run URL).

    Loads the module the way `fal deploy` does, then deploys a subclass built by
    ``make_holder`` through the SDK's deployment path instead of the module's
    own class. The SDK internals this needs are confined to this function:
    ``_find_target``, ``SyncServerlessClient._create_host``, ``AppData`` from
    ``fal.cli._utils``, ``fal._serialization`` (by-value module registration),
    a hand-built ``LoadedFunction`` and ``PreparedDeployment`` passed to
    ``execute_prepared_deployment``, setting ``app_name``/``app_auth`` on the
    wrapped function, and ``DeploymentResult.urls``. pyproject.toml
    ``[tool.fal]`` options are not applied.
    """
    import runpy

    import fal._serialization
    from fal.api.client import SyncServerlessClient
    from fal.api.deploy import PreparedDeployment, execute_prepared_deployment
    from fal.app import App, wrap_app
    from fal.cli._utils import AppData
    from fal.dev.app_holder import ROUTE_PREFIX, make_holder
    from fal.utils import LoadedFunction, _find_target

    sys.path.insert(0, str(root))
    module = runpy.run_path(file_path)
    target, _, _, class_name = _find_target(module, class_name)
    if not (isinstance(target, type) and issubclass(target, App)) or not class_name:
        raise DevAppError(f"{file_path}::{class_name} is not a fal.App class.")

    fal._serialization.include_package_from_path(file_path)
    # The box never calls the app's code, but unpickling the class still needs
    # its local helper modules, so ship them by value. Installed packages (a
    # virtualenv inside the root) stay by reference; fal is already by value.
    for name, loaded_module in list(sys.modules.items()):
        source = getattr(loaded_module, "__file__", None)
        if not source or name.partition(".")[0] == "fal":
            continue
        path = Path(source).resolve()
        installed = "site-packages" in path.parts or SKIP_DIRS.intersection(path.parts)
        if path.is_relative_to(root) and not installed:
            fal._serialization.include_module(name)

    holder = make_holder(
        target,
        app_name=app_name,
        session=session,
        token=token,
        keep_alive=keep_alive,
        machine_type=machine_type,
    )
    client = SyncServerlessClient(host=host, team=team)
    serverless = client._create_host(
        local_file_path=file_path, environment_name=environment
    )
    function = wrap_app(holder, host=serverless)
    function.app_name = app_name
    function.app_auth = "private"
    loaded = LoadedFunction(
        function,
        app_name=app_name,
        app_auth="private",
        source_code=Path(file_path).read_text(),
        class_name=holder.__name__,
    )
    result = execute_prepared_deployment(
        PreparedDeployment(
            host=serverless,
            loaded=loaded,
            app_data=AppData(
                auth="private",
                deployment_strategy="recreate",
                reset_scale=True,
                name=app_name,
            ),
            display_name=app_name,
            environment_name=environment,
        )
    )
    status_route = f"{ROUTE_PREFIX}status"
    status_url = result.urls["sync"][status_route]
    return class_name, status_url[: -len(status_route)]


def box_exists(
    name: str, *, team: str | None, host: str | None, environment: str | None
) -> bool:
    from fal.api.client import SyncServerlessClient
    from fal.sdk import construct_alias

    # Aliases outside the main environment carry a suffix; match either form.
    names = {name, construct_alias(name, environment)}
    apps = SyncServerlessClient(host=host, team=team).apps.list(
        environment_name=environment
    )
    return any(app.alias in names for app in apps)


def stop_box(session: Session) -> tuple[list[str], bool]:
    """Kill the box's runners and delete its app.

    Returns the killed runner ids and whether the app still existed.
    """
    import grpc

    from fal.cli._utils import get_client
    from fal.flags import GRPC_HOST

    if not session.app_name.startswith(NAME_PREFIX):
        raise DevAppError(f"Refusing to delete {session.app_name!r}: not a dev box.")
    client = get_client(session.host or GRPC_HOST, session.team)
    killed = []
    with client.connect() as connection:
        try:
            runners = connection.list_alias_runners(
                session.app_name, environment_name=session.environment
            )
        except grpc.RpcError as exc:
            if exc.code() != grpc.StatusCode.NOT_FOUND:
                raise
            runners = []
        for runner in runners:
            connection.kill_runner(runner.runner_id)
            killed.append(runner.runner_id)
        # Same call as `fal apps delete`; the SDK client has no app delete yet.
        revision = connection.delete_alias(
            session.app_name, environment_name=session.environment
        )
    return killed, revision is not None
