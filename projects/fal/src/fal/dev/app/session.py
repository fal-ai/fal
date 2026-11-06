"""Local state and transport for `fal dev app` sessions.

A session is one dev box for one app. Its state (box URL, token, the manifest of
the last sync) lives in ``~/.fal/dev/apps/<name>.json``. The box is a private
app deployed under the user's team; requests go to its ``/__fal_dev/*`` routes
on fal.run.
"""

from __future__ import annotations

import copy
import dataclasses
import getpass
import json
import os
import re
import secrets
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from fal.dev.app.sync import SKIP_DIRS

if TYPE_CHECKING:
    from fal.app import App

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
    from fal.dev.remote.holder import ROUTE_PREFIX, TOKEN_FIELD

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


@dataclasses.dataclass
class AppFiles:
    """The app's ``app_files`` settings after pyproject.toml is applied."""

    files: list[str]
    ignore: list[str]
    # Folder app_files paths are relative to; on a runner this is /app.
    context_dir: Path
    # The app file's folder; on a runner this is the working directory.
    script_dir: Path


@dataclasses.dataclass
class Layout:
    """Where the app's files sit in the synced tree, relative to ``root``."""

    root: Path
    # sys.path entries, and the working directory, of the app process.
    paths: list[str]
    workdir: str
    # Folder app_files_ignore patterns are matched from.
    ignore_base: str
    ignore: list[str]


@dataclasses.dataclass
class DeployedBox:
    class_name: str
    run_url: str
    layout: Layout


def resolve_app(app_ref: tuple[str | None, str | None]) -> tuple[str, str | None, Any]:
    """(file, class name, AppData) for a file ref or a pyproject.toml app name.

    Same resolution as `fal deploy`: an app name reads ``[tool.fal.apps.<name>]``
    (ref, requirements, machine type, app_files and the rest); a file ref does
    not read pyproject.toml.
    """
    from fal.api.deploy import _resolve_deployment_reference

    try:
        (file_path, class_name), app_data = _resolve_deployment_reference(app_ref)
    except ValueError as exc:
        raise DevAppError(str(exc)) from exc
    if file_path is None or not file_path.endswith(".py"):
        raise DevAppError(
            "Pass the app as path/to/file.py[::App], or the name of an app in "
            "pyproject.toml whose ref is a Python file."
        )
    return str(Path(file_path).resolve()), class_name, app_data


def package_root(file: Path) -> Path:
    """The folder holding the file's top-level package (or the file's folder).

    `fal deploy` ships that whole package with the app, so it is the tree the
    app imports from.
    """
    folder = file.parent
    while (folder / "__init__.py").exists():
        folder = folder.parent
    return folder


def _app_files(target: type[App], file_path: str, options: Any) -> AppFiles:
    """The app_files settings `fal deploy` would ship ``target`` with.

    Same merge as ``load_function_from``: the class's host options, then the
    pyproject.toml ones where the class keeps App's defaults or sets nothing.
    """
    from fal.app import App
    from fal.file_sync import _get_script_dir
    from fal.utils import _merge_host_config, _toml_host_keys_overriding_app_defaults

    host = dict(target.host_kwargs)
    if options is not None:
        overrides = _toml_host_keys_overriding_app_defaults(target, App, options)
        _merge_host_config(host, options.host, overrides)
    return AppFiles(
        files=list(host.get("app_files") or []),
        ignore=list(host.get("app_files_ignore") or []),
        context_dir=_get_script_dir(file_path, host.get("app_files_context_dir")),
        script_dir=Path(file_path).resolve().parent,
    )


def layout(import_root: Path, app_files: AppFiles, *, explicit_root: bool) -> Layout:
    """The synced tree and the app's place in it, as on a deployed runner.

    There the app_files context folder is /app, and the app file's folder and
    /app are on sys.path, the former as the working directory. Without
    app_files the app imports from ``import_root``. Unlike a runner, these
    entries go first on sys.path (as the dev box always did).
    """
    root = import_root
    if app_files.files and not app_files.context_dir.is_relative_to(root):
        if explicit_root:
            raise DevAppError(
                f"The app's app_files folder {app_files.context_dir} is not inside "
                f"--root {root}."
            )
        root = Path(os.path.commonpath([root, app_files.context_dir]))
    # The whole tree is synced: never a home folder or anything above one.
    if root == Path(root.anchor) or Path.home().resolve().is_relative_to(root):
        fix = (
            "Pass --root with a narrower folder."
            if explicit_root or not app_files.files
            else "Point app_files_context_dir at a narrower folder."
        )
        raise DevAppError(f"Refusing to sync {root}: it holds your home folder. {fix}")

    def rel(path: Path) -> str:
        return path.relative_to(root).as_posix() or "."

    paths = [rel(import_root)]
    workdir = rel(import_root)
    ignore_base = rel(root)
    # Like `fal deploy`, app_files_ignore only applies to apps with app_files.
    ignore = list(app_files.ignore) if app_files.files else []
    for pattern in ignore:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise DevAppError(f"Bad app_files_ignore pattern {pattern!r}: {exc}")
    if app_files.files:
        workdir = rel(app_files.script_dir)
        ignore_base = rel(app_files.context_dir)
        for path in (workdir, ignore_base):
            if path not in paths:
                paths.append(path)
    return Layout(root, paths, workdir, ignore_base, ignore)


def deploy_box(
    file_path: str,
    class_name: str | None,
    *,
    app_data: Any = None,
    root: Path,
    explicit_root: bool = False,
    app_name: str,
    session: str,
    token: str,
    keep_alive: int,
    machine_type: str | None,
    team: str | None,
    host: str | None,
    environment: str | None,
) -> DeployedBox:
    """Deploy the holder for the app in ``file_path``.

    Loads the app through `fal deploy`'s loader (``load_function_from``), so
    pyproject.toml options in ``app_data.options`` apply as they would on a
    deploy, and deploys a subclass built by ``make_holder`` in place of the
    app's own class. ``root`` is the folder the app imports from; the synced
    tree's layout is worked out (and refused) before anything is deployed.
    The box's scaling, request timeout, health check, auth, team and port
    always come from here, never from pyproject.toml. The app's ``app_files``
    are not uploaded with the box: the dev sync sends the source tree instead.

    SDK internals used here and by its helpers: ``load_function_from``'s
    ``prepare_app`` hook and the option merge it does (``_merge_host_config``,
    ``_toml_host_keys_overriding_app_defaults``), ``_resolve_deployment_reference``
    (``resolve_app``), ``fal.file_sync._get_script_dir``,
    ``SyncServerlessClient._create_host``, ``AppData`` from ``fal.cli._utils``,
    ``fal._serialization`` (by-value module registration),
    ``execute_prepared_deployment`` and ``DeploymentResult.urls``.
    """
    import fal._serialization
    from fal.api import FalServerlessError
    from fal.api.client import SyncServerlessClient
    from fal.api.deploy import PreparedDeployment, execute_prepared_deployment
    from fal.cli._utils import AppData
    from fal.dev.remote.holder import ROUTE_PREFIX, box_host_options, make_holder
    from fal.utils import load_function_from

    found: dict[str, Any] = {}

    def prepare(target: type[App]) -> type[App]:
        found["target"] = target
        found["layout"] = layout(
            root, _app_files(target, file_path, options), explicit_root=explicit_root
        )
        return make_holder(
            target,
            app_name=app_name,
            session=session,
            token=token,
            keep_alive=keep_alive,
            machine_type=machine_type,
        )

    options = None
    if app_data is not None:
        # The box serves its own routes on the port every App gets.
        options = copy.deepcopy(app_data.options)
        options.gateway.pop("exposed_port", None)
    sys.path.insert(0, str(root))
    client = SyncServerlessClient(host=host, team=team)
    serverless = client._create_host(
        local_file_path=file_path, environment_name=environment
    )
    try:
        loaded = load_function_from(
            serverless,
            file_path,
            class_name,
            options=options,
            app_name=app_name,
            app_auth="private",
            prepare_app=prepare,
        )
    except FalServerlessError as exc:
        raise DevAppError(str(exc)) from exc
    if "target" not in found:
        what = f"{file_path}::{class_name}" if class_name else file_path
        raise DevAppError(f"No fal.App class found at {what}.")
    function = loaded.function
    # pyproject.toml may set scaling, timeouts or a health check; the box keeps
    # its own. Its app_files are synced, not uploaded.
    function.options.host.update(box_host_options(keep_alive, machine_type))
    for key in ("app_files", "app_files_ignore", "app_files_context_dir"):
        function.options.host.pop(key, None)

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
    assert loaded.class_name
    return DeployedBox(
        # The name the module exposes it under, which the box imports.
        class_name=loaded.class_name,
        run_url=status_url[: -len(status_route)],
        layout=found["layout"],
    )


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
