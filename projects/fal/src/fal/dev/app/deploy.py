"""Deploy and delete the box of a `fal dev app` session."""

from __future__ import annotations

import copy
import dataclasses
import os
import re
import secrets
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fal.dev.account import Account
from fal.dev.app.session import NAME_PREFIX, Session, find_session
from fal.dev.errors import DevError
from fal.dev.files import SKIP_DIRS

if TYPE_CHECKING:
    from fal.app import App


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
        raise DevError(str(exc)) from exc
    if file_path is None or not file_path.endswith(".py"):
        raise DevError(
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


def layout(
    import_root: Path,
    file_path: str,
    host: dict[str, Any],
    *,
    explicit_root: bool,
) -> Layout:
    """The synced tree and the app's place in it, as on a deployed runner.

    ``host`` holds the app's host options, pyproject.toml's applied. On a
    runner the app_files context folder is /app, and the app file's folder and
    /app are on sys.path, the former as the working directory. Without
    app_files the app imports from ``import_root``. Unlike a runner, these
    entries go first on sys.path.
    """
    app_files = host.get("app_files")
    script_dir = Path(file_path).resolve().parent
    context_dir = (script_dir / (host.get("app_files_context_dir") or "")).resolve()
    root = import_root
    if app_files and not context_dir.is_relative_to(root):
        if explicit_root:
            raise DevError(
                f"The app's app_files folder {context_dir} is not inside "
                f"--root {root}."
            )
        root = Path(os.path.commonpath([root, context_dir]))
    # The whole tree is synced: never a home folder or anything above one.
    if root == Path(root.anchor) or Path.home().resolve().is_relative_to(root):
        fix = (
            "Pass --root with a narrower folder."
            if explicit_root or not app_files
            else "Point app_files_context_dir at a narrower folder."
        )
        raise DevError(f"Refusing to sync {root}: it holds your home folder. {fix}")

    def rel(path: Path) -> str:
        return path.relative_to(root).as_posix() or "."

    paths = [rel(import_root)]
    workdir = rel(import_root)
    ignore_base = rel(root)
    # Like `fal deploy`, app_files_ignore only applies to apps with app_files.
    ignore = list(host.get("app_files_ignore") or []) if app_files else []
    for pattern in ignore:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise DevError(f"Bad app_files_ignore pattern {pattern!r}: {exc}") from exc
    if app_files:
        workdir = rel(script_dir)
        ignore_base = rel(context_dir)
        for path in (workdir, ignore_base):
            if path not in paths:
                paths.append(path)
    return Layout(root, paths, workdir, ignore_base, ignore)


def deploy_box(
    file_path: str,
    class_name: str | None,
    *,
    app_data: Any,
    root: Path,
    explicit_root: bool = False,
    name: str,
    token: str,
    keep_alive: int,
    machine_type: str | None,
    team: str | None,
    host: str | None,
    environment: str | None,
) -> DeployedBox:
    """Deploy the box for the app in ``file_path``, named ``name``.

    The app is loaded the way `fal deploy` loads it, so the pyproject.toml
    options in ``app_data.options`` apply, and a subclass built by
    ``make_holder`` is deployed in place of the app's own class. ``root`` is
    the folder the app imports from; the synced tree's layout is worked out
    (and refused) before anything is deployed. The box's scaling, request
    timeout, health check, auth, team and port always come from here, never
    from pyproject.toml. The app's ``app_files`` are not uploaded with the box:
    the dev sync sends the source tree instead.
    """
    import fal._serialization
    from fal.api import FalServerlessError
    from fal.api.client import SyncServerlessClient
    from fal.api.deploy import PreparedDeployment, execute_prepared_deployment
    from fal.cli._utils import AppData
    from fal.dev.remote.holder import box_host_options, make_holder
    from fal.dev.remote.protocol import ROUTE_PREFIX
    from fal.utils import load_function_from

    found: dict[str, Layout] = {}

    def prepare(target: type[App], host_options: dict[str, Any]) -> type[App]:
        found["layout"] = layout(
            root, file_path, host_options, explicit_root=explicit_root
        )
        return make_holder(
            target,
            name=name,
            token=token,
            keep_alive=keep_alive,
            machine_type=machine_type,
        )

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
            app_name=name,
            app_auth="private",
            prepare_app=prepare,
        )
    except FalServerlessError as exc:
        raise DevError(str(exc)) from exc
    if "layout" not in found or not loaded.class_name:
        what = f"{file_path}::{class_name}" if class_name else file_path
        raise DevError(f"No fal.App class found at {what}.")
    function = loaded.function
    # pyproject.toml may set scaling, timeouts or a health check; the box keeps
    # its own. Its app_files are synced, not uploaded.
    function.options.host.update(box_host_options(keep_alive, machine_type))
    for key in ("app_files", "app_files_ignore", "app_files_context_dir"):
        function.options.host.pop(key, None)

    # The box never calls the app's code, but unpickling the class still needs
    # its local helper modules, so ship them by value. Installed packages (a
    # virtualenv inside the root) stay by reference; fal is already by value.
    for module_name, module in list(sys.modules.items()):
        source = getattr(module, "__file__", None)
        if not source or module_name.partition(".")[0] == "fal":
            continue
        path = Path(source).resolve()
        installed = "site-packages" in path.parts or SKIP_DIRS.intersection(path.parts)
        if path.is_relative_to(root) and not installed:
            fal._serialization.include_module(module_name)

    result = execute_prepared_deployment(
        PreparedDeployment(
            host=serverless,
            loaded=loaded,
            app_data=AppData(
                auth="private",
                deployment_strategy="recreate",
                reset_scale=True,
                name=name,
            ),
            display_name=name,
            environment_name=environment,
        )
    )
    status_route = f"{ROUTE_PREFIX}status"
    status_url = result.urls["sync"][status_route]
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


def _has_live_runner(
    name: str, account: Account, host: str | None, environment: str | None
) -> bool:
    """Whether the box has a runner that could still be serving its old token."""
    from fal.api.client import SyncServerlessClient
    from fal.dev.box import FINAL_STATES

    try:
        runners = SyncServerlessClient(host=host, team=account.fal_team).apps.runners(
            name, environment_name=environment
        )
    except Exception:
        return False  # unknown: let the deploy go ahead as before
    return any(r.state not in FINAL_STATES for r in runners)


def start_box(
    file: Path,
    class_name: str | None,
    app_data: Any,
    *,
    name: str,
    root: Path,
    explicit_root: bool,
    replace: bool,
    keep_alive: int,
    machine_type: str | None,
    account: Account,
    host: str | None,
    environment: str | None,
    session_env: list[str],
) -> Session:
    """Deploy a box named ``name`` for the app in ``file``, and save its session.

    Refuses to replace an app this machine did not start, unless ``replace``.
    """
    try:
        existing = find_session(name)
    except DevError:
        existing = None
    ours = existing is not None and (
        existing.account.team,
        existing.host,
        existing.environment,
        existing.file_path,
    ) == (account.team, host, environment, str(file))
    exists = box_exists(name, team=account.fal_team, host=host, environment=environment)
    if exists and not ours and not replace:
        raise DevError(
            f"An app named {name!r} already exists and was not started from this "
            "machine. Pick another --name, or pass --replace to take it over."
        )
    if exists and not replace and _has_live_runner(name, account, host, environment):
        raise DevError(
            f"The dev box {name!r} is already running. Push your changes with "
            "`fal dev app sync`, or pass --replace to redeploy it."
        )
    token = secrets.token_urlsafe(32)
    deployed = deploy_box(
        str(file),
        class_name,
        app_data=app_data,
        root=root,
        explicit_root=explicit_root,
        name=name,
        token=token,
        keep_alive=keep_alive,
        machine_type=machine_type,
        team=account.fal_team,
        host=host,
        environment=environment,
    )
    layout = deployed.layout
    session = Session(
        name=name,
        file_path=str(file),
        root=str(layout.root),
        module=".".join(file.relative_to(root).with_suffix("").parts),
        class_name=deployed.class_name,
        run_url=deployed.run_url,
        token=token,
        account=account,
        host=host,
        environment=environment,
        machine_type=machine_type,
        paths=layout.paths,
        workdir=layout.workdir,
        ignore=layout.ignore,
        ignore_base=layout.ignore_base,
        session_env=session_env,
    )
    session.save()
    return session


def stop_box(session: Session) -> tuple[list[str], bool]:
    """Kill the box's runners and delete its app.

    Returns the killed runner ids and whether the app still existed.
    """
    import grpc

    from fal.cli._utils import get_client
    from fal.flags import GRPC_HOST

    if not session.name.startswith(NAME_PREFIX):
        raise DevError(f"Refusing to delete {session.name!r}: not a dev box.")
    client = get_client(session.host or GRPC_HOST, session.account.fal_team)
    killed = []
    with client.connect() as connection:
        try:
            runners = connection.list_alias_runners(
                session.name, environment_name=session.environment
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
            session.name, environment_name=session.environment
        )
    return killed, revision is not None
