import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

from fal.api.api import ResultHandler
from fal.api.client import SyncServerlessClient
from fal.sdk import RunnerInfo, RunnerState

from .parser import FalClientParser

_DEFAULT_MACHINE_TYPE = "S"
_DEFAULT_IDLE_TIMEOUT = 2 * 60 * 60
_STARTUP_TIMEOUT = 15 * 60
_POLL_INTERVAL = 1
_SHELLABLE_STATES = {RunnerState.RUNNING, RunnerState.IDLE}
_RUNNER_START_PATTERN = re.compile(r"^Starting runner (?P<runner_id>\S+)$")


@dataclass
class _DevboxStartup:
    runner_ids: Queue[str]
    errors: Queue[BaseException]


class _DevboxResultHandler(ResultHandler):
    def __init__(self, runner_ids: Queue[str]) -> None:
        self._runner_ids = runner_ids
        self._runner_id: str | None = None
        self._startup_complete = False

    def on_log(self, log: Any) -> None:
        if self._startup_complete:
            return

        match = _RUNNER_START_PATTERN.match(log.message)
        if match is not None:
            self._runner_id = match.group("runner_id")
        elif log.message == "Runner started":
            if self._runner_id is None:
                raise RuntimeError("Devbox started without reporting a runner ID.")
            self._startup_complete = True
            self._runner_ids.put(self._runner_id)


def _project_key(project_dir: Path, host: str | None, team: str | None) -> str:
    identity = f"{project_dir.resolve()}\0{host or ''}\0{team or ''}"
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


def _state_path(project_key: str) -> Path:
    return Path.home() / ".fal" / "devboxes" / f"{project_key}.json"


def _read_runner_id(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    runner_id = data.get("runner_id") if isinstance(data, dict) else None
    return runner_id if isinstance(runner_id, str) else None


def _write_runner_id(path: Path, runner_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps({"runner_id": runner_id}) + "\n")
    temporary_path.replace(path)


def _find_runner(runners: list[RunnerInfo], runner_id: str) -> RunnerInfo | None:
    return next(
        (
            runner
            for runner in runners
            if runner.runner_id == runner_id and runner.state in _SHELLABLE_STATES
        ),
        None,
    )


def _devbox_process() -> None:
    time.sleep(24 * 60 * 60)


def _start_devbox(
    args, client: SyncServerlessClient, project_dir: Path
) -> _DevboxStartup:
    from fal.api.api import function
    from fal.api.run import run

    host = client._create_host(local_file_path=str(project_dir))
    devbox_function = function(host=host)(_devbox_process)
    devbox_function.options.host.update(
        machine_type=args.machine_type,
        keep_alive=args.idle_timeout,
    )

    startup = _DevboxStartup(Queue(maxsize=1), Queue(maxsize=1))
    result_handler = _DevboxResultHandler(startup.runner_ids)

    def run_devbox() -> None:
        try:
            run(devbox_function, result_handler=result_handler)
        except BaseException as exc:
            startup.errors.put(exc)

    Thread(target=run_devbox, daemon=True).start()
    return startup


def _wait_for_runner_id(startup: _DevboxStartup) -> str:
    deadline = time.monotonic() + _STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        try:
            raise startup.errors.get_nowait()
        except Empty:
            pass

        try:
            return startup.runner_ids.get(timeout=_POLL_INTERVAL)
        except Empty:
            pass

    raise RuntimeError("Timed out waiting for the devbox runner to become ready.")


def _attach(args, runner_id: str) -> int:
    from .runners import _shell_session

    args.id = runner_id
    return _shell_session(
        args,
        command=["/bin/bash", "-lc", "cd /app && exec /bin/bash -l"],
        interactive=True,
        remote_tty=True,
    )


def _dev(args):
    project_dir = Path.cwd().resolve()
    project_key = _project_key(project_dir, args.host, args.team)

    print("Project dir", project_dir)
    print("Project key", project_key)

    state_path = _state_path(project_key)
    client = SyncServerlessClient(host=args.host, team=args.team)

    saved_runner_id = _read_runner_id(state_path)
    if saved_runner_id is not None:
        runner = _find_runner(client.runners.list(), saved_runner_id)
        if runner is not None:
            args.console.print(f"Reattaching to devbox {runner.runner_id}...")
            return _attach(args, runner.runner_id)
        try:
            state_path.unlink()
        except FileNotFoundError:
            pass

    args.console.print("Creating devbox...")
    startup = _start_devbox(args, client, project_dir)
    runner_id = _wait_for_runner_id(startup)
    args.console.print(f"Runner id: {runner_id}")
    _write_runner_id(state_path, runner_id)
    args.console.print(f"Attached to devbox {runner_id}")
    return _attach(args, runner_id)


def add_parser(main_subparsers, parents):
    dev_help = "Create or reattach to a development runner."
    parser = main_subparsers.add_parser(
        "dev",
        description=dev_help,
        help=dev_help,
        parents=[*parents, FalClientParser(add_help=False)],
    )
    parser.add_argument(
        "--machine-type",
        default=_DEFAULT_MACHINE_TYPE,
        help=(
            "Machine type to use when creating a devbox "
            f"(default: {_DEFAULT_MACHINE_TYPE})."
        ),
    )
    parser.add_argument(
        "--idle-timeout",
        default=_DEFAULT_IDLE_TIMEOUT,
        type=int,
        metavar="SECONDS",
        help="How long the runner remains available after detaching (default: 7200).",
    )
    parser.set_defaults(func=_dev)
