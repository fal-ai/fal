"""`fal dev app`: a GPU dev box for an existing fal App, with hot reload.

`start` deploys a private box that inherits the app's machine, requirements,
image and secrets, uploads the app's source tree, and runs ``setup()`` once in a
resident process. `sync` uploads changed files: endpoint edits reload in place
with the weights still loaded; edits to ``setup()`` (or what it uses) restart
the process; requirement or image changes need `start` again.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any

from .parser import FalClientParser, RefAction, add_env_argument

DEFAULT_KEEP_ALIVE = 1800
POLL_SECONDS = 1.0


def _console():
    from fal.console import console

    return console


def _session(args):
    from fal.dev.app_session import find_session

    return find_session(args.session)


def _print_lines(lines: list[str], width: int = 400) -> None:
    for line in lines:
        print(line if len(line) <= width else f"{line[:width]} ...", flush=True)


def _wait_ready(session, *, after: int | None = None, timeout: float = 3600) -> dict:
    """Stream the app's log until it is ready, or raise if it stops."""
    from fal.dev.app_holder import STOPPED_PHASES
    from fal.dev.app_session import DevAppError, call

    deadline = time.monotonic() + timeout
    if after is None:
        after = call(session, "logs", {"lines": 0})["total"]
    while time.monotonic() < deadline:
        logs = call(session, "logs", {"after": after})
        _print_lines(logs["lines"])
        after = logs["total"]
        status = call(session, "status")
        phase = status["phase"]
        if phase == "ready":
            return status
        if phase in STOPPED_PHASES:
            raise DevAppError(
                f"The app did not start ({phase} {status['detail']}). "
                "Fix the code and run `fal dev app restart`."
            )
        time.sleep(POLL_SECONDS)
    raise DevAppError("Timed out waiting for setup() to finish.")


def _sync(session, *, full: bool = False, reload: bool = True) -> dict[str, Any] | None:
    from fal.dev import app_sync
    from fal.dev.app_session import call

    skipped: list[str] = []
    files = app_sync.collect(Path(session.root), skipped)
    if skipped:
        _console().print(
            f"[yellow]Not synced (over 10 MiB)[/]: {', '.join(skipped[:8])}"
        )
    manifest = app_sync.manifest(files)
    previous = {} if full else session.manifest
    changed, deleted = app_sync.diff(previous, manifest)
    if not changed and not deleted and not full:
        return None
    # Large trees go in several requests. Deletes and pruning ride on the
    # first, before any writes; the reload on the last.
    groups = app_sync.batches(files, changed)
    result: dict[str, Any] = {}
    for index, group in enumerate(groups):
        last = index == len(groups) - 1
        payload: dict[str, Any] = {"reload": reload and last}
        if group:
            payload["archive"] = app_sync.pack(files, group)
        if index == 0:
            payload.update(full=full, deleted=deleted)
            if full:
                payload["all_files"] = sorted(files)
        result = call(session, "files", payload, timeout=600)
    action = (result.get("reload") or {}).get("action")
    if action not in ("rebuild", "error", "not_running"):
        # Otherwise keep the old manifest, so the next sync sends the files
        # again and repeats the warning.
        session.manifest = manifest
        session.save()
    result["changed"] = changed
    result["deleted"] = deleted
    return result


def _report_sync(result: dict[str, Any] | None, started: float) -> None:
    console = _console()
    if result is None:
        console.print("No changes.")
        return
    names = ", ".join((result["changed"] + result["deleted"])[:8])
    count = len(result["changed"]) + len(result["deleted"])
    elapsed = time.perf_counter() - started
    console.print(f"Synced {count} file(s) in {elapsed:.2f}s: {names}")
    reload = result.get("reload")
    if not reload:
        return
    action = reload.get("action")
    if action == "reload":
        console.print(
            f"[green]Reloaded[/] in {reload['ms']} ms "
            f"(code version {reload['code_version']}), setup() kept."
        )
    elif action == "restart":
        console.print(
            f"[yellow]Full restart[/]: {reload['reason']}. setup() runs again."
        )
    elif action == "pending":
        console.print(
            f"Saved; {reload['reason']}, so the edit reloads once it is ready."
        )
    elif action == "rebuild":
        console.print(
            f"[red]Not reloaded[/]: {reload['reason']}. "
            "Run `fal dev app start` again to rebuild the box."
        )
    else:
        console.print(f"[red]Not reloaded[/]: {reload.get('reason')}")
        if reload.get("error"):
            print(reload["error"], file=sys.stderr)


def _boot(session) -> dict:
    """Upload the whole source tree and the SDK, then run setup() on the box."""
    from fal.dev import app_sync
    from fal.dev.app_session import call

    _sync(session, full=True, reload=False)
    sdk = app_sync.sdk_files()
    after = call(session, "logs", {"lines": 0}, timeout=600)["total"]
    call(
        session,
        "start",
        {
            "module": session.module,
            "class_name": session.class_name,
            "sdk_archive": app_sync.pack(sdk, sorted(sdk)),
            "sdk_requirements": app_sync.sdk_requirements(),
        },
        timeout=300,
    )
    _console().print(f"Running {session.class_name}.setup() on the box ...")
    return _wait_ready(session, after=after)


def _ensure_booted(session) -> bool:
    """Boot again when the box was scaled to zero or replaced since `start`."""
    from fal.dev.app_session import call

    # A cold box first waits for a runner, which can take minutes.
    status = call(session, "status", timeout=600)
    if status["phase"] != "idle" or status["module"]:
        return False
    _console().print(
        "[yellow]The box is on a new runner[/] (idle shutdown or replacement): "
        "uploading the source and running setup() again."
    )
    _boot(session)
    return True


def _start(args):
    from fal.config import Config
    from fal.dev.app_session import (
        DevAppError,
        Session,
        box_exists,
        deploy_box,
        find_session,
        new_token,
        session_name,
    )

    console = _console()
    file_path, class_name = args.app_ref
    if file_path is None or not file_path.endswith(".py"):
        raise ValueError("Pass the app as path/to/file.py or path/to/file.py::App.")
    file = Path(file_path).resolve()
    root = Path(args.root).resolve() if args.root else file.parent
    if not file.is_relative_to(root):
        raise ValueError(f"{file} is not inside --root {root}")
    module = ".".join(file.relative_to(root).with_suffix("").parts)
    name = session_name(args.session, class_name or file.stem)
    # Pin the team now, so a later `fal team set` cannot retarget the session.
    team = args.team or Config().get_internal("team")
    try:
        existing = find_session(name)
        ours = (
            existing.team,
            existing.host,
            existing.environment,
            existing.root,
        ) == (team, args.host, args.env, str(root))
    except DevAppError:
        ours = False
    exists = box_exists(name, team=team, host=args.host, environment=args.env)
    if exists and not ours and not args.replace:
        raise DevAppError(
            f"An app named {name!r} already exists and was not started from this "
            "machine. Pick another --session, or pass --replace to take it over."
        )

    timings: dict[str, float] = {}
    started = time.perf_counter()
    console.print(f"Deploying dev box [bold]{name}[/] ...")
    token = new_token()
    class_name, run_url = deploy_box(
        str(file),
        class_name,
        root=root,
        app_name=name,
        session=name,
        token=token,
        keep_alive=args.keep_alive,
        machine_type=args.machine_type,
        team=team,
        host=args.host,
        environment=args.env,
    )
    timings["deploy"] = time.perf_counter() - started
    session = Session(
        name=name,
        app_name=name,
        file_path=str(file),
        root=str(root),
        module=module,
        class_name=class_name,
        run_url=run_url,
        token=token,
        team=team,
        host=args.host,
        environment=args.env,
        machine_type=args.machine_type,
    )
    session.save()

    step = time.perf_counter()
    console.print("Waiting for a runner, then uploading the source tree ...")
    status = _boot(session)
    timings["runner+setup"] = time.perf_counter() - step
    app = status.get("app", {})
    summary = ", ".join(f"{k} {v:.1f}s" for k, v in timings.items())
    console.print(f"[green]Ready[/] ({summary}). Endpoints: {app.get('endpoints')}")
    console.print(
        "Next: `fal dev app request / --data '{...}'`, "
        "`fal dev app sync --watch`, `fal dev app stop`."
    )


def _sync_once(session, args, *, full: bool = False) -> None:
    """Sync, wait out a restart, then send the --request if one was given."""
    started = time.perf_counter()
    result = _sync(session, full=full)
    _report_sync(result, started)
    action = ((result or {}).get("reload") or {}).get("action")
    if action in ("restart", "pending"):
        _wait_ready(session)
    if args.request is not None and action in ("reload", "restart", "pending"):
        _do_request(session, args.request, args.data, started)


def _sync_cmd(args):
    from fal.dev import app_sync
    from fal.dev.app_session import DevAppError

    session = _session(args)
    _ensure_booted(session)
    _sync_once(session, args, full=args.full)
    if not args.watch:
        return
    root = Path(session.root)
    _console().print(f"Watching {root} (Ctrl+C to stop) ...")
    signature = app_sync.stat_signature(app_sync.collect(root))
    try:
        while True:
            time.sleep(0.3)
            current = app_sync.stat_signature(app_sync.collect(root))
            if current == signature:
                continue
            signature = current
            try:
                _ensure_booted(session)
                _sync_once(session, args)
            except (DevAppError, OSError, ValueError) as exc:
                # Keep watching: the next save may fix it.
                _console().print(f"[red]{exc}[/]")
    except KeyboardInterrupt:
        pass


def _load_data(data: str | None) -> Any:
    if data is None:
        return {}
    if data.startswith("@"):
        return json.loads(Path(data[1:]).read_text())
    return json.loads(data)


def _do_request(session, path: str, data: str | None, edit_started: float | None):
    from fal.dev.app_session import call

    console = _console()
    started = time.perf_counter()
    result = call(
        session,
        "request",
        {"path": path, "body": _load_data(data), "method": "POST"},
        timeout=3600,
    )
    total = time.perf_counter() - started
    status = result["status_code"]
    color = "green" if status < 400 else "red"
    timing = f"app {result['elapsed_seconds']:.2f}s, round trip {total:.2f}s"
    if edit_started is not None:
        timing += f", edit to result {time.perf_counter() - edit_started:.2f}s"
    console.print(f"[{color}]{status}[/] {path} ({timing})")
    if "body" in result:
        try:
            print(json.dumps(json.loads(result["body"]), indent=2))
        except ValueError:
            print(result["body"])
    else:
        size = len(base64.b64decode(result["body_base64"]))
        print(f"<{size} bytes of {result['content_type']}>")
    return result


def _request(args):
    session = _session(args)
    _ensure_booted(session)
    result = _do_request(session, args.path, args.data, None)
    if args.output_file and "body" in result:
        Path(args.output_file).write_text(result["body"])


def _restart(args):
    from fal.dev.app_session import call

    session = _session(args)
    started = time.perf_counter()
    if _ensure_booted(session):
        _console().print(f"[green]Ready[/] after {time.perf_counter() - started:.1f}s.")
        return
    if args.sync:
        _sync(session, reload=False)
    after = call(session, "logs", {"lines": 0})["total"]
    call(session, "start", {}, timeout=300)
    _wait_ready(session, after=after)
    _console().print(f"[green]Ready[/] after {time.perf_counter() - started:.1f}s.")


def _logs(args):
    from fal.dev.app_session import call

    session = _session(args)
    logs = call(session, "logs", {"lines": args.lines})
    _print_lines(logs["lines"])
    after = logs["total"]
    while args.follow:
        time.sleep(POLL_SECONDS)
        logs = call(session, "logs", {"after": after})
        _print_lines(logs["lines"])
        after = logs["total"]


def _status(args):
    from fal.dev.app_session import DevAppError, call

    session = _session(args)
    info: dict[str, Any] = {
        "session": session.name,
        "app": f"{session.file_path}::{session.class_name}",
        "box": session.run_url,
        "team": session.team,
    }
    try:
        info["remote"] = call(session, "status")
    except DevAppError as exc:
        info["remote"] = {"error": str(exc)}
    print(json.dumps(info, indent=2, default=str))


def _stop(args):
    from fal.dev.app_session import stop_box

    session = _session(args)
    killed, existed = stop_box(session)
    session.delete()
    deleted = "deleted the app" if existed else "the app was already gone"
    _console().print(
        f"Stopped {session.name}: {deleted}, killed {len(killed)} runner(s)."
    )


def add_parser(main_subparsers, parents=None):
    """Register `app` under a `fal dev` subparser group."""
    parents = parents or []
    app_help = "A dev box for an existing fal App, with hot reload."
    parser = main_subparsers.add_parser(
        "app",
        description=__doc__,
        help=app_help,
        parents=parents,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        title="Commands",
        metavar="command",
        dest="cmd",
        required=True,
        parser_class=FalClientParser,
    )
    session_parser = argparse.ArgumentParser(add_help=False)
    session_parser.add_argument(
        "--session",
        "-s",
        help="Box name, always prefixed dev- (defaults to the only session).",
    )

    start = subparsers.add_parser(
        "start",
        help="Deploy a dev box from an app and run its setup() once.",
        parents=[*parents, session_parser],
    )
    start.add_argument(
        "app_ref", action=RefAction, help="path/to/app.py or path/to/app.py::App"
    )
    start.add_argument("--machine-type", help="Override the app's machine type.")
    start.add_argument(
        "--root", help="Source tree to sync (defaults to the app file's folder)."
    )
    add_env_argument(start)
    start.add_argument(
        "--replace",
        action="store_true",
        help="Take over an existing app with the same dev- name.",
    )
    start.add_argument(
        "--keep-alive",
        type=int,
        default=DEFAULT_KEEP_ALIVE,
        help="Seconds without requests before the box shuts down "
        f"(default {DEFAULT_KEEP_ALIVE}).",
    )
    start.set_defaults(func=_start)

    sync = subparsers.add_parser(
        "sync",
        help="Upload changed files; endpoints reload in place.",
        parents=[*parents, session_parser],
    )
    sync.add_argument("--watch", action="store_true", help="Keep syncing on save.")
    sync.add_argument("--full", action="store_true", help="Upload every file.")
    sync.add_argument(
        "--request",
        metavar="PATH",
        help="Send a request to PATH after each sync (with --data).",
    )
    sync.add_argument("--data", help="JSON body, or @file.json.")
    sync.set_defaults(func=_sync_cmd)

    request = subparsers.add_parser(
        "request",
        help="Send a request to the app on the box.",
        parents=[*parents, session_parser],
    )
    request.add_argument("path", help="Endpoint path, e.g. /")
    request.add_argument("--data", help="JSON body, or @file.json.")
    request.add_argument("--output-file", help="Write the response body here.")
    request.set_defaults(func=_request)

    restart = subparsers.add_parser(
        "restart",
        help="Restart the app process; setup() runs again.",
        parents=[*parents, session_parser],
    )
    restart.add_argument(
        "--no-sync", dest="sync", action="store_false", help="Skip syncing first."
    )
    restart.set_defaults(func=_restart)

    logs = subparsers.add_parser(
        "logs",
        help="Show the app process's output.",
        parents=[*parents, session_parser],
    )
    logs.add_argument("--lines", type=int, default=200)
    logs.add_argument("--follow", "-f", action="store_true")
    logs.set_defaults(func=_logs)

    status = subparsers.add_parser(
        "status", help="Show the box and app state.", parents=[*parents, session_parser]
    )
    status.set_defaults(func=_status)

    stop = subparsers.add_parser(
        "stop",
        help="Shut the box down and delete it.",
        parents=[*parents, session_parser],
    )
    stop.set_defaults(func=_stop)
    return parser
