"""`fal dev`: personal GPU dev boxes on fal (experimental).

The CLI layer is thin: box and hot reload logic lives in `fal.dev`. Subcommand
groups register through SUBCOMMANDS, so other `fal dev` commands plug in with
their own `add_parser(subparsers, parents)`.
"""

from __future__ import annotations

import argparse
import contextlib
import getpass
import json
import os
import re
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, List

from fal.dev import config, hot, state
from fal.dev.box import FINAL_STATES, Box, DevError, newest_first, ssh_config

from fal.dev.cli import app as dev_app
from fal.dev.cli import dash as dev_dash
from fal.cli.parser import FalClientParser

EXPERIMENTAL = "(experimental) "
HOT_ACTIONS = ["start", "sync", "run", "status", "log", "restart", "stop"]


def _box_name(value: str) -> str:
    if not config.NAME_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a valid box name: use lowercase letters, digits and "
            f"dashes (set one with --name or ${config.ENV_DEFAULT_NAME})"
        )
    return value


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return number


def default_name() -> str:
    if os.environ.get(config.ENV_DEFAULT_NAME):
        return os.environ[config.ENV_DEFAULT_NAME]
    return re.sub(r"[^a-z0-9-]+", "-", getpass.getuser().lower()).strip("-")[:40]


def resolve_name(args) -> str:
    """--name, else the box this folder last used, else $FAL_DEV_NAME or your username.

    An explicit --name is remembered for the folder.
    """
    cwd = Path.cwd()
    if args.name:
        # Best effort: a read-only home must not break the command.
        with contextlib.suppress(OSError):
            state.update(cwd, args.host, args.team, name=args.name)
        return args.name
    remembered = state.load(cwd, args.host, args.team).get("name")
    if remembered and config.NAME_PATTERN.fullmatch(remembered):
        return remembered
    return _box_name(default_name())


def _get_box(args) -> Box:
    return Box(resolve_name(args), host=args.host, team=args.team)


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def _cleanup_on_hangup() -> None:
    # Closing the terminal mid-start cancels the start, like Ctrl-C does.
    signal.signal(signal.SIGTERM, _interrupt)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _interrupt)


def _deploy(args):
    box = _get_box(args)
    print(
        box.deploy(
            gpu=args.gpu,
            num_gpus=args.num_gpus,
            idle_minutes=args.idle_minutes,
            claude=args.claude,
            no_save=args.no_save,
            dockerfile=args.dockerfile,
            team_data=args.team_data,
        )
    )


def _up(args):
    _cleanup_on_hangup()
    box = _get_box(args)
    runner = box.up(args.timeout, args.gpu)
    print(f"ready: runner {runner.runner_id} ({runner.machine_type})")
    if args.dash:
        dev_dash.show(box, args.console)
    else:
        print("connect with: fal dev shell, or watch it with: fal dev dash")


def _down(args):
    _get_box(args).down(args.no_save)
    print("stopped. Your files on /data are kept.")


def _status(args):
    for line in _get_box(args).status():
        print(line)


def _session_args(box, runner_id: str):
    from fal.console import console

    return argparse.Namespace(
        host=box.host, team=box.team, id=runner_id, console=console
    )


def _shell(args):
    from fal.cli.runners import _shell_session

    box = _get_box(args)
    session = _session_args(box, box.require_runner())
    return _shell_session(session, command=None, interactive=True, remote_tty=True)


def _exec(args):
    from fal.cli.runners import _shell_session

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        args.parser.error("usage: fal dev exec -- <command>")
    box = _get_box(args)
    session = _session_args(box, box.require_runner())
    return _shell_session(session, command=command, interactive=False, remote_tty=False)


def _logs(args):
    from fal.cli.main import parse_args

    box = _get_box(args)
    extra = ["--host", box.host] if box.host else []
    extra += ["--team", box.team] if box.team else []
    if not args.all:
        runner_args = ["runners", "logs", box.require_runner(), "--follow", *extra]
    else:
        # The API only accepts a start time within the last 24 hours.
        since = datetime.now(timezone.utc) - timedelta(hours=23)
        runners = box.runners(since=since)
        if not runners:
            raise DevError(f"{box.app} has no runners in the last 23 hours")
        # Prefer a runner that is still around (e.g. failing to start), else the
        # shortest-lived one.
        active = [r for r in runners if r.state not in FINAL_STATES]
        runner_id = newest_first(active or runners)[0].runner_id
        runner_args = ["runners", "logs", runner_id, *extra]
    logs_args = parse_args(runner_args)
    return logs_args.func(logs_args)


def _hold(args):
    if _get_box(args).hold(args.hours):
        print(f"idle shutdown suspended for {args.hours:g}h (until the box next stops)")
    else:
        print("a longer hold is already in place; kept it")


def _fal_command() -> str:
    """This fal executable, by absolute path: editors run ssh outside your shell."""
    candidate = Path(sys.executable).with_name("fal")
    return str(candidate) if candidate.exists() else (sys.argv[0] or "fal")


def _ssh_config(args):
    print(ssh_config(resolve_name(args), _fal_command(), team=args.team))


def _ssh_proxy(args):
    _cleanup_on_hangup()
    return _get_box(args).ssh_proxy(args.timeout)


def _hot(args):
    box = _get_box(args)
    cwd = Path.cwd()
    remembered = state.load(cwd, args.host, args.team)
    action = args.action
    local = args.local
    if action in ("start", "sync"):
        local, remote = hot.project_paths(local, args.dir, remembered)
        if action == "sync" and not local:
            args.parser.error("fal dev hot sync needs a local project folder")
        with contextlib.suppress(OSError):
            state.update(cwd, args.host, args.team, hot_local=local, hot_dir=remote)

    hot_box = hot.HotBox(box)
    if action == "sync":
        root = Path(local)
        written = hot_box.push(root, remote, sorted(hot.scan(root)))
        print(f"synced {len(written)} changed files to {remote}")
        if args.watch:
            params = hot.parse_params(args.param, args.prompt) if args.run else None
            hot_box.watch(root, remote, params, args.output)
    elif action == "start":
        if local:
            root = Path(local)
            written = hot_box.push(root, remote, sorted(hot.scan(root)))
            print(f"synced {len(written)} changed files to {remote}")
        start = time.time()
        status, info = hot_box.json("start", {"dir": remote})
        if status != 200:
            raise DevError(f"could not start ({status}): {info.get('error')}")
        session = config.HOT_TMUX_SESSION
        print(
            f"started the hot server in {remote} (tmux session {session!r}); "
            "loading ..."
        )
        info = hot_box.wait_ready()
        print(
            f"ready in {time.time() - start:.1f}s "
            f"(model load {info['load_seconds']}s). "
            "Send a request: fal dev hot run"
        )
    elif action == "run":
        params = hot.parse_params(args.param, args.prompt)
        return 0 if hot_box.request(params, args.output) else 1
    elif action == "restart":
        start = time.time()
        status, info = hot_box.json("restart")
        if status != 200:
            raise DevError(f"could not restart ({status}): {info.get('error')}")
        time.sleep(2)  # let the old process exit before polling
        info = hot_box.wait_ready()
        print(
            f"restarted in {time.time() - start:.1f}s "
            f"(model load {info['load_seconds']}s)"
        )
    elif action == "stop":
        status, info = hot_box.json("stop")
        if status != 200:
            raise DevError(f"could not stop ({status}): {info.get('error')}")
        print("stopped the hot server")
    else:  # status, log
        _, info = hot_box.json(action)
        for line in info.pop("log", None) or []:
            print(line)
        if action == "status":
            print(json.dumps(info, indent=2))
    return 0


def _add_box_parsers(subparsers, parents) -> None:
    name_parser = argparse.ArgumentParser(add_help=False)
    name_parser.add_argument(
        "--name",
        type=_box_name,
        help=(
            "Box name; the app is devbox-<name>. Remembered for the current folder "
            f"(default: the folder's last box, else ${config.ENV_DEFAULT_NAME} or "
            "your OS username)."
        ),
    )
    box_parents = [*parents, name_parser]

    def add(cmd: str, help_text: str, func: Callable) -> argparse.ArgumentParser:
        parser = subparsers.add_parser(
            cmd, description=help_text, help=help_text, parents=box_parents
        )
        parser.set_defaults(func=func)
        return parser

    no_save_help = (
        f"Skip saving {config.LOCAL_WORK} first; changes since the last save are lost."
    )
    p = add(
        "deploy",
        "Build the image and (re)deploy your box as a private app. Ends open shells.",
        _deploy,
    )
    p.add_argument(
        "--gpu",
        help=(
            "Machine type, e.g. GPU-H100 "
            f"(default: the current one, else {config.DEFAULT_GPU})."
        ),
    )
    p.add_argument(
        "--num-gpus",
        type=_positive_int,
        default=config.DEFAULT_NUM_GPUS,
        help="GPUs per box (default: %(default)s).",
    )
    p.add_argument(
        "--idle-minutes",
        type=_positive_int,
        default=config.DEFAULT_IDLE_MINUTES,
        help=(
            "Minutes without input or GPU use before the box stops itself "
            "(default: %(default)s)."
        ),
    )
    p.add_argument("--no-save", action="store_true", help=no_save_help)
    p.add_argument(
        "--claude",
        action="store_true",
        help=(
            "Run `claude remote-control` on the box, to use it from the Claude app "
            "(not remembered)."
        ),
    )
    p.add_argument(
        "--dockerfile",
        help="Your own image, based on the default (see fal/dev/remote/Dockerfile).",
    )
    p.add_argument(
        "--team-data",
        action="store_true",
        help=(
            "Keep the box on the team-shared /data (shared model weights, other "
            "apps' files) instead of its own volume. By default /data is the box's "
            "volume, devbox-<name>, and the team /data is not mounted; the first "
            "start copies the box's workspace over from the team /data. Not "
            "remembered: pass it on every deploy."
        ),
    )

    p = add(
        "up", "Start the box (deploying it the first time) and wait until ready.", _up
    )
    p.add_argument(
        "--gpu",
        help="Switch the machine type of a stopped box; it stays until changed again.",
    )
    p.add_argument(
        "--dash",
        action="store_true",
        help="Open the live dashboard (fal dev dash) once the box is ready.",
    )
    p.add_argument(
        "--timeout",
        type=_positive_int,
        default=1800,
        help="Seconds to wait (default: %(default)s).",
    )

    p = add(
        "down", f"Save {config.LOCAL_WORK}, then stop the box. /data is kept.", _down
    )
    p.add_argument("--no-save", action="store_true", help=no_save_help)
    add("status", "Show whether the box is running.", _status)
    add("shell", "Open an interactive shell on the box.", _shell)

    p = add("exec", "Run one command on the box: fal dev exec -- nvidia-smi", _exec)
    p.add_argument(
        "command", nargs=argparse.REMAINDER, help="The command to run, after --."
    )

    p = add("logs", "Follow the box's logs.", _logs)
    p.add_argument(
        "--all",
        action="store_true",
        help="Show the full log of the latest runner, even if it stopped.",
    )

    p = add(
        "hold",
        f"Suspend idle shutdown for N hours (at most {config.MAX_HOLD_HOURS}).",
        _hold,
    )
    p.add_argument(
        "hours", type=float, help="How long to keep the box up regardless of activity."
    )

    add(
        "ssh-config",
        "Print an ~/.ssh/config entry for Cursor / VS Code Remote-SSH.",
        _ssh_config,
    )
    p = add(
        "ssh-proxy",
        "SSH ProxyCommand used by the ssh-config entry; starts a stopped box.",
        _ssh_proxy,
    )
    p.add_argument(
        "--timeout",
        type=_positive_int,
        default=900,
        help="Seconds to wait for a stopped box (default: %(default)s).",
    )

    hot_help = (
        "Hot reload: keep a model loaded on the box and reload your inference code "
        "on save."
    )
    p = add("hot", hot_help, _hot)
    p.description = (
        "A project folder holds load.py (load() -> state, run once) and infer.py "
        "(infer(state, **params), reloaded on every save). Saving load.py or "
        "requirements.txt restarts the server, which reloads the weights. Example: "
        "fal dev hot start ./my-model, then "
        "fal dev hot sync --watch --run --prompt 'a red fox'"
    )
    p.add_argument("action", choices=HOT_ACTIONS)
    p.add_argument(
        "local",
        nargs="?",
        help=(
            "start / sync: a local project folder to push to the box first "
            "(default: the one last used from this folder)."
        ),
    )
    p.add_argument(
        "--dir",
        help=(
            "Project folder on the box "
            f"(default: {config.LOCAL_WORK}/<local folder name>)."
        ),
    )
    p.add_argument(
        "--watch", action="store_true", help="sync: keep pushing files as you save."
    )
    p.add_argument(
        "--run",
        action="store_true",
        help="sync --watch: send a request after each push.",
    )
    p.add_argument("--prompt", help="run: sets the prompt parameter.")
    p.add_argument(
        "-p",
        "--param",
        action="append",
        help="run: a parameter for infer(), key=value (JSON values).",
    )
    p.add_argument(
        "-o",
        "--output",
        help="run: where to save an image result (default: hot-out/<time>.png).",
    )


# Each registers its commands on the `fal dev` subparsers.
SUBCOMMANDS: List[Callable[[argparse._SubParsersAction, list], None]] = [
    _add_box_parsers,
    dev_app.add_parser,
    dev_dash.add_parser,
]


def add_parser(main_subparsers, parents):
    dev_help = (
        f"{EXPERIMENTAL}Personal GPU dev boxes: a persistent box you shell or SSH "
        "into, with hot reload of inference code."
    )
    parser = main_subparsers.add_parser(
        "dev", description=dev_help, help=dev_help, parents=parents
    )
    subparsers = parser.add_subparsers(
        title="Commands",
        metavar="command",
        required=True,
        parser_class=FalClientParser,
    )
    for register in SUBCOMMANDS:
        register(subparsers, parents)
