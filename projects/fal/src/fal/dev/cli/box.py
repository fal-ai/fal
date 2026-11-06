"""`fal dev` box commands: deploy, up, down, shell, logs and friends."""

from __future__ import annotations

import argparse
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rich.markup import escape

from fal.cli.main import parse_args
from fal.cli.runners import _shell_session
from fal.dev.box import FINAL_STATES, DevError, newest_first, ssh_config
from fal.dev.cli.common import get_box, resolve_name


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def _cleanup_on_hangup() -> None:
    # Closing the terminal mid-start cancels the start, like Ctrl-C does.
    signal.signal(signal.SIGTERM, _interrupt)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _interrupt)


def _session_args(args, box) -> argparse.Namespace:
    return argparse.Namespace(
        host=box.host, team=box.team, id=box.require_runner(), console=args.console
    )


def cmd_deploy(args):
    box = get_box(args)
    message = box.deploy(
        gpu=args.machine_type,
        num_gpus=args.num_gpus,
        idle_minutes=args.idle_minutes,
        claude=args.claude,
        no_save=args.no_save,
        dockerfile=args.dockerfile,
        team_data=args.team_data,
    )
    print(message)


def cmd_up(args):
    _cleanup_on_hangup()
    box = get_box(args)
    runner = box.up(args.timeout, args.machine_type)
    args.console.print(
        f"ready: runner {runner.runner_id} ({escape(str(runner.machine_type))})"
    )
    if args.dash:
        from fal.dev.cli.dash import show

        show(box, args.console)
    else:
        args.console.print(
            "connect with: fal dev shell, or watch it with: fal dev dash"
        )


def cmd_down(args):
    get_box(args).down(args.no_save)
    args.console.print("stopped. Your files on /data are kept.")


def cmd_status(args):
    for line in get_box(args).status():
        print(line)


def cmd_shell(args):
    box = get_box(args)
    session = _session_args(args, box)
    return _shell_session(session, command=None, interactive=True, remote_tty=True)


def cmd_exec(args):
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        args.parser.error("usage: fal dev exec -- <command>")
    box = get_box(args)
    session = _session_args(args, box)
    return _shell_session(session, command=command, interactive=False, remote_tty=False)


def cmd_logs(args):
    box = get_box(args)
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


def cmd_hold(args):
    if get_box(args).hold(args.hours):
        args.console.print(
            f"idle shutdown suspended for {args.hours:g}h (until the box next stops)"
        )
    else:
        args.console.print("a longer hold is already in place; kept it")


def _fal_command() -> str:
    """This fal executable, by absolute path: editors run ssh outside your shell."""
    candidate = Path(sys.executable).with_name("fal")
    return str(candidate) if candidate.exists() else (sys.argv[0] or "fal")


def cmd_ssh_config(args):
    print(ssh_config(resolve_name(args), _fal_command(), team=args.team))


def cmd_ssh_proxy(args):
    _cleanup_on_hangup()
    return get_box(args).ssh_proxy(args.timeout)
