"""`fal dev` box commands: deploy, up, down, shell, logs and friends."""

from __future__ import annotations

import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fal.cli.runners import print_logs, shell_session
from fal.dev.box import FINAL_STATES, newest_first, ssh_config
from fal.dev.cli.common import get_box, resolve_name
from fal.dev.errors import DevError
from fal.dev.loading import STATES, Loading, runner_phase
from fal.dev.team import confirm_deploy


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def _cleanup_on_hangup() -> None:
    # Closing the terminal mid-start cancels the start, like Ctrl-C does.
    signal.signal(signal.SIGTERM, _interrupt)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _interrupt)


def _confirm_deploy(box, args) -> None:
    confirm_deploy(
        box.client, box.account, box.app, box.host, yes=args.yes, name=box.name
    )


def cmd_deploy(args):
    box = get_box(args)
    _confirm_deploy(box, args)
    message = box.deploy(
        machine_type=args.machine_type,
        num_gpus=args.num_gpus,
        idle_minutes=args.idle_minutes,
        claude=args.claude,
        starship=args.starship,
        no_save=args.no_save,
        dockerfile=args.dockerfile,
        team_data=args.team_data,
    )
    print(message)


def cmd_up(args):
    _cleanup_on_hangup()
    box = get_box(args)
    if box.app_info() is None:
        _confirm_deploy(box, args)
    label, expected = STATES["gpu-runner"]
    wait = Loading(
        label,
        title="GPU",
        subtitle="runner pending",
        expected=expected,
        hint="Ctrl-C cancels the start and stops billing",
    )

    def on_wait(state: str) -> None:
        wait.update(f"{label}: {runner_phase(state)}")
        wait.start()  # only once the runner wait begins, after any deploy

    try:
        box.up(args.timeout, args.machine_type, on_wait, starship=args.starship)
    except KeyboardInterrupt:
        wait.close()
        if box.start_cancelled:
            args.console.print("cancelled the start: the box scaled back to 0")
        elif box.start_cancelled is None and wait.started:
            args.console.print(
                "stopped waiting; the box was already up and keeps running"
            )
        else:
            args.console.print(
                "cancelled; check with: fal dev status, stop with: fal dev down"
            )
        return 130
    except BaseException:
        wait.close()
        raise
    waited = wait.elapsed if wait.started else None
    if args.dash:
        wait.done()
        from fal.dev.cli.dash import show

        show(box, args.console)
        return None
    from fal.dev.cli import summary

    wait.update("the runner is up: reading the box")
    wait.hint = "the box is up and billing; Ctrl-C stops waiting, not the box"
    try:
        snap = summary.snapshot(box)
        card = summary.ready_card(snap, box, waited)
    except KeyboardInterrupt:
        wait.close()
        args.console.print("the box is up and billing; stop it with: fal dev down")
        return 130
    except Exception as exc:
        # The box is up: say so, and do not fail `fal dev up && fal dev shell`.
        wait.close()
        args.console.print(
            f"the box is up and billing (could not read its details: {exc}); "
            "stop it with: fal dev down",
            markup=False,
        )
        return None
    finally:
        wait.close()
    wait.done()
    args.console.print(card)
    return None


def cmd_down(args):
    get_box(args).down(args.no_save)
    args.console.print("stopped. Your files on /data are kept.")


def cmd_status(args):
    from fal.dev.cli import summary

    box = get_box(args)
    args.console.print(summary.status_line(summary.snapshot(box), box))


def cmd_shell(args):
    box = get_box(args)
    return shell_session(
        box.client,
        box.require_runner(),
        None,
        interactive=True,
        remote_tty=True,
        console=args.console,
    )


def cmd_exec(args):
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        args.parser.error("usage: fal dev exec -- <command>")
    box = get_box(args)
    return shell_session(
        box.client,
        box.require_runner(),
        command,
        interactive=False,
        remote_tty=False,
        console=args.console,
    )


def cmd_logs(args):
    box = get_box(args)
    if not args.all:
        print_logs(box.client, box.require_runner(), args.console, follow=True)
        return
    # The API only accepts a start time within the last 24 hours.
    since = datetime.now(timezone.utc) - timedelta(hours=23)
    runners = box.runners(since=since)
    if not runners:
        raise DevError(f"{box.app} has no runners in the last 23 hours")
    # Prefer a runner that is still around (e.g. failing to start), else the
    # shortest-lived one.
    active = [r for r in runners if r.state not in FINAL_STATES]
    runner_id = newest_first(active or runners)[0].runner_id
    print_logs(box.client, runner_id, args.console, follow=False, since=since)


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
    print(ssh_config(resolve_name(args), _fal_command(), args.host, args.team))


def cmd_ssh_proxy(args):
    _cleanup_on_hangup()
    # Editors show stderr on every connect, so the account is not announced.
    return get_box(args, quiet=True).ssh_proxy(args.timeout)
