"""`fal dev` box commands: deploy, up, down, shell, logs and friends."""

from __future__ import annotations

import contextlib
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fal.cli.runners import LogPrinter, print_logs, shell_session
from fal.cli.runners import _get_runner_info as get_runner_info
from fal.dev import config, execute
from fal.dev.box import FINAL_STATES, newest_first, ssh_config
from fal.dev.cli.common import get_box, note_new_box, remember_name, resolve_name
from fal.dev.errors import DevError
from fal.dev.loading import STATES, runner_phase
from fal.dev.steps import Steps
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
    box = get_box(args, creates=True)
    if box.app_info() is None:
        note_new_box(args, box)
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
    remember_name(args, box.name)
    print(message)


def _up_steps(box, args, first_deploy: bool, info) -> Steps:
    """The checklist for this run: only the steps that apply."""
    specs: list[tuple] = []
    if first_deploy:
        specs += [
            ("volume", "create volume"),
            ("build", "build environment"),
            ("deploy", "deploy"),
        ]
    specs += [
        ("runner", "start runner", STATES["gpu-runner"][1]),
        ("ready", "ready"),
    ]
    types = getattr(info, "machine_types", None)
    machine = args.machine_type or (types[0] if types else None) or config.DEFAULT_GPU
    account = box.account
    team = "personal" if account.personal else str(account.team or "unnamed")
    return Steps(
        specs,
        title=box.name,
        team=team,
        machine=machine,
        hint="Ctrl-C cancels the start and stops billing",
        verbose=getattr(args, "verbose", False),
    )


def cmd_up(args):
    _cleanup_on_hangup()
    box = get_box(args, creates=True)
    info = box.app_info()
    if info is None:
        note_new_box(args, box)
        _confirm_deploy(box, args)
    steps = _up_steps(box, args, info is None, info)
    box.progress = steps
    steps.open()

    def on_wait(state: str) -> None:
        steps.update("runner", runner_phase(state))

    def show_log() -> None:
        steps.print_log(args.console)

    try:
        box.up(args.timeout, args.machine_type, on_wait, starship=args.starship)
    except KeyboardInterrupt:
        steps.fail("cancelled")
        if box.start_cancelled:
            args.console.print("cancelled the start: the box scaled back to 0")
        elif box.start_cancelled is None and steps.started("runner"):
            args.console.print(
                "stopped waiting; the box was already up and keeps running"
            )
        else:
            args.console.print(
                "cancelled; check with: fal dev status, stop with: fal dev down"
            )
        return 130
    except BaseException as exc:
        steps.fail(str(exc).splitlines()[0] if str(exc) else None)
        show_log()
        raise
    finally:
        # A box that was created but did not come up is still the folder's box.
        with contextlib.suppress(Exception):
            if box.app_info() is not None:
                remember_name(args, box.name)
    waited = steps.elapsed("runner")
    if args.dash:
        steps.finish("ready")
        steps.done()
        if steps.verbose:
            show_log()
        from fal.dev.cli.dash import show

        show(box, args.console)
        return None
    from fal.dev.cli import summary

    steps.hint = "the box is up and billing; Ctrl-C stops waiting, not the box"
    steps.begin("ready", "reading the box")
    try:
        snap = summary.snapshot(box)
        card = summary.ready_card(snap, box, waited)
    except KeyboardInterrupt:
        steps.fail("cancelled")
        args.console.print("the box is up and billing; stop it with: fal dev down")
        return 130
    except Exception as exc:
        # The box is up: say so, and do not fail `fal dev up && fal dev shell`.
        steps.fail("could not read the box")
        if steps.verbose:
            show_log()
        args.console.print(
            f"the box is up and billing (could not read its details: {exc}); "
            "stop it with: fal dev down",
            markup=False,
        )
        return None
    steps.finish("ready")
    steps.done()
    if steps.verbose:
        show_log()
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
    return execute.run(box.client, box.require_runner(), command, args.console)


FOLLOW_BACKLOG_LINES = 20
MAX_RUNNER_LOOKUPS = 20


def _latest_runner(box, runners) -> str:
    """The runner that started last: a live one, else the newest by start time.

    A finished runner's uptime is its lifetime, so sorting by uptime would pick the
    shortest-lived one, not the latest.
    """
    active = [r for r in runners if r.state not in FINAL_STATES]
    if active:
        return newest_first(active)[0].runner_id
    rest = box.client._create_rest_client()
    started = []
    failed = 0
    for runner in runners[:MAX_RUNNER_LOOKUPS]:
        try:
            at = get_runner_info(rest, runner.runner_id).started_at
        except Exception:
            failed += 1
            continue
        if at is not None:
            started.append((at, runner.runner_id))
    if failed:
        print(
            f"could not read the start time of {failed} runner(s); "
            "the log shown may not be the latest",
            file=sys.stderr,
        )
    if not started:
        return newest_first(runners)[0].runner_id
    return max(started)[1]


def cmd_logs(args):
    box = get_box(args)
    if not args.all:
        try:
            runner_id = box.require_runner()
        except DevError as exc:
            raise DevError(
                f"{str(exc).rstrip('. ')}. To read the last runner's log: "
                "fal dev logs --all"
            ) from None
        # The tail first, so a quiet box still shows something, then new lines.
        args.console.print(f"[dim]last lines of {runner_id}, then following ...[/]")
        # Taken before the tail, so no line falls between the two reads.
        now = datetime.now(timezone.utc)
        printer = LogPrinter(args.console)
        # The API's newest-first history, so a quiet box shows its latest lines.
        for entry in box.logs(
            runner_id, FOLLOW_BACKLOG_LINES, now - timedelta(hours=23)
        ):
            printer.print(entry)
        print_logs(
            box.client,
            runner_id,
            args.console,
            follow=True,
            since=now,
        )
        return
    # The API only accepts a start time within the last 24 hours.
    since = datetime.now(timezone.utc) - timedelta(hours=23)
    runners = box.runners(since=since)
    if not runners:
        raise DevError(f"{box.app} has no runners in the last 23 hours")
    runner_id = _latest_runner(box, runners)
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
    # No name lookup here: editors start the proxy on every connect.
    return get_box(args, quiet=True, remember=False).ssh_proxy(args.timeout)
