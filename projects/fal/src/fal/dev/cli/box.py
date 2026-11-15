"""`fal dev` box commands: deploy, up, down, shell, logs and friends."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm, Prompt
from rich.text import Text

from fal.api.client import SyncServerlessClient
from fal.cli.runners import LogPrinter, _iter_logs, _stream_logs, shell_session
from fal.cli.runners import _get_runner_info as get_runner_info
from fal.dev import config, execute, inventory, state
from fal.dev import copy as file_copy
from fal.dev.box import (
    FINAL_STATES,
    install_ssh_config,
    newest_first,
    newest_live,
    ssh_config,
)
from fal.dev.cli.common import (
    emit_json,
    err_console,
    get_box,
    note_new_box,
    remember_name,
    remembered_name,
    resolve_name,
    usage_error,
)
from fal.dev.cli.error_panel import warn
from fal.dev.dash.view import duration
from fal.dev.errors import EXIT_NO_BOX, EXIT_USAGE, DevError
from fal.dev.guard import warn_if_foreign
from fal.dev.loading import STATES, format_elapsed, runner_phase
from fal.dev.steps import Steps
from fal.dev.team import announce, confirm_deploy, resolve_account


def _json(args) -> bool:
    return getattr(args, "json", False) is True


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
        take_over=args.take_over,
        secrets=args.secret,
        set_env=dict(args.set_env),
        unset_secrets=args.unset_secret,
        unset_env=args.unset_env,
    )
    remember_name(args, box.name)
    print(message)


def _host(args) -> str | None:
    return getattr(args, "host", None)


def _last_start_seconds(box, args) -> float | None:
    """How long this box's runner took to start last time, if remembered here."""
    app = getattr(box, "app", None)
    if not isinstance(app, str):
        return None
    seconds = state.load_box(app, _host(args)).get("last_start_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        return None
    return float(seconds) if 0 < seconds < 3600 else None


def _record_use(box, args, start_seconds: float | None = None) -> None:
    """Remember when this box was last used here, and how long its start took."""
    values: dict = {"last_used_at": time.time()}
    if start_seconds:
        values["last_start_seconds"] = round(start_seconds)
    with contextlib.suppress(Exception):
        state.update_box(box.app, _host(args), **values)


def _up_steps(box, args, first_deploy: bool, info) -> Steps:
    """The checklist for this run: only the steps that apply."""
    last = _last_start_seconds(box, args)
    specs: list[tuple[str, str] | tuple[str, str, float | None]] = []
    if first_deploy:
        specs += [
            ("volume", "create volume"),
            ("build", "build environment"),
            ("deploy", "deploy"),
        ]
    specs += [
        ("runner", "start runner", last or STATES["gpu-runner"][1]),
        ("ready", "ready"),
    ]
    types = getattr(info, "machine_types", None)
    machine = args.machine_type or (types[0] if types else None) or config.DEFAULT_GPU
    account = box.account
    team = "personal" if account.personal else str(account.team or "unnamed")
    steps = Steps(
        specs,
        title=box.name,
        team=team,
        machine=machine,
        hint="Ctrl-C cancels the start and stops billing",
        verbose=getattr(args, "verbose", False),
    )
    if last:
        steps.set_note("runner", f"last start took {format_elapsed(last)}")
    return steps


def _up_result(box, snap, waited, total, already_running) -> dict:
    """The `up --json` object: the box and how long it took to come up."""
    runner = snap.state.runner if snap and snap.state else None
    account = box.account
    return {
        "name": box.name,
        "app": box.app,
        "team": account.team,
        "personal": account.personal,
        "state": "running",
        "already_running": already_running,
        "machine_type": getattr(runner, "machine_type", None),
        "runner_id": getattr(runner, "runner_id", None),
        "timings": {
            "runner_seconds": None if waited is None else round(waited, 1),
            "total_seconds": round(total, 1),
        },
    }


def cmd_up(args):
    started = time.monotonic()
    _cleanup_on_hangup()
    if _json(args) and args.dash:
        raise usage_error(
            args,
            "--json prints one result and exits; --dash keeps running",
            [("fal dev up --json", "print the result and exit")],
        )
    box = get_box(args, creates=True)
    err = err_console(args)
    info = box.app_info()
    if info is None:
        note_new_box(args, box)
        _confirm_deploy(box, args)
    else:
        warn_if_foreign(box, args)
    steps = _up_steps(box, args, info is None, info)
    box.progress = steps
    steps.open()

    def on_wait(runner_state: str) -> None:
        phase = (
            "requesting a machine"
            if runner_state == "requested"
            else runner_phase(runner_state)
        )
        steps.update("runner", phase)

    def show_log() -> None:
        steps.print_log(err)

    try:
        box.up(
            args.timeout,
            args.machine_type,
            on_wait,
            starship=args.starship,
            take_over=args.take_over,
            secrets=args.secret,
            set_env=dict(args.set_env),
        )
    except KeyboardInterrupt:
        steps.fail("cancelled")
        if box.start_cancelled:
            err.print("cancelled the start: the box scaled back to 0")
        elif box.start_cancelled is None and steps.started("runner"):
            err.print("stopped waiting; the box was already up and keeps running")
        else:
            err.print("cancelled; check with: fal dev status, stop with: fal dev down")
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
    already_running = getattr(box, "was_live", None) is True
    # Only a real start says how long the next one takes.
    fresh = getattr(box, "was_live", None) is False
    _record_use(box, args, waited if fresh else None)
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
        total = time.monotonic() - started
        if _json(args):
            card = _up_result(box, snap, waited, total, already_running)
        else:
            card = summary.ready_card(
                snap,
                box,
                waited,
                total=total,
                already_running=already_running,
            )
    except KeyboardInterrupt:
        steps.fail("cancelled")
        err.print("the box is up and billing; stop it with: fal dev down")
        return 130
    except Exception as exc:
        # The box is up: say so, and do not fail `fal dev up && fal dev shell`.
        steps.fail("could not read the box")
        if steps.verbose:
            show_log()
        err.print(
            f"the box is up and billing (could not read its details: {exc}); "
            "stop it with: fal dev down",
            markup=False,
        )
        if _json(args):
            emit_json(
                args,
                _up_result(
                    box, None, waited, time.monotonic() - started, already_running
                ),
            )
        return None
    steps.finish("ready")
    steps.done()
    if steps.verbose:
        show_log()
    if _json(args):
        emit_json(args, card)
    else:
        args.console.print(card)
    return None


# Seconds the busy check may take; a wedged box must not delay its own stop.
DOWN_PROBE_TIMEOUT = 10
SHELLS = frozenset({"sh", "bash", "zsh", "fish", "dash", "sshd", "tmux", "login"})


def busy_reasons(probe: dict) -> list[str]:
    """Why stopping the box now would cut work short; empty when it looks idle."""
    reasons = []
    gpus = probe["gpus"]
    busy = [g for g in gpus if (g["util_pct"] or 0) >= config.GPU_BUSY_PCT]
    if busy:
        peak = max(g["util_pct"] for g in busy)
        reasons.append(f"the GPU is busy ({peak:g}% utilization)")
    # The probe tags GPU memory only on its busiest processes by CPU, so a job
    # that is idle on the CPU can be missed; the utilization check above is the
    # backstop for those.
    jobs = [
        p
        for p in (probe.get("system") or {}).get("processes") or []
        if p.get("gpu_mib") and p["name"] not in SHELLS
    ]
    if jobs:
        shown = ", ".join(f"{p['name']} (pid {p['pid']})" for p in jobs[:3])
        more = f" and {len(jobs) - 3} more" if len(jobs) > 3 else ""
        reasons.append(f"{shown}{more} is using the GPU")
    return reasons


def _down_warnings(box) -> list[str]:
    """What is running on a live box; best effort, never blocks a stop."""
    from fal.dev.dash.model import parse_probe, probe_command

    try:
        live = newest_live(box.runners())
        if not live:
            return []
        _, output = box.run(
            live[0].runner_id,
            probe_command(box, False),
            timeout=DOWN_PROBE_TIMEOUT,
            until=config.PROBE_END.encode(),
        )
        return busy_reasons(parse_probe(output))
    except Exception:
        return []


def _confirm_down(box, args) -> None:
    if args.yes:
        return
    reasons = _down_warnings(box)
    if not reasons:
        return
    message = "; ".join(reasons)
    if not sys.stdin.isatty():
        raise DevError(
            f"{message}. Pass --yes to stop it anyway",
            title="the box is busy",
            fixes=[("fal dev down --yes", "stop it anyway")],
        )
    err = Console(stderr=True)
    warn(f"{message}.", err)
    if not Confirm.ask("Stop the box anyway?", default=False, console=err):
        raise DevError("cancelled")


def _save_report(output: str | None) -> str:
    """'/work saved (4.0K)' from the save script's report."""
    import re

    text = " ".join((output or "").split()).removeprefix("devbox-save:").strip()
    found = re.search(r"saved \S+ \(([^)]+)\)", text)
    if found:
        return f"{config.LOCAL_WORK} saved ({found.group(1)})"
    if "no changes" in text:
        return f"{config.LOCAL_WORK} saved (no changes since the last save)"
    return f"{config.LOCAL_WORK} saved" + (f" ({text})" if text else "")


def _up_again(args) -> str:
    name = getattr(args, "name", None)
    return f"fal dev up --name {name}" if name else "fal dev up"


def cmd_down(args):
    began = time.monotonic()
    box = get_box(args)
    err = err_console(args)
    _confirm_down(box, args)
    try:
        saved = box.down(args.no_save)
    except KeyboardInterrupt:
        err.print(
            "interrupted: the box may still be running; check with: fal dev status"
        )
        return 130
    _record_use(box, args)
    if _json(args):
        emit_json(
            args,
            {
                "name": box.name,
                "app": box.app,
                "state": "stopped",
                # True once saved, False with --no-save, None if it was not running.
                "saved": False if args.no_save else (None if saved is None else True),
                "save_report": None
                if args.no_save or saved is None
                else _save_report(saved),
                "kept": config.VOLUME_MOUNT,
                "timings": {"total_seconds": round(time.monotonic() - began, 1)},
            },
        )
        return None
    if args.no_save:
        work = f"{config.LOCAL_WORK} NOT saved (--no-save)"
    elif saved is None:
        work = f"{config.LOCAL_WORK}: nothing to save (the box was not running)"
    else:
        work = _save_report(saved)
    for line in (
        f"stopped {box.name}",
        f"  {work}",
        f"  {config.VOLUME_MOUNT} kept",
        f"  start again: {_up_again(args)} (to delete it instead: fal dev destroy)",
    ):
        err.print(line, markup=False, highlight=False)
    return None


SIZE_SENTINEL = 2**62


def _null_sentinels(value):
    """A copy with byte counts above 2**62 (an 8 EiB 'unlimited' size) as null."""
    if isinstance(value, dict):
        return {
            key: (
                None
                if key.endswith("_bytes")
                and isinstance(item, (int, float))
                and not isinstance(item, bool)
                and item > SIZE_SENTINEL
                else _null_sentinels(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_null_sentinels(item) for item in value]
    return value


def cmd_status(args):
    from fal.dev.cli import summary

    box = get_box(args)
    snap = summary.snapshot(box)
    if args.json:
        from fal.dev.cli.dash import payload

        # Exit 0 for a stopped box: only a state that could not be read fails; probe
        # and cost errors are in the JSON's "errors" (`dash --json` exits 1 on them).
        data = _null_sentinels(json.loads(payload(box, snap)))
        args.console.print_json(json.dumps(data))
        return
    args.console.print(summary.status_line(snap, box))


def _hold_text(row, now: float | None = None) -> str:
    if not row.hold_known:
        return "?"
    left = inventory.hold_left(row, now)
    return "-" if left is None else f"{duration(left)} left"


LIST_STOPPED_SHOWN = 10
STATE_ORDER = ("RUNNING", "STARTING", "STOPPING", "FAILED", "STOPPED", "UNKNOWN")


def _recency(row, host) -> float:
    """When this machine last used the box; 0 when it never did."""
    value = state.load_box(row.app, host).get("last_used_at")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _ordered(rows, host):
    """Running boxes first, then the rest by how recently this machine used them."""
    return sorted(
        rows, key=lambda r: (r.state != "RUNNING", -_recency(r, host), r.name)
    )


def _summary_line(rows) -> str:
    counts = {code: sum(r.state == code for r in rows) for code in STATE_ORDER}
    return ", ".join(f"{n} {code.lower()}" for code, n in counts.items() if n)


def cmd_list(args):
    from rich.table import Table

    # Without --team this is your personal account; a team is never remembered here.
    account = resolve_account(args.team, config.APP_PREFIX, args.host, remember=False)
    client = SyncServerlessClient(host=args.host, team=account.fal_team)
    mine = remembered_name(args)
    rows = _ordered(inventory.list_boxes(client, account, args.host), args.host)
    if args.json:
        args.console.print_json(
            json.dumps([{**r.as_dict(), "remembered": r.name == mine} for r in rows])
        )
        return
    announce(account, verbose=getattr(args, "verbose", False) is True)
    if not rows:
        err_console(args).print(
            f"no dev boxes in {account.runs_as} (for a team, pass --team <name>); "
            "create one with: fal dev up"
        )
        return
    shown = rows
    hidden = 0
    if not getattr(args, "all", False):
        stopped = [r for r in rows if r.state == "STOPPED"]
        # This folder's box stays visible even when it is an old one.
        keep = {r.name for r in stopped[:LIST_STOPPED_SHOWN]} | {mine}
        shown = [r for r in rows if r.state != "STOPPED" or r.name in keep]
        hidden = len(rows) - len(shown)
    args.console.print(_summary_line(rows), markup=False)
    table = Table(box=None, pad_edge=False)
    for column in ("NAME", "STATE", "MACHINE", "UPTIME", "HOLD", "VOLUME"):
        table.add_column(column)
    for row in shown:
        name = Text(row.name)
        if row.name == mine:
            name = Text(f"{row.name}  <- this folder", style="bold cyan")
        table.add_row(
            name,
            row.state,
            row.machine_type or "-",
            duration(row.uptime_seconds) if row.uptime_seconds is not None else "-",
            _hold_text(row),
            escape(row.volume),
        )
    args.console.print(table)
    if hidden:
        err_console(args).print(
            f"+{hidden} stopped (fal dev list --all)", markup=False, style="dim"
        )


def _live_runner_count(box) -> int | None:
    try:
        return sum(r.state not in FINAL_STATES for r in box.runners())
    except Exception:
        return None


def _destroy_summary(box, runners: int | None) -> list[str]:
    """The confirmation text: what goes, what stays, how to get a box back."""
    count = "its runners" if runners is None else f"{runners} runner(s)"
    volume = config.volume_name(box.name)
    return [
        f"Delete {box.app} of {box.account.runs_as}?",
        f"  will delete: the app {box.app} and {count} "
        f"(a running box loses {config.LOCAL_WORK} changes since its last save)",
        f"  will keep:   volume {volume}, with {config.VOLUME_MOUNT} and the "
        f"{config.LOCAL_WORK} snapshot (it cannot be deleted yet)",
        f"  recreate with: fal dev up --name {box.name}",
    ]


def _confirm_destroy(box, args) -> None:
    if args.yes:
        return
    if not sys.stdin.isatty():
        raise DevError(f"pass --yes to delete {box.app} without a terminal to ask on")
    err = Console(stderr=True)
    for line in _destroy_summary(box, _live_runner_count(box)):
        err.print(line, markup=False, highlight=False)
    typed = Prompt.ask(f"Type the box name ({box.name}) to confirm", console=err)
    if typed.strip() != box.name:
        raise DevError("cancelled")


def cmd_destroy(args):
    if args.yes and not args.name:
        # A remembered folder box must not be deleted without being named.
        raise usage_error(
            args,
            "--yes needs --name: say which box to delete",
            [("fal dev destroy --name <box> --yes", "name the box")],
        )
    box = get_box(args, remember=False, remember_team=False)
    if not box.app.startswith(config.APP_PREFIX):
        raise DevError(f"refusing to delete {box.app!r}: not a dev box")
    if box.app_info() is None:
        raise DevError(
            f"no box named {box.name} ({box.app}) exists",
            title="no such box",
            fixes=[
                ("fal dev list", "see your boxes"),
                (f"fal dev up --name {box.name}", "create it"),
            ],
            code=EXIT_NO_BOX,
        )
    _confirm_destroy(box, args)
    killed, _ = box.destroy()
    with contextlib.suppress(OSError):
        state.forget_box(box.app, args.host)
    with contextlib.suppress(OSError):
        state.forget_name(Path.cwd(), box.name, args.host)
    err = err_console(args)
    err.print(f"deleted {box.app}, killed {len(killed)} runner(s).")
    err.print(
        f"/data volume {config.volume_name(box.name)} is kept (no delete API yet); "
        f"`fal dev up --name {box.name}` reattaches it.",
        markup=False,
        highlight=False,
    )


def cmd_shell(args):
    box = get_box(args)
    warn_if_foreign(box, args)
    runner_id = box.require_runner()
    print(
        "exit or ctrl-d to leave; the box keeps running (fal dev down to stop)",
        file=sys.stderr,
        flush=True,
    )
    return shell_session(
        box.client,
        runner_id,
        None,
        interactive=True,
        remote_tty=True,
        console=args.console,
    )


def _machine_type(box) -> str | None:
    """The running box's machine type; None when it cannot be read."""
    try:
        return box.state().machine_type
    except Exception:
        return None


def cmd_exec(args):
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise usage_error(
            args,
            "nothing to run: put the command after --",
            [("fal dev exec -- nvidia-smi", "run a command on the box")],
        )
    stdin = getattr(args, "stdin", False)
    if stdin and sys.stdin.isatty():
        raise usage_error(
            args,
            "-i forwards piped input, and nothing is piped in",
            [
                ("echo hi | fal dev exec -i -- cat", "forward piped input"),
                ("fal dev shell", "an interactive session"),
            ],
        )
    box = get_box(args)
    warn_if_foreign(box, args)
    code = execute.run(
        box.client, box.require_runner(), command, args.console, stdin=stdin
    )
    if code == execute.NOT_FOUND_EXIT:
        machine = _machine_type(box)
        print(execute.not_found_note(machine), file=sys.stderr)
    return code


FOLLOW_BACKLOG_LINES = 20
MAX_RUNNER_LOOKUPS = 20
# The API only accepts a start time within the last 24 hours.
LOG_WINDOW = timedelta(hours=23)
LOG_LINE_MAX = 500
LOG_LINE_KEEP = 200


def _collapse(message, full: bool):
    """A very long log line cut to its start, with its length."""
    if full or not isinstance(message, str) or len(message) <= LOG_LINE_MAX:
        return message
    return f"{message[:LOG_LINE_KEEP]}... ({len(message)} chars, --full to show)"


class _LinePrinter(LogPrinter):
    """The runners log printer, collapsing very long lines unless `full`."""

    def __init__(self, console, full: bool = False) -> None:
        super().__init__(console)
        self.full = full
        self.count = 0

    def print(self, log: dict) -> None:
        self.count += 1
        if "message" in log:
            log = {**log, "message": _collapse(log["message"], self.full)}
        super().print(log)


def _started_at(rest, runner_id: str):
    try:
        return get_runner_info(rest, runner_id).started_at
    except Exception:
        return None


def _latest_runner(box, runners):
    """The runner to read with --all and when it started (None if unknown).

    A live runner first, then one still starting or backing off (it can have no log
    yet), else the one that started last. A finished runner's uptime is its
    lifetime, so sorting by uptime would pick the shortest-lived one, not the
    latest.
    """
    rest = box.client._create_rest_client()
    live = newest_live(runners)
    if live:
        return live[0], _started_at(rest, live[0].runner_id)
    # Still around (starting, backing off): its log is the one that matters.
    around = newest_first([r for r in runners if r.state not in FINAL_STATES])
    if around:
        return around[0], _started_at(rest, around[0].runner_id)
    started = []
    failed = 0
    for runner in runners[:MAX_RUNNER_LOOKUPS]:
        try:
            at = get_runner_info(rest, runner.runner_id).started_at
        except Exception:
            failed += 1
            continue
        if at is not None:
            started.append((at, runner))
    if failed:
        warn(
            f"could not read the start time of {failed} runner(s); "
            "the log shown may not be the latest",
            Console(stderr=True),
        )
    if not started:
        return newest_first(runners)[0], None
    at, runner = max(started, key=lambda item: item[0])
    return runner, at


def _clock(at) -> str:
    return at.astimezone().strftime("%H:%M") if at else "?"


def _print_history(box, runner_id, printer, since, tail) -> None:
    """The runner's log up to now, oldest first: all of it, or the last `tail`."""
    if tail is not None:
        # The API's newest-first history, so a quiet box shows its latest lines.
        for entry in box.logs(runner_id, tail, since):
            printer.print(entry)
        return
    rest = box.client._create_rest_client()
    info = get_runner_info(rest, runner_id)
    if info.started_at is not None and since < info.started_at:
        since = info.started_at
    entries = list(_iter_logs(rest, {"job_id": runner_id}, since, info.ended_at))
    for entry in reversed(entries):
        printer.print(entry)


def _follow(box, runner_id, printer, since) -> None:
    rest = box.client._create_rest_client()
    for entry in _stream_logs(rest, {"job_id": runner_id}, since, None):
        printer.print(entry)


def cmd_logs(args):
    box = get_box(args)
    printer = _LinePrinter(args.console, getattr(args, "full", False))
    tail = getattr(args, "tail", None)
    # Taken before the history, so no line falls between the two reads.
    now = datetime.now(timezone.utc)
    since = now - LOG_WINDOW
    if not args.all:
        try:
            runner_id = box.require_runner()
        except DevError as exc:
            raise DevError(
                f"{str(exc).rstrip('. ')}. To read the last runner's log: "
                "fal dev logs --all"
            ) from None
        follow = not getattr(args, "no_follow", False)
        # The tail first, so a quiet box still shows something, then new lines.
        err_console(args).print(
            f"[dim]last lines of {runner_id}"
            + (", then following (ctrl-c to stop) ...[/]" if follow else "[/]")
        )
        _print_history(
            box,
            runner_id,
            printer,
            since,
            FOLLOW_BACKLOG_LINES if tail is None else tail,
        )
        if follow:
            with contextlib.suppress(KeyboardInterrupt):
                _follow(box, runner_id, printer, now)
        return
    runners = box.runners(since=since)
    if not runners:
        raise DevError(f"{box.app} has no runners in the last 23 hours")
    runner, started = _latest_runner(box, runners)
    runner_id = runner.runner_id
    err_console(args).print(
        f"[dim]full log of {runner_id} ({runner.state.value.lower()}, "
        f"started {_clock(started)})[/]"
        if tail is None
        else f"[dim]last {tail} lines of {runner_id} "
        f"({runner.state.value.lower()}, started {_clock(started)})[/]"
    )
    _print_history(box, runner_id, printer, since, tail)
    if printer.count == 0:
        err_console(args).print(f"no log lines from {runner_id} yet", markup=False)


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds")


def cmd_hold(args):
    err = err_console(args)
    if args.off or args.hours == 0:  # 0 predates --off and still clears
        if args.off and args.hours is not None:
            raise usage_error(
                args,
                "--off takes no hours",
                [("fal dev hold --off", "clear the hold")],
            )
        box = get_box(args)
        cleared = box.clear_hold()
        if _json(args):
            emit_json(
                args,
                {
                    "name": box.name,
                    "state": "cleared" if cleared else "none",
                    "held_until": None,
                },
            )
        elif cleared:
            err.print("hold cleared; idle shutdown applies again")
        else:
            err.print("no hold was set")
        return
    if args.hours is None:
        raise usage_error(
            args,
            "say how many hours to hold, or pass --off",
            [
                ("fal dev hold 4", "keep the box up for 4 hours"),
                ("fal dev hold --off", "clear the hold"),
            ],
        )
    if not 0 < args.hours <= config.MAX_HOLD_HOURS:
        raise DevError(
            f"a hold is more than 0 and at most {config.MAX_HOLD_HOURS} hours "
            f"({config.MAX_HOLD_HOURS // 24} days); got {args.hours:g}",
            fixes=[(f"fal dev hold {config.MAX_HOLD_HOURS}", "the longest hold")],
            code=EXIT_USAGE,
        )
    box = get_box(args)
    set_now = box.hold(args.hours)
    if _json(args):
        if set_now:
            until = time.time() + args.hours * 3600
        else:
            # A longer hold was kept; say when it ends if the box can tell.
            try:
                until = box.hold_until()
            except Exception:
                until = None
        emit_json(
            args,
            {
                "name": box.name,
                "state": "held" if set_now else "kept_longer",
                "hours": args.hours,
                "held_until": None if until is None else _iso(until),
            },
        )
    elif set_now:
        until = time.strftime("%H:%M", time.localtime(time.time() + args.hours * 3600))
        err.print(
            f"idle shutdown held until {until} ({args.hours:g}h). "
            "fal dev hold --off to clear",
            markup=False,
        )
    else:
        err.print("a longer hold is already in place; kept it")


def _fal_command() -> str:
    """This fal executable, by absolute path: editors run ssh outside your shell."""
    candidate = Path(sys.executable).with_name("fal")
    return str(candidate) if candidate.exists() else (sys.argv[0] or "fal")


def _proxy_fal() -> tuple[str, str | None]:
    """The fal to put in a ProxyCommand, and a comment saying why if it is a path.

    A bare `fal` reads best, and is right when PATH finds this same install.
    """
    absolute = _fal_command()
    found = shutil.which("fal")
    if found and os.path.realpath(found) == os.path.realpath(absolute):
        return "fal", None
    reason = (
        "`fal` is not on PATH"
        if found is None
        else "`fal` on PATH is a different install"
    )
    return absolute, f"absolute path to this fal: {reason}"


def _ssh_entry(name: str, args) -> str:
    command, note = _proxy_fal()
    return ssh_config(name, command, args.host, args.team, note)


def cmd_ssh_config(args):
    name = resolve_name(args)
    entry = _ssh_entry(name, args)
    if not getattr(args, "install", False):
        print(entry)
        return
    path = Path.home() / ".ssh" / "config"
    action, backup = install_ssh_config(path, name, entry)
    where = "~/.ssh/config"
    err = err_console(args)
    if action == "unchanged":
        err.print(f"{where} already has {config.app_name(name)}", markup=False)
    else:
        verb = "added to" if action == "added" else "updated in"
        kept = f" (previous file kept as {backup})" if backup else ""
        err.print(f"{config.app_name(name)} {verb} {where}{kept}", markup=False)
    err.print(f"connect with: ssh {config.app_name(name)}", markup=False)


def _run_scp(argv: list[str]) -> int:
    """Run scp, stopping it too when this process is interrupted or terminated."""
    proc = subprocess.Popen(argv)
    try:
        return proc.wait()
    except BaseException:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)
        raise


def cmd_cp(args):
    _cleanup_on_hangup()
    copy = file_copy.plan(args.src, args.dst)
    file_copy.check_local(copy, args.recursive)
    scp = file_copy.find_scp()
    # Resolved here, without announcing the account: the proxy does it per connect.
    name = resolve_name(args)
    entry = _ssh_entry(name, args)
    with tempfile.TemporaryDirectory(prefix="fal-dev-cp-") as folder:
        config_path = Path(folder) / "ssh_config"
        config_path.write_text(entry + "\n")
        argv = file_copy.scp_command(
            copy, config.app_name(name), str(config_path), args.recursive, scp
        )
        err = err_console(args)
        err.print("[dim]connecting to the box (starts it if stopped) ...[/]")
        if copy.upload:
            size = file_copy.human_size(file_copy.tree_size(copy.local))
            err.print(f"[dim]uploading about {size} to :{copy.remote}[/]")
        else:
            err.print(f"[dim]downloading :{copy.remote}[/]")
        began = time.monotonic()
        code = _run_scp(argv)
        took = format_elapsed(time.monotonic() - began)
    if code != 0:
        hint = " (could not connect; see the messages above)" if code == 255 else ""
        raise DevError(
            f"scp failed with exit {code}{hint}; the destination may be incomplete"
        )
    if copy.upload:
        size = file_copy.human_size(file_copy.tree_size(copy.local))
        err.print(
            f"uploaded {size} to :{file_copy.uploaded_path(copy)} ({took})",
            markup=False,
        )
        return
    landed = file_copy.downloaded_path(copy)
    if landed is None:
        err.print(f"downloaded to {copy.local} ({took})", markup=False)
    else:
        size = file_copy.human_size(file_copy.tree_size(landed))
        err.print(f"downloaded {size} to {landed} ({took})", markup=False)


def cmd_ssh_proxy(args):
    _cleanup_on_hangup()
    # Editors show stderr on every connect, so the account is not announced.
    # No name lookup here: editors start the proxy on every connect.
    try:
        box = get_box(args, quiet=True, remember=False)
        warn_if_foreign(box, args)
        return box.ssh_proxy(args.timeout)
    except KeyboardInterrupt:
        # ssh ends the proxy with a signal once it is done; that is no failure.
        return 0
    except DevError:
        # The CLI's error panel prints these with their fix hints.
        raise
    except Exception as exc:
        # Logged out, not deployed, unreachable: the CLI's mapping, with its exit code.
        from fal.dev.cli.parser import known_failure

        known = known_failure(exc)
        if known is not None:
            raise known from exc
        # Editors show only stderr; ssh itself reports a bare "Connection closed".
        reason = str(exc).strip() or type(exc).__name__
        raise DevError(reason, title="fal dev ssh-proxy failed") from exc
