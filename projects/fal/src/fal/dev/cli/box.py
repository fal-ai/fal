"""`fal dev` box commands: deploy, up, down, shell, logs and friends."""

from __future__ import annotations

import contextlib
import json
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm, Prompt

from fal.api.client import SyncServerlessClient
from fal.cli.runners import LogPrinter, print_logs, shell_session
from fal.cli.runners import _get_runner_info as get_runner_info
from fal.dev import config, execute, inventory, state
from fal.dev import copy as file_copy
from fal.dev.box import FINAL_STATES, newest_first, newest_live, ssh_config
from fal.dev.cli.common import (
    get_box,
    note_new_box,
    remember_name,
    remembered_name,
    resolve_name,
)
from fal.dev.dash.view import duration
from fal.dev.errors import DevError
from fal.dev.guard import warn_if_foreign
from fal.dev.loading import STATES, runner_phase
from fal.dev.steps import Steps
from fal.dev.team import announce, confirm_deploy, resolve_account


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
    else:
        warn_if_foreign(box, args)
    steps = _up_steps(box, args, info is None, info)
    box.progress = steps
    steps.open()

    def on_wait(state: str) -> None:
        steps.update("runner", runner_phase(state))

    def show_log() -> None:
        steps.print_log(args.console)

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
        raise DevError(f"{message}. Pass --yes to stop it anyway")
    print(f"warning: {message}.", file=sys.stderr)
    if not Confirm.ask(
        "Stop the box anyway?", default=False, console=Console(stderr=True)
    ):
        raise DevError("cancelled")


def cmd_down(args):
    box = get_box(args)
    _confirm_down(box, args)
    box.down(args.no_save)
    args.console.print(
        "stopped. Your files on /data are kept. To delete the box itself: "
        "fal dev destroy"
    )


def cmd_status(args):
    from fal.dev.cli import summary

    box = get_box(args)
    snap = summary.snapshot(box)
    if args.json:
        from fal.dev.cli.dash import payload

        # Exit 0 for a stopped box: only a state that could not be read fails; probe
        # and cost errors are in the JSON's "errors" (`dash --json` exits 1 on them).
        args.console.print_json(payload(box, snap))
        return
    args.console.print(summary.status_line(snap, box))


def _hold_text(row, now: float | None = None) -> str:
    if not row.hold_known:
        return "?"
    left = inventory.hold_left(row, now)
    return "-" if left is None else f"{duration(left)} left"


def cmd_list(args):
    from rich.table import Table

    # Without --team this is your personal account; a team is never remembered here.
    account = resolve_account(args.team, config.APP_PREFIX, args.host, remember=False)
    client = SyncServerlessClient(host=args.host, team=account.fal_team)
    mine = remembered_name(args)
    rows = inventory.list_boxes(client, account, args.host)
    if args.json:
        args.console.print_json(
            json.dumps([{**r.as_dict(), "remembered": r.name == mine} for r in rows])
        )
        return
    announce(account)
    if not rows:
        args.console.print(
            f"no dev boxes in {account.runs_as} (for a team, pass --team <name>); "
            "create one with: fal dev up"
        )
        return
    table = Table(box=None, pad_edge=False)
    for column in ("NAME", "STATE", "MACHINE", "UPTIME", "HOLD", "VOLUME"):
        table.add_column(column)
    for row in rows:
        marker = " *" if row.name == mine else ""
        table.add_row(
            escape(row.name) + marker,
            row.state,
            row.machine_type or "-",
            duration(row.uptime_seconds) if row.uptime_seconds is not None else "-",
            _hold_text(row),
            escape(row.volume),
        )
    args.console.print(table)
    if mine in {r.name for r in rows}:
        args.console.print("[dim]* this folder's box[/]")


def _confirm_destroy(box, args) -> None:
    if args.yes:
        return
    if not sys.stdin.isatty():
        raise DevError(f"pass --yes to delete {box.app} without a terminal to ask on")
    err = Console(stderr=True)
    err.print(
        f"This deletes {box.app} of {box.account.runs_as} and ends its runners. "
        f"A running box loses "
        f"changes in {config.LOCAL_WORK} since its last save. The "
        f"{config.LOCAL_WORK} snapshot and the files on {config.VOLUME_MOUNT} stay on "
        f"the volume {config.volume_name(box.name)}, which cannot be deleted yet.",
        highlight=False,
    )
    typed = Prompt.ask(f"Type the box name ({box.name}) to confirm", console=err)
    if typed.strip() != box.name:
        raise DevError("cancelled")


def cmd_destroy(args):
    if args.yes and not args.name:
        # A remembered folder box must not be deleted without being named.
        raise DevError("--yes needs --name: say which box to delete")
    box = get_box(args, remember=False, remember_team=False)
    if not box.app.startswith(config.APP_PREFIX):
        raise DevError(f"refusing to delete {box.app!r}: not a dev box")
    if box.app_info() is None:
        raise DevError(f"no box named {box.name} ({box.app}) exists")
    _confirm_destroy(box, args)
    killed, _ = box.destroy()
    with contextlib.suppress(OSError):
        state.forget_box(box.app, args.host)
    with contextlib.suppress(OSError):
        state.forget_name(Path.cwd(), box.name, args.host)
    args.console.print(f"deleted {box.app}, killed {len(killed)} runner(s).")
    args.console.print(
        f"/data volume {config.volume_name(box.name)} is kept (no delete API yet); "
        f"`fal dev up --name {box.name}` reattaches it.",
        markup=False,
        highlight=False,
    )


def cmd_shell(args):
    box = get_box(args)
    warn_if_foreign(box, args)
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
    warn_if_foreign(box, args)
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
    if args.off or args.hours == 0:
        if args.off and args.hours is not None:
            args.parser.error("--off takes no hours")
        if get_box(args).clear_hold():
            args.console.print("hold cleared; idle shutdown applies again")
        else:
            args.console.print("no hold was set")
        return
    if args.hours is None:
        args.parser.error("usage: fal dev hold <hours> | --off")
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
    entry = ssh_config(name, _fal_command(), args.host, args.team)
    with tempfile.TemporaryDirectory(prefix="fal-dev-cp-") as folder:
        config_path = Path(folder) / "ssh_config"
        config_path.write_text(entry + "\n")
        argv = file_copy.scp_command(
            copy, config.app_name(name), str(config_path), args.recursive, scp
        )
        if copy.upload:
            size = file_copy.human_size(file_copy.tree_size(copy.local))
            args.console.print(
                f"[dim]uploading about {size} to {copy.remote} "
                "(starts the box if stopped) ...[/]"
            )
        else:
            args.console.print(
                f"[dim]downloading {copy.remote} (starts the box if stopped) ...[/]"
            )
        code = _run_scp(argv)
    if code != 0:
        hint = " (could not connect; see the messages above)" if code == 255 else ""
        raise DevError(
            f"scp failed with exit {code}{hint}; the destination may be incomplete"
        )
    if copy.upload:
        args.console.print(f"uploaded to {copy.remote}")
        return
    landed = file_copy.downloaded_path(copy)
    if landed is None:
        args.console.print(f"downloaded to {copy.local}")
    else:
        size = file_copy.human_size(file_copy.tree_size(landed))
        args.console.print(f"downloaded {size} to {landed}")


def cmd_ssh_proxy(args):
    _cleanup_on_hangup()
    # Editors show stderr on every connect, so the account is not announced.
    # No name lookup here: editors start the proxy on every connect.
    try:
        box = get_box(args, quiet=True, remember=False)
        warn_if_foreign(box, args)
        return box.ssh_proxy(args.timeout)
    except DevError:
        # The CLI's error panel prints these with their fix hints.
        raise
    except Exception as exc:
        # Editors show only stderr; ssh itself reports a bare "Connection closed".
        reason = str(exc).strip() or type(exc).__name__
        print(f"fal dev ssh-proxy failed: {reason}", file=sys.stderr, flush=True)
        return 1
