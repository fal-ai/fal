"""`fal dev dash`: live dashboard for a running dev box."""

from __future__ import annotations

import dataclasses
import json

from fal.dev.box import Box
from fal.dev.cli.common import get_box
from fal.dev.dash.model import Poller, Snapshot
from fal.dev.dash.tui import print_once, run_dashboard


def show(box: Box, console, interval: float = 3.0) -> None:
    """Run the dashboard for a box until q or Ctrl-C."""
    run_dashboard(box, interval=interval, console=console)


def _info(box: Box, snapshot: Snapshot) -> dict | None:
    state = snapshot.state
    if state is None:
        return None
    runner = state.runner
    return {
        "name": box.name,
        "app_name": box.app,
        "team": box.account.team,
        "owner": box.owner(),
        "machine_type": state.machine_type,
        "state": state.code,
        "runner_id": runner.runner_id if runner else None,
        "runner_state": runner.state.value.lower() if runner else None,
        "uptime_seconds": runner.uptime.total_seconds() if runner else None,
        "min_concurrency": state.app.min_concurrency if state.app else None,
        "fetched_at": snapshot.state_at,
    }


def cmd_dash(args) -> None:
    box = get_box(args)
    if args.json:
        poller = Poller(box)
        poller.prime()
        snapshot = poller.snapshot
        cost = dataclasses.asdict(snapshot.cost) if snapshot.cost else None
        args.console.print_json(
            json.dumps(
                {
                    "info": _info(box, snapshot),
                    "probe": snapshot.probe,
                    "cost": cost,
                    "errors": snapshot.errors,
                },
                default=str,
            )
        )
        if snapshot.errors:
            raise SystemExit(1)
    elif args.once:
        print_once(box, args.console)
    else:
        run_dashboard(box, interval=args.interval, console=args.console)
