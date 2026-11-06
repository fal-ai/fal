"""`fal dev dash`: live dashboard for a running dev box."""

from __future__ import annotations

import json

from fal.config import Config
from fal.dev.box import Box
from fal.dev.cli.common import get_box
from fal.dev.dash.model import FalBox, Poller
from fal.dev.dash.tui import print_once, run_dashboard


def _view(box: Box, **kwargs) -> FalBox:
    # The team names the account in the dashboard; the box itself needs none.
    team = box.team or Config().get_internal("team")
    return FalBox.from_box(Box(box.name, host=box.host, team=team), **kwargs)


def show(box: Box, console, interval: float = 3.0) -> None:
    """Run the dashboard for a box until q or Ctrl-C."""
    run_dashboard(_view(box), interval=interval, console=console)


def cmd_dash(args) -> None:
    box = _view(get_box(args), idle_minutes=args.idle_minutes, app_name=args.app)
    if args.json:
        poller = Poller(box)
        poller.prime()
        snapshot = poller.snapshot
        info = snapshot.info.__dict__ if snapshot.info else None
        args.console.print_json(
            json.dumps(
                {"info": info, "probe": snapshot.probe, "errors": snapshot.errors},
                default=str,
            )
        )
        if snapshot.errors:
            raise SystemExit(1)
    elif args.once:
        print_once(box, args.console)
    else:
        run_dashboard(box, interval=args.interval, console=args.console)
