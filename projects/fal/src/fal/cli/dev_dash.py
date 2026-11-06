"""`fal dev dash`: live dashboard for a running dev box."""

from __future__ import annotations

import argparse
import json

from fal.dev import config

# Each refresh is a runner exec; faster than this only loads the box and the API.
MIN_INTERVAL = 1.0


def _box_name(value: str) -> str:
    if not config.NAME_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a valid box name: "
            "use lowercase letters, digits and dashes"
        )
    return value


def _interval(value: str) -> float:
    seconds = float(value)
    if seconds < MIN_INTERVAL:
        raise argparse.ArgumentTypeError(f"must be at least {MIN_INTERVAL:g} seconds")
    return seconds


def show(box, console, interval: float = 3.0) -> None:
    """Run the dashboard for a `fal.dev.box.Box` until q or Ctrl-C."""
    from fal.config import Config
    from fal.dev.box import Box
    from fal.dev.dash import FalBox
    from fal.dev.dash_tui import run_dashboard

    # The team names the account in the dashboard; the box itself needs none.
    team = box.team or Config().get_internal("team")
    view = FalBox.from_box(Box(box.name, host=box.host, team=team))
    run_dashboard(view, interval=interval, console=console)


def _dash(args) -> None:
    from fal.config import Config
    from fal.dev.box import Box
    from fal.dev.dash import FalBox, Poller
    from fal.dev.dash_tui import print_once, run_dashboard

    from .dev import resolve_name

    # The team names the account in the dashboard; the box itself needs none.
    team = args.team or Config().get_internal("team")
    box = FalBox.from_box(
        Box(resolve_name(args), host=args.host, team=team),
        idle_minutes=args.idle_minutes,
        app_name=args.app,
    )
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


def add_parser(subparsers, parents=()) -> None:
    dash_help = "Live dashboard for a running dev box."
    parser = subparsers.add_parser(
        "dash",
        description=(
            f"{dash_help} Shows GPU use, idle shutdown, connections, the /work "
            "snapshot and recent logs. Keys: s shell, h hold, d down, "
            "r hot restart, q quit. Watching the box does not keep it awake. "
            "Idle shutdown counts terminal input, a connected SSH client or "
            "editor, GPU use of 10%% or more, Claude sessions, /hot requests "
            "and holds as activity. Cost is an estimate at public list prices."
        ),
        help=dash_help,
        parents=list(parents),
    )
    parser.add_argument(
        "name",
        nargs="?",
        type=_box_name,
        default=None,
        help=(
            "Box name (default: the box this folder last used, else $FAL_DEV_NAME "
            "or your OS username)."
        ),
    )
    parser.add_argument("--app", help="App name of the box (default: devbox-<name>).")
    parser.add_argument(
        "--idle-minutes",
        type=int,
        help="Idle shutdown timeout in minutes (default: what the box reports).",
    )
    parser.add_argument(
        "--interval",
        type=_interval,
        default=3.0,
        help="Seconds between refreshes (default: %(default)s).",
    )
    parser.add_argument("--once", action="store_true", help="Print one frame and exit.")
    parser.add_argument(
        "--json", action="store_true", help="Print the raw status as JSON and exit."
    )
    parser.set_defaults(func=_dash)
