"""`fal dev dash`: live dashboard for a running dev box."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re

NAME_ENV = "DEVBOX_NAME"
# Each refresh is a runner exec; faster than this only loads the box and the API.
MIN_INTERVAL = 1.0


def _default_name() -> str:
    """$DEVBOX_NAME, else the OS username; resolved only when the command runs."""
    if os.environ.get(NAME_ENV):
        return os.environ[NAME_ENV]
    try:
        user = getpass.getuser()
    except (KeyError, OSError) as exc:
        raise SystemExit(f"could not tell your username; pass a box name ({exc})")
    name = re.sub(r"[^a-z0-9-]+", "-", user.lower()).strip("-")[:40]
    if not name:
        raise SystemExit(f"cannot make a box name from user {user!r}; pass one")
    return name


def _interval(value: str) -> float:
    seconds = float(value)
    if seconds < MIN_INTERVAL:
        raise argparse.ArgumentTypeError(f"must be at least {MIN_INTERVAL:g} seconds")
    return seconds


def _team(args) -> str | None:
    if getattr(args, "team", None):
        return args.team
    from fal.config import Config

    return Config().get_internal("team")


def _dash(args) -> None:
    from fal.dev.dash import FalBox, Poller
    from fal.dev.dash_tui import print_once, run_dashboard

    name = args.name or _default_name()
    box = FalBox(
        name,
        args.app or f"devbox-{name}",
        team=_team(args),
        host=getattr(args, "host", None),
        idle_minutes=args.idle_minutes,
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
        default=None,
        help=f"Box name (default: ${NAME_ENV} or your OS username).",
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
