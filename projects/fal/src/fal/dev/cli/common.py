"""What every `fal dev` command needs to find its box."""

from __future__ import annotations

import contextlib
import getpass
import json
import os
import re
from pathlib import Path
from typing import Sequence

from fal.dev import config, state
from fal.dev.box import Box, log
from fal.dev.cli.parser import box_name
from fal.dev.errors import EXIT_USAGE, DevError
from fal.dev.team import announce, resolve_account


def err_console(args):
    """Where status chatter goes: stderr, so stdout keeps only data.

    `fal dev` sets `args.err` when a command runs; without it, the console.
    """
    return getattr(args, "err", None) or args.console


def emit_json(args, data) -> None:
    """Print one result object on stdout."""
    args.console.print_json(json.dumps(data))


def usage_error(args, message: str, fixes: Sequence[tuple[str, str]] = ()) -> DevError:
    """A usage mistake the parser could not see, with the command's own usage.

    Raise the result: `raise usage_error(args, "...", [("fal dev hold 4", "...")])`.
    """
    from fal.cli.parser import _find_parser

    usage = None
    func = getattr(args, "func", None)
    parser = getattr(args, "parser", None)
    if func is not None and parser is not None:
        found = _find_parser(parser, func)
        if found is not None and hasattr(found, "format_usage"):
            usage = found.format_usage()
    return DevError(
        message,
        fixes=fixes,
        code=EXIT_USAGE,
        usage=usage if isinstance(usage, str) else None,
    )


def default_name() -> str:
    if os.environ.get(config.ENV_DEFAULT_NAME):
        return os.environ[config.ENV_DEFAULT_NAME]
    return re.sub(r"[^a-z0-9-]+", "-", getpass.getuser().lower()).strip("-")[:40]


def remembered_name(args) -> str | None:
    """The box this folder last used, if it is still a valid name."""
    remembered = state.load(Path.cwd(), args.host).get("name")
    if remembered and config.NAME_PATTERN.fullmatch(remembered):
        return remembered
    return None


def resolve_name(args) -> str:
    """--name, else the box this folder last used, else $FAL_DEV_NAME or your username.

    Nothing is remembered here: a name is only remembered once the box exists
    (see `get_box` and `remember_name`), so a typo cannot stick to the folder.
    """
    return args.name or remembered_name(args) or box_name(default_name())


def remember_name(args, name: str) -> None:
    """Make `name` this folder's box. Best effort: a read-only home must not fail."""
    with contextlib.suppress(OSError):
        state.update(Path.cwd(), args.host, name=name)


def get_box(
    args,
    *,
    quiet: bool = False,
    creates: bool = False,
    remember: bool = True,
    remember_team: bool = True,
) -> Box:
    """The box, with the account it runs as announced on stderr unless `quiet`.

    An explicit --name is remembered for the folder once the box is known to
    exist. A command that `creates` the box remembers it itself, on success.
    """
    name = resolve_name(args)
    account = resolve_account(
        args.team, config.app_name(name), args.host, remember=remember_team
    )
    if not quiet:
        announce(account, name, verbose=getattr(args, "verbose", False) is True)
    box = Box(name, host=args.host, account=account)
    if remember and args.name and not creates and args.name != remembered_name(args):
        # Best effort: a failed lookup leaves the folder as it was.
        with contextlib.suppress(Exception):
            if box.app_info() is not None:
                remember_name(args, name)
    return box


def note_new_box(args, box: Box) -> None:
    """Say so before a command creates a box the user did not name this time."""
    if args.name:
        return
    source = (
        "remembered for this folder" if remembered_name(args) else "your default name"
    )
    log(
        f"no box named {box.name} exists yet ({source}); creating it. "
        "Press Ctrl-C and pass --name to pick another."
    )
