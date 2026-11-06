"""What every `fal dev` command needs to find its box."""

from __future__ import annotations

import contextlib
import getpass
import os
import re
from pathlib import Path

from fal.dev import config, state
from fal.dev.box import Box, log
from fal.dev.cli.parser import box_name
from fal.dev.team import announce, resolve_account


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
    args, *, quiet: bool = False, creates: bool = False, remember: bool = True
) -> Box:
    """The box, with the account it runs as announced on stderr unless `quiet`.

    An explicit --name is remembered for the folder once the box is known to
    exist. A command that `creates` the box remembers it itself, on success.
    """
    name = resolve_name(args)
    account = resolve_account(args.team, config.app_name(name), args.host)
    if not quiet:
        announce(account)
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
