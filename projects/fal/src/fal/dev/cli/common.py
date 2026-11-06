"""What every `fal dev` command needs to find its box."""

from __future__ import annotations

import contextlib
import getpass
import os
import re
from pathlib import Path

from fal.dev import config, state
from fal.dev.box import Box
from fal.dev.cli.parser import box_name
from fal.dev.team import announce, resolve_account


def default_name() -> str:
    if os.environ.get(config.ENV_DEFAULT_NAME):
        return os.environ[config.ENV_DEFAULT_NAME]
    return re.sub(r"[^a-z0-9-]+", "-", getpass.getuser().lower()).strip("-")[:40]


def resolve_name(args) -> str:
    """--name, else the box this folder last used, else $FAL_DEV_NAME or your username.

    An explicit --name is remembered for the folder.
    """
    cwd = Path.cwd()
    if args.name:
        # Best effort: a read-only home must not break the command.
        with contextlib.suppress(OSError):
            state.update(cwd, args.host, name=args.name)
        return args.name
    remembered = state.load(cwd, args.host).get("name")
    if remembered and config.NAME_PATTERN.fullmatch(remembered):
        return remembered
    return box_name(default_name())


def get_box(args, *, quiet: bool = False) -> Box:
    """The box, with the account it runs as announced on stderr unless `quiet`."""
    name = resolve_name(args)
    account = resolve_account(args.team, config.app_name(name), args.host)
    if not quiet:
        announce(account)
    return Box(name, host=args.host, account=account)
