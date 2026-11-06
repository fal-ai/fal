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
            state.update(cwd, args.host, args.team, name=args.name)
        return args.name
    remembered = state.load(cwd, args.host, args.team).get("name")
    if remembered and config.NAME_PATTERN.fullmatch(remembered):
        return remembered
    return box_name(default_name())


def get_box(args) -> Box:
    return Box(resolve_name(args), host=args.host, team=args.team)
