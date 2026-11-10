"""Guardrails for teams that share one fal account: box owners and deploy limits.

A box's owner is stamped on every deploy as a revision annotation, and read back
from the app's metadata before the next one, so two teammates who pick the same
box name do not silently redeploy over each other. Everything here runs on your
laptop: it stops mistakes, it is not enforcement.
"""

from __future__ import annotations

import getpass
import os
import re
import socket
import time
from typing import TYPE_CHECKING

from fal.dev import config, state
from fal.dev.errors import DevError

if TYPE_CHECKING:
    from fal.dev.box import Box

OWNER_MAX = 64
CHECK_TTL_SECONDS = 3600

LIMITS_NOTE = (
    f"A client-side guardrail, not enforcement: ${config.ENV_ALLOWED_MACHINES} "
    f"(comma list of machine types) and ${config.ENV_MAX_GPUS} limit what this "
    "command deploys."
)


def clean(value: object) -> str:
    """An owner string safe to print: printable characters only, bounded."""
    return "".join(c for c in str(value) if c.isprintable()).strip()[:OWNER_MAX]


def identity() -> str:
    """Who you are: $FAL_DEV_OWNER, else your fal login, else OS user and host.

    Under a team the login is still your personal account's nickname, so teammates
    differ. An API key names a shared account, not a person, so it falls back.
    """
    if override := clean(os.environ.get(config.ENV_OWNER, "")):
        return override
    from fal.auth import key_credentials

    if not key_credentials():
        from fal.dev.team import accounts

        try:
            listed = accounts()
        except Exception as exc:
            # A silent fallback would make your own box look like a stranger's.
            raise DevError(
                f"could not read your fal login to name the box owner ({exc}); "
                f"retry, or set ${config.ENV_OWNER}"
            ) from exc
        personal = next((a for a in listed if a["is_personal"]), None)
        if personal and clean(personal["nickname"]):
            return clean(personal["nickname"])
    return clean(f"{getpass.getuser()}@{socket.gethostname()}")


def suggested_name(owner: str, name: str) -> str:
    """A box name that is yours: `<you>-<name>`, valid as a box name."""
    mine = re.sub(r"[^a-z0-9-]+", "-", owner.split("@")[0].lower()).strip("-")
    return f"{mine or 'me'}-{name}"[: config.NAME_MAX].rstrip("-")


def check_owner(box: Box, *, take_over: bool) -> None:
    """Refuse to deploy over a box another owner made, unless `take_over`.

    A guard against accidents, not enforcement: two deploys at the same moment
    can both pass, and the stamp is whatever the deployer's client says.
    """
    if not box.account.team:
        box._say("warning: the account has no name, so box owners are not checked")
        return
    try:
        theirs = box.stored_owner()
    except DevError as exc:
        if not take_over:
            raise
        box._say(f"warning: could not read who owns {box.app} ({exc}); taking over")
        return
    if not theirs:
        return  # a new box, or one deployed before owners were stamped
    mine = box.owner_identity()
    if theirs == mine:
        return
    if take_over:
        box._say(f"warning: taking over {box.app} from {theirs}")
        return
    pick = suggested_name(mine, box.name)
    raise DevError(
        f"{box.app} belongs to {theirs} (you are {mine}); deploying would replace "
        "their box and end their shells",
        title="this box is someone else's",
        fixes=[
            (f"fal dev deploy --name {pick}", "make a box of your own"),
            (f"fal dev deploy --name {box.name} --take-over", "replace theirs anyway"),
        ],
    )


def warn_if_foreign(box: Box, args) -> None:
    """Once per box (on this machine), say you are using a teammate's box.

    Best effort, and cheap: editors run this on every connect, so a box that was
    warned about or looked up within CHECK_TTL_SECONDS is not looked up again.
    """
    try:
        known = state.load_box(box.app, args.host)
        if known.get("owner_warned"):
            return
        if time.time() - known.get("owner_checked_at", 0) < CHECK_TTL_SECONDS:
            return
        theirs = box.stored_owner(attempts=1)
        state.update_box(box.app, args.host, owner_checked_at=time.time())
        if not theirs or theirs == box.owner_identity():
            return
        box._say(f"note: {box.app} belongs to {theirs}; you are using their box")
        state.update_box(box.app, args.host, owner_warned=theirs)
    except Exception:
        return


def check_limits(machine_type: str | None, num_gpus: int | None = None) -> None:
    """Enforce $FAL_DEV_ALLOWED_MACHINES and $FAL_DEV_MAX_GPUS, if set."""
    allowed = [
        m.strip()
        for m in os.environ.get(config.ENV_ALLOWED_MACHINES, "").split(",")
        if m.strip()
    ]
    if machine_type and allowed:
        if machine_type.lower() not in {m.lower() for m in allowed}:
            raise DevError(
                f"machine type {machine_type} is not allowed here "
                f"(${config.ENV_ALLOWED_MACHINES}: {', '.join(allowed)})",
                title="machine type not allowed",
                fixes=[(f"--machine-type {allowed[0]}", "an allowed type")],
            )
    raw = os.environ.get(config.ENV_MAX_GPUS, "").strip()
    if raw and num_gpus is not None:
        try:
            cap = int(raw)
        except ValueError:
            raise DevError(
                f"${config.ENV_MAX_GPUS} must be a whole number, not {raw!r}"
            ) from None
        if num_gpus > cap:
            raise DevError(
                f"{num_gpus} GPUs is over the limit of {cap} (${config.ENV_MAX_GPUS})",
                title="too many GPUs",
                fixes=[(f"--num-gpus {cap}", "the most allowed")],
            )
