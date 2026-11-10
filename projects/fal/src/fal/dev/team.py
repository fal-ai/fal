"""Choosing the account a `fal dev` box runs as, and guarding production teams.

Without a team, fal uses the one set with `fal account set`. A box never does:
it runs as the team passed with --team (remembered per box), else as your
personal account, and the account's name is always handed to fal explicitly, so
that default cannot apply. Under an API key the key picks the account; its name
is handed over too, so fal has nothing to say about ignoring a team.
"""

from __future__ import annotations

import contextlib
import sys

from rich.console import Console
from rich.prompt import Confirm

from fal.api.client import SyncServerlessClient
from fal.dev import config, state
from fal.dev.account import Account
from fal.dev.app.session import NAME_PREFIX
from fal.dev.box import log
from fal.dev.errors import DevError

SHOWN_APPS = 5


def accounts() -> list[dict]:
    """The accounts you can act as, personal first."""
    from fal.auth import UserAccess

    return UserAccess().accounts


def key_account() -> Account | None:
    """The account an API key belongs to; None when you are logged in as a user."""
    from fal.auth import current_user_info, key_credentials
    from fal.exceptions import FalServerlessException
    from fal.sdk import FalServerlessKeyCredentials, is_permission_denied

    key = key_credentials()
    if not key:
        return None
    headers = FalServerlessKeyCredentials(*key).to_headers()
    try:
        nickname = current_user_info(headers)["nickname"]
    except FalServerlessException as exc:
        # A key without permission to read its account still works.
        if not is_permission_denied(exc.__cause__):
            raise DevError(
                f"could not read the API key's account: {exc.__cause__ or exc}"
            ) from exc
        nickname = None
    except OSError as exc:
        raise DevError(f"could not read the API key's account: {exc}") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise DevError(
            f"could not read the API key's account: unexpected response ({exc!r})"
        ) from exc
    return Account(nickname, False, via_key=True)


def resolve_account(
    explicit: str | None, app: str, host: str | None, *, remember: bool = True
) -> Account:
    """--team, else the box's remembered team, else your personal account.

    A team is remembered for the box once it has proved to be one of yours.
    """
    if key := key_account():
        if explicit and explicit.lower() != (key.team or "").lower():
            log(f"--team {explicit} is ignored: an API key runs as its own account")
        return key
    listed = accounts()
    personal = next((a for a in listed if a["is_personal"]), None)
    if personal is None:
        raise DevError("your fal login has no personal account")
    wanted = explicit or state.load_box(app, host).get("team")
    if not wanted:
        return Account(personal["nickname"], True)
    found = next((a for a in listed if a["nickname"].lower() == wanted.lower()), None)
    if found is None:
        names = ", ".join(a["nickname"] for a in listed)
        message = f"you have no team named {wanted!r} (yours: {names})"
        if not explicit:
            message += (
                f". It was remembered for {app}: pass --team <name>, or "
                f"--team {personal['nickname']} for your personal account"
            )
        raise DevError(message)
    if explicit and remember:
        # Best effort: a read-only home must not break the command.
        with contextlib.suppress(OSError):
            state.update_box(app, host, team=found["nickname"])
    return Account(found["nickname"], found["is_personal"])


def announce(account: Account) -> None:
    """Say which account a command runs as, on stderr (stdout carries data)."""
    log(f"Team: {account}")


def production_apps(client: SyncServerlessClient) -> list[str]:
    """The team's apps that are not dev boxes.

    Dev boxes are recognised by their devbox-/dev- name prefixes, so a production
    app named that way is not counted.
    """
    prefixes = (config.APP_PREFIX, NAME_PREFIX)
    return sorted(
        app.alias for app in client.apps.list() if not app.alias.startswith(prefixes)
    )


def confirm_deploy(
    client: SyncServerlessClient,
    account: Account,
    app: str,
    host: str | None,
    *,
    yes: bool,
    name: str | None = None,
) -> None:
    """Before deploying box app `app`, have the user accept a team with production apps.

    Asked once per box and team: the answer is remembered. `name` is the box as
    the user knows it, when that differs from the app's name.
    """
    if yes or account.personal:
        return
    # An API key whose account could not be named is remembered as "key".
    confirmed = account.team or "key"
    if state.load_box(app, host).get("confirmed_team") == confirmed:
        return
    try:
        apps = production_apps(client)
    except Exception as exc:
        raise DevError(f"could not list the apps of {account.runs_as}: {exc}") from exc
    if not apps:
        return
    shown = ", ".join(apps[:SHOWN_APPS])
    shown += f" and {len(apps) - SHOWN_APPS} more" if len(apps) > SHOWN_APPS else ""
    owner = f"{account.runs_as} has production apps ({shown})"
    box = f"{name} ({app})" if name else app
    if not sys.stdin.isatty():
        raise DevError(f"{owner}; pass --yes to deploy {box} anyway")
    log(f"warning: {owner}. {box} would be deployed next to them.")
    if not Confirm.ask("Continue?", default=False, console=Console(stderr=True)):
        raise DevError("cancelled")
    with contextlib.suppress(OSError):
        state.update_box(app, host, confirmed_team=confirmed)
