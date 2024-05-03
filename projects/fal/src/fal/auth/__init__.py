from __future__ import annotations

import os
from dataclasses import dataclass, field

import click

from fal.auth import auth0, local
from fal.console import console
from fal.console.icons import CHECK_ICON
from fal.exceptions.auth import UnauthenticatedException


def key_credentials() -> tuple[str, str] | None:
    # Ignore key credentials when the user forces auth by user.
    if os.environ.get("FAL_FORCE_AUTH_BY_USER") == "1":
        return None

    if "FAL_KEY" in os.environ:
        key = os.environ["FAL_KEY"]
        key_id, key_secret = key.split(":", 1)
        return (key_id, key_secret)
    elif "FAL_KEY_ID" in os.environ and "FAL_KEY_SECRET" in os.environ:
        return (os.environ["FAL_KEY_ID"], os.environ["FAL_KEY_SECRET"])
    else:
        return None


def login():
    token_data = auth0.login()
    with local.lock_token():
        local.save_token(token_data["refresh_token"])

    USER.invalidate()


def logout():
    refresh_token, _ = local.load_token()
    if refresh_token is None:
        raise click.ClickException(message="You're not logged in")
    auth0.revoke(refresh_token)
    with local.lock_token():
        local.delete_token()

    USER.invalidate()
    console.print(f"{CHECK_ICON} Logged out of [cyan bold]fal[/]. Bye!")


def _fetch_access_token() -> str:
    """
    Load the refresh token, request a new access_token (refreshing the refresh token)
    and return the access_token.
    """
    # We need to lock both read and write access because we could be reading a soon
    # invalid refresh_token
    with local.lock_token():
        refresh_token, access_token = local.load_token()

        if refresh_token is None:
            raise UnauthenticatedException()

        if access_token is not None:
            try:
                auth0.verify_access_token_expiration(access_token)
                return access_token
            except Exception:
                # access_token expired, will refresh
                pass

        try:
            token_data = auth0.refresh(refresh_token)

            # NOTE: Auth0 Refresh Token Rotation enabled
            # So the old refresh_token is no longer valid
            local.save_token(token_data["refresh_token"], token_data["access_token"])
        except:
            local.delete_token()
            raise

        return token_data["access_token"]


@dataclass
class UserAccess:
    _access_token: str | None = field(repr=False, default=None)
    _user_info: dict | None = field(repr=False, default=None)
    _exc: Exception | None = field(repr=False, default=None)

    def invalidate(self) -> None:
        self._access_token = None
        self._user_info = None
        self._exc = None

    @property
    def info(self) -> dict:
        if self._user_info is None:
            self._user_info = auth0.get_user_info(self.bearer_token)

        return self._user_info

    @property
    def access_token(self) -> str:
        if self._exc is not None:
            # We access this several times, so we want to raise the
            # original exception instead of the newer exceptions we
            # would get from the effects of the original exception.
            raise self._exc

        if self._access_token is None:
            try:
                self._access_token = _fetch_access_token()
            except Exception as e:
                self._exc = e
                raise

        return self._access_token

    @property
    def bearer_token(self) -> str:
        return "Bearer " + self.access_token


USER = UserAccess()
