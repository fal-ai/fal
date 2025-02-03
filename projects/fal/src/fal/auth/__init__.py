from __future__ import annotations

import os
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

import click

from fal.auth import auth0, local, workos_auth
from fal.config import Config
from fal.console import console
from fal.console.icons import CHECK_ICON
from fal.exceptions.auth import UnauthenticatedException


class GoogleColabState:
    def __init__(self):
        self.is_checked = False
        self.lock = Lock()
        self.secret: Optional[str] = None


_colab_state = GoogleColabState()


def is_google_colab() -> bool:
    try:
        from IPython import get_ipython

        return "google.colab" in str(get_ipython())
    except ModuleNotFoundError:
        return False
    except NameError:
        return False


def get_colab_token() -> Optional[str]:
    if not is_google_colab():
        return None
    with _colab_state.lock:
        if _colab_state.is_checked:  # request access only once
            return _colab_state.secret

        try:
            from google.colab import userdata  # noqa: I001
        except ImportError:
            return None

        try:
            token = userdata.get("FAL_KEY")
            _colab_state.secret = token.strip()
        except Exception:
            _colab_state.secret = None

        _colab_state.is_checked = True
        return _colab_state.secret


def key_credentials() -> tuple[str, str] | None:
    # Ignore key credentials when the user forces auth by user.
    if os.environ.get("FAL_FORCE_AUTH_BY_USER") == "1":
        return None

    config = Config()

    key = os.environ.get("FAL_KEY") or config.get("key") or get_colab_token()
    if key:
        key_id, key_secret = key.split(":", 1)
        return (key_id, key_secret)
    elif "FAL_KEY_ID" in os.environ and "FAL_KEY_SECRET" in os.environ:
        return (os.environ["FAL_KEY_ID"], os.environ["FAL_KEY_SECRET"])
    else:
        return None


def login() -> None:
    if os.environ.get("FAL_USE_AUTH0") == "true":
        auth_config = auth0.login()
    else:
        auth_config = workos_auth.login()

    with local.lock_token():
        local.save_auth_config(auth_config)

    USER.invalidate()


def logout():
    auth_token = local.load_auth_config()
    if auth_token is None:
        raise click.ClickException(message="You're not logged in")

    if auth_token.provider == "auth0":
        auth0.revoke(auth_token.refresh_token)
    if auth_token.provider == "workos":
        # We should always have an access token for WorkOS
        if auth_token.access_token:
            workos_auth.revoke(auth_token.access_token)

    with local.lock_token():
        local.delete_token()

    USER.invalidate()
    console.print(f"{CHECK_ICON} Logged out of [cyan bold]fal[/]. Bye!")


def _fetch_auth_config(
    force_refresh: bool = False,
) -> tuple[local.ActiveAuthConfig, local.UserInfo | None]:
    """
    Load the config, request a new access_token (refreshing the refresh token),
    save and return the updated config.
    """
    # We need to lock both read and write access because we could be reading a soon
    # invalid refresh_token
    with local.lock_token():
        auth_config = local.load_auth_config()

        if auth_config is None:
            raise UnauthenticatedException()

        if not force_refresh and auth_config.access_token is not None:
            try:
                local.verify_access_token_expiration(auth_config.access_token)
                return local.ActiveAuthConfig(
                    provider=auth_config.provider,
                    access_token=auth_config.access_token,
                    refresh_token=auth_config.refresh_token,
                ), None
            except Exception:
                # access_token expired, will refresh
                pass

        # Refresh the token
        # NOTE: Both Auth0 and WorkOS rotate refresh tokens
        # so we have to save the new one
        try:
            if auth_config.provider == "auth0":
                active_config = auth0.refresh(auth_config.refresh_token)
                user_info = None
            else:
                active_config, user_info = workos_auth.refresh(
                    auth_config.refresh_token
                )

            local.save_auth_config(active_config)

            return active_config, user_info
        except:
            local.delete_token()
            raise


@dataclass
class UserAccess:
    _auth_config: local.ActiveAuthConfig | None = field(repr=False, default=None)
    _user_info: local.UserInfo | None = field(repr=False, default=None)
    _exc: Exception | None = field(repr=False, default=None)

    def invalidate(self) -> None:
        self._auth_config = None
        self._user_info = None
        self._exc = None

    @property
    def auth_config(self) -> local.ActiveAuthConfig:
        if self._exc is not None:
            # We access this several times, so we want to raise the
            # original exception instead of the newer exceptions we
            # would get from the effects of the original exception.
            raise self._exc

        if self._auth_config is None:
            try:
                self._auth_config, user_info = _fetch_auth_config()
                if user_info:
                    self._user_info = user_info
            except Exception as e:
                self._exc = e
                raise

        return self._auth_config

    @property
    def info(self) -> local.UserInfo:
        # Fetch the auth config which may populate the user info
        auth_config = self.auth_config

        if self._user_info is None:
            if auth_config.provider == "auth0":
                self._user_info = auth0.get_user_info(self.bearer_token)
            else:
                # For WorkOS we can only get the user info by refreshing the token
                self._auth_config, self._user_info = _fetch_auth_config(
                    force_refresh=True
                )
                if not self._user_info:
                    raise click.ClickException("Failed to get user info")

        return self._user_info

    @property
    def access_token(self) -> str:
        return self.auth_config.access_token

    @property
    def bearer_token(self) -> str:
        return "Bearer " + self.access_token


USER = UserAccess()
