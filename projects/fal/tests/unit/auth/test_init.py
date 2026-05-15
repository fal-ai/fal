from unittest.mock import patch

import pytest

from fal.auth import AuthCredentials, fetch_auth_credentials, login, logout
from fal.exceptions import FalServerlessException
from fal.exceptions.auth import UnauthenticatedException


def test_auth_credentials_header_value():
    assert AuthCredentials("Key", "id:secret").header_value == "Key id:secret"
    assert AuthCredentials("Bearer", "jwt").header_value == "Bearer jwt"


def test_fetch_auth_credentials_returns_key_creds_when_available():
    with patch(
        "fal.auth.key_credentials", return_value=("kid", "ksecret")
    ) as key_mock, patch("fal.auth._fetch_access_token") as token_mock:
        auth = fetch_auth_credentials()

    assert auth == AuthCredentials("Key", "kid:ksecret")
    key_mock.assert_called_once_with()
    # bearer path should not be reached when key creds are present
    token_mock.assert_not_called()


def test_fetch_auth_credentials_falls_back_to_bearer_token():
    with patch("fal.auth.key_credentials", return_value=None), patch(
        "fal.auth._fetch_access_token", return_value="jwt-token"
    ):
        auth = fetch_auth_credentials()

    assert auth == AuthCredentials("Bearer", "jwt-token")


def test_fetch_auth_credentials_propagates_unauthenticated():
    with patch("fal.auth.key_credentials", return_value=None), patch(
        "fal.auth._fetch_access_token", side_effect=UnauthenticatedException()
    ):
        with pytest.raises(UnauthenticatedException):
            fetch_auth_credentials()


def test_login_saves_refresh_token():
    console = object()
    token_data = {"refresh_token": "refresh-token", "access_token": "access-token"}

    with patch("fal.auth.auth0.login", return_value=token_data) as auth0_login, patch(
        "fal.auth.local.lock_token"
    ) as lock_token, patch("fal.auth.local.save_token") as save_token:
        login(console, connection="github", no_browser=True)

    auth0_login.assert_called_once_with(
        console,
        connection="github",
        no_browser=True,
    )
    lock_token.assert_called_once_with()
    save_token.assert_called_once_with("refresh-token")


def test_logout_revokes_and_deletes_refresh_token():
    console = object()

    with patch(
        "fal.auth.local.load_token", return_value=("refresh-token", "access-token")
    ), patch("fal.auth.auth0.revoke") as revoke, patch(
        "fal.auth.local.lock_token"
    ) as lock_token, patch("fal.auth.local.delete_token") as delete_token:
        logout(console, no_browser=True)

    revoke.assert_called_once_with("refresh-token", console, no_browser=True)
    lock_token.assert_called_once_with()
    delete_token.assert_called_once_with()


def test_logout_raises_when_not_logged_in():
    with patch("fal.auth.local.load_token", return_value=(None, None)):
        with pytest.raises(FalServerlessException, match="You're not logged in"):
            logout(object())
