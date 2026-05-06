from unittest.mock import patch

import pytest

from fal.auth import AuthCredentials, fetch_auth_credentials
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
