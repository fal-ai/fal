from unittest.mock import patch

import pytest

from fal.sdk import FalServerlessKeyCredentials, get_credentials


def test_get_credentials_returns_key_credentials_when_available():
    with patch("fal.sdk.key_credentials", return_value=("key-id", "key-secret")):
        with patch("fal.config.Config") as mock_config_class:
            mock_config_class.return_value.get_internal.return_value = None
            credentials = get_credentials()

    assert isinstance(credentials, FalServerlessKeyCredentials)
    assert credentials.key_id == "key-id"
    assert credentials.key_secret == "key-secret"


def test_get_credentials_prints_when_team_differs_from_key_owner_nickname():
    message = (
        "Ignoring explicit team config-team because key is used. "
        "The key belongs to Test User: my-nickname - usr_123."
    )
    with patch("fal.sdk.key_credentials", return_value=("key-id", "key-secret")):
        with patch(
            "fal.sdk.current_user_info",
            return_value={
                "full_name": "Test User",
                "nickname": "my-nickname",
                "user_id": "usr_123",
            },
        ):
            with patch("fal.sdk.console.print") as mock_console_print:
                credentials = get_credentials(team="config-team")

    mock_console_print.assert_called_once_with(message)
    assert isinstance(credentials, FalServerlessKeyCredentials)


def test_get_credentials_does_not_print_when_team_matches_nickname():
    with patch("fal.sdk.key_credentials", return_value=("key-id", "key-secret")):
        with patch(
            "fal.sdk.current_user_info",
            return_value={
                "full_name": "Test User",
                "nickname": "my-team",
                "user_id": "usr_123",
            },
        ):
            with patch("fal.sdk.console.print") as mock_console_print:
                credentials = get_credentials(team="my-team")

    mock_console_print.assert_not_called()
    assert isinstance(credentials, FalServerlessKeyCredentials)


def test_get_credentials_raises_when_key_owner_lookup_fails():
    with patch("fal.sdk.key_credentials", return_value=("key-id", "key-secret")):
        with patch("fal.sdk.current_user_info", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                get_credentials(team="my-team")


def test_get_credentials_passes_profile_to_key_credentials():
    with patch("fal.sdk.key_credentials", return_value=None) as mock_key_credentials:
        with patch("fal.config.Config") as mock_config_class:
            mock_config_class.return_value.get_internal.return_value = None
            get_credentials(profile="my-profile")

    mock_key_credentials.assert_called_once_with(profile="my-profile")
