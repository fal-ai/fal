from unittest.mock import patch

import pytest

from fal.api.client import SyncServerlessClient
from fal.sdk import FalServerlessKeyCredentials


def test_sync_client_warns_when_team_set_with_explicit_api_key():
    client = SyncServerlessClient(api_key="key-id:key-secret", team="config-team")
    warning_message = (
        "Ignoring explicit team config-team because key is used. "
        "The key belongs to Test User: my-nickname - usr_456."
    )

    with patch(
        "fal.sdk.current_user_info",
        return_value={
            "full_name": "Test User",
            "nickname": "my-nickname",
            "user_id": "usr_456",
        },
    ):
        with pytest.warns(
            UserWarning,
            match=warning_message,
        ):
            credentials = client._credentials

    assert isinstance(credentials, FalServerlessKeyCredentials)
    assert credentials.key_id == "key-id"
    assert credentials.key_secret == "key-secret"


def test_sync_client_invalid_api_key_format_raises_value_error():
    client = SyncServerlessClient(api_key="invalid-format")

    with pytest.raises(
        ValueError, match="api_key must be in 'KEY_ID:KEY_SECRET' format"
    ):
        _ = client._credentials


def test_sync_client_passes_profile_to_get_credentials():
    mock_credentials = object()
    with patch("fal.sdk.get_credentials", return_value=mock_credentials) as mock_get:
        client = SyncServerlessClient(team="team-a", profile="my-profile")
        credentials = client._credentials

    assert credentials is mock_credentials
    mock_get.assert_called_once_with(team="team-a", key=None, profile="my-profile")


def test_sync_client_passes_key_and_profile_to_get_credentials():
    mock_credentials = object()
    with patch("fal.sdk.get_credentials", return_value=mock_credentials) as mock_get:
        client = SyncServerlessClient(
            team="team-a", profile="my-profile", api_key="key-id:key-secret"
        )
        credentials = client._credentials

    assert credentials is mock_credentials
    mock_get.assert_called_once_with(
        team="team-a", key="key-id:key-secret", profile="my-profile"
    )
