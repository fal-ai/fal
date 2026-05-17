from unittest.mock import MagicMock, call, patch
from urllib.parse import unquote

from fal.auth.auth0 import _build_device_login_url, build_jwk_client, login


def test_build_device_login_url():
    url = _build_device_login_url(
        user_code="ABCD-EFGH",
        verification_uri_complete="https://auth.fal.ai/activate?user_code=ABCD-EFGH",
        connection="github",
    )

    assert url.startswith("https://fal.ai/api/auth/cli/session-seed?")
    decoded = unquote(url)
    assert "/api/auth/cli/session-seed" in decoded
    assert "connection=github" in decoded
    assert "user_code=ABCD-EFGH" in decoded


def test_build_jwk_client_uses_certifi_ssl_context():
    build_jwk_client.cache_clear()

    try:
        with patch("certifi.where", return_value="certifi.pem"), patch(
            "ssl.create_default_context"
        ) as create_default_context, patch("jwt.PyJWKClient") as py_jwk_client:
            ssl_context = create_default_context.return_value

            build_jwk_client()

        create_default_context.assert_called_once_with(cafile="certifi.pem")
        py_jwk_client.assert_called_once_with(
            "https://auth.fal.ai/.well-known/jwks.json",
            cache_keys=True,
            ssl_context=ssl_context,
        )
    finally:
        build_jwk_client.cache_clear()


def test_login_with_connection_completes_device_flow():
    console = MagicMock()
    console.status.return_value.__enter__.return_value = MagicMock()
    device_response = MagicMock(
        status_code=200,
        json=MagicMock(
            return_value={
                "device_code": "device-code",
                "user_code": "ABCD-EFGH",
                "verification_uri_complete": (
                    "https://auth.fal.ai/activate?user_code=ABCD-EFGH"
                ),
                "interval": 1,
            }
        ),
    )
    pending_response = MagicMock(
        status_code=403,
        json=MagicMock(return_value={"error": "authorization_pending"}),
    )
    token_data = {
        "access_token": "access-token",
        "refresh_token": "refresh-token",
        "id_token": "id-token",
    }
    token_response = MagicMock(status_code=200, json=MagicMock(return_value=token_data))

    with patch(
        "httpx.post",
        side_effect=[device_response, pending_response, token_response],
    ) as post, patch("fal.auth.auth0._open_browser") as open_browser, patch(
        "fal.auth.auth0.validate_id_token"
    ) as validate_id_token, patch("fal.auth.auth0.time.sleep") as sleep:
        assert login(console, connection="github", no_browser=True) == token_data

    assert post.call_args_list == [
        call(
            "https://auth.fal.ai/oauth/device/code",
            data={
                "audience": "fal-cloud",
                "client_id": "TwXR51Vz8JbY8GUUMy6EyuVR0fTO7N4N",
                "scope": "openid profile email offline_access",
            },
        ),
        call(
            "https://auth.fal.ai/oauth/token",
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": "device-code",
                "client_id": "TwXR51Vz8JbY8GUUMy6EyuVR0fTO7N4N",
            },
        ),
        call(
            "https://auth.fal.ai/oauth/token",
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": "device-code",
                "client_id": "TwXR51Vz8JbY8GUUMy6EyuVR0fTO7N4N",
            },
        ),
    ]
    opened_url = open_browser.call_args.args[0]
    assert opened_url.startswith("https://fal.ai/api/auth/cli/session-seed?")
    decoded = unquote(opened_url)
    assert "connection=github" in decoded
    assert "user_code=ABCD-EFGH" in decoded
    open_browser.assert_called_once_with(
        opened_url,
        "ABCD-EFGH",
        console,
        no_browser=True,
    )
    validate_id_token.assert_called_once_with("id-token")
    sleep.assert_called_once_with(1)
