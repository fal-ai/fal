from urllib.parse import unquote

from fal.auth.auth0 import _build_device_login_url


def test_build_device_login_url():
    url = _build_device_login_url(
        user_code="ABCD-EFGH",
        verification_uri_complete="https://auth.fal.ai/activate?user_code=ABCD-EFGH",
        connection="github",
    )

    # Should route through Auth0 logout → fal.ai/login/cli
    assert "auth.fal.ai/v2/logout" in url
    decoded = unquote(url)
    assert "fal.ai/login/cli" in decoded
    assert "connection=github" in decoded
    assert "user_code=ABCD-EFGH" in decoded
