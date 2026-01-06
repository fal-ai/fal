import base64
import json
import time
from pathlib import Path


from fal_client.auth import AuthCredentials, fetch_auth_credentials


def _make_jwt(expires_in_seconds: int) -> str:
    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "none", "typ": "JWT"}).encode("utf-8")
    ).rstrip(b"=")
    payload = base64.urlsafe_b64encode(
        json.dumps({"exp": int(time.time()) + expires_in_seconds}).encode("utf-8")
    ).rstrip(b"=")
    # Signature is unused because we don't verify it.
    return f"{header.decode()}.{payload.decode()}."


def _write_auth_file(tmp_path: Path, refresh_token: str, access_token: str) -> Path:
    path = tmp_path / "auth0_token"
    path.write_text(f"{refresh_token}\n{access_token}")
    return path


def test_fetch_auth_credentials_prefers_key(monkeypatch, tmp_path):
    monkeypatch.setenv("FAL_HOME_DIR", str(tmp_path))
    monkeypatch.setenv("FAL_KEY", "abc123")
    monkeypatch.delenv("FAL_KEY_ID", raising=False)
    monkeypatch.delenv("FAL_KEY_SECRET", raising=False)

    auth = fetch_auth_credentials()

    assert auth == AuthCredentials("Key", "abc123")


def test_fetch_auth_credentials_uses_login_token_without_refresh(
    monkeypatch, tmp_path, mocker
):
    monkeypatch.setenv("FAL_HOME_DIR", str(tmp_path))
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.delenv("FAL_KEY_ID", raising=False)
    monkeypatch.delenv("FAL_KEY_SECRET", raising=False)

    access_token = _make_jwt(expires_in_seconds=3600)
    _write_auth_file(tmp_path, "refresh-token", access_token)

    httpx_post = mocker.patch("fal_client.auth.httpx.post")

    auth = fetch_auth_credentials()

    assert auth == AuthCredentials("Bearer", access_token)
    httpx_post.assert_not_called()


def test_fetch_auth_credentials_refreshes_expired_token(monkeypatch, tmp_path, mocker):
    monkeypatch.setenv("FAL_HOME_DIR", str(tmp_path))
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.delenv("FAL_KEY_ID", raising=False)
    monkeypatch.delenv("FAL_KEY_SECRET", raising=False)

    expired_token = _make_jwt(expires_in_seconds=-3600)
    token_file = _write_auth_file(tmp_path, "old-refresh-token", expired_token)

    new_access_token = _make_jwt(expires_in_seconds=7200)
    response = mocker.Mock()
    response.status_code = 200
    response.json.return_value = {
        "access_token": new_access_token,
        "refresh_token": "new-refresh-token",
    }
    httpx_post = mocker.patch("fal_client.auth.httpx.post", return_value=response)

    auth = fetch_auth_credentials()

    assert auth == AuthCredentials("Bearer", new_access_token)
    httpx_post.assert_called_once()

    saved = token_file.read_text().splitlines()
    assert saved[0] == "new-refresh-token"
    assert saved[1] == new_access_token
