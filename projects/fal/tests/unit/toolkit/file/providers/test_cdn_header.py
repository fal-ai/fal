from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from fal.auth import AuthCredentials
from fal.toolkit.file.providers import fal as providers


class FakeRequest:
    """Fake request object with headers."""

    def __init__(
        self, headers: dict[str, str] | None = None, request_id: str | None = None
    ):
        self.headers = headers or {}
        self.request_id = request_id


class FakeApp:
    """Fake app object for testing."""

    def __init__(self, current_request: FakeRequest | None = None):
        self.current_request = current_request


def _create_fake_app_with_cdn_token(cdn_token: str | None = None) -> FakeApp:
    """Helper to create a fake app with optional CDN token."""
    if cdn_token:
        fake_request = FakeRequest(headers={"x-fal-cdn-token": cdn_token})
    else:
        fake_request = FakeRequest(headers={})
    return FakeApp(current_request=fake_request)


def _create_fake_app_with_request_id(request_id: str | None = None) -> FakeApp:
    """Helper to create a fake app with optional request ID."""
    fake_request = FakeRequest(headers={}, request_id=request_id)
    return FakeApp(current_request=fake_request)


def _create_fake_app(
    cdn_token: str | None = None, request_id: str | None = None
) -> FakeApp:
    """Helper to create a fake app with optional CDN token and request ID."""
    headers = {"x-fal-cdn-token": cdn_token} if cdn_token else {}
    fake_request = FakeRequest(headers=headers, request_id=request_id)
    return FakeApp(current_request=fake_request)


def test_caller_cdn_header_no_current_app():
    """When there is no current app, headers should not be modified."""
    headers: dict[str, str] = {}

    with patch.object(providers, "get_current_app", return_value=None):
        providers._caller_cdn_header(headers)

    assert "X-Fal-CDN-Token" not in headers


def test_caller_cdn_header_no_request():
    """When current app has no request, headers should not be modified."""
    headers: dict[str, str] = {}
    fake_app = FakeApp(current_request=None)

    with patch.object(providers, "get_current_app", return_value=fake_app):
        providers._caller_cdn_header(headers)

    assert "X-Fal-CDN-Token" not in headers


def test_caller_cdn_header_no_cdn_token():
    """When request has no cdn token header, headers should not be modified."""
    headers: dict[str, str] = {}
    fake_request = FakeRequest(headers={"other-header": "value"})
    fake_app = FakeApp(current_request=fake_request)

    with patch.object(providers, "get_current_app", return_value=fake_app):
        providers._caller_cdn_header(headers)

    assert "X-Fal-CDN-Token" not in headers


def test_caller_cdn_header_adds_token():
    """When request has cdn token header, it should be added to headers."""
    headers: dict[str, str] = {}
    cdn_token = "test-cdn-token-12345"
    fake_request = FakeRequest(headers={"x-fal-cdn-token": cdn_token})
    fake_app = FakeApp(current_request=fake_request)

    with patch.object(providers, "get_current_app", return_value=fake_app):
        providers._caller_cdn_header(headers)

    assert headers["X-Fal-CDN-Token"] == cdn_token


def test_caller_cdn_header_preserves_existing():
    """Existing headers should be preserved when adding cdn token."""
    headers = {"Authorization": "Bearer token", "User-Agent": "test"}
    cdn_token = "test-cdn-token"
    fake_request = FakeRequest(headers={"x-fal-cdn-token": cdn_token})
    fake_app = FakeApp(current_request=fake_request)

    with patch.object(providers, "get_current_app", return_value=fake_app):
        providers._caller_cdn_header(headers)

    assert headers["Authorization"] == "Bearer token"
    assert headers["User-Agent"] == "test"
    assert headers["X-Fal-CDN-Token"] == cdn_token


def test_caller_cdn_header_adds_request_id():
    """When request has request_id, it should be added to headers."""
    headers: dict[str, str] = {}
    request_id = "test-request-id-12345"
    fake_app = _create_fake_app_with_request_id(request_id)

    with patch.object(providers, "get_current_app", return_value=fake_app):
        providers._caller_cdn_header(headers)

    assert headers["X-Fal-Request-ID"] == request_id


def test_caller_cdn_header_no_request_id_when_not_present():
    """When request has no request_id, it should not be added to headers."""
    headers: dict[str, str] = {}
    fake_app = _create_fake_app_with_request_id(None)

    with patch.object(providers, "get_current_app", return_value=fake_app):
        providers._caller_cdn_header(headers)

    assert "X-Fal-Request-ID" not in headers


def test_caller_cdn_header_adds_both_cdn_token_and_request_id():
    """When request has both cdn token and request_id, both should be added."""
    headers: dict[str, str] = {}
    cdn_token = "test-cdn-token"
    request_id = "test-request-id-67890"
    fake_app = _create_fake_app(cdn_token=cdn_token, request_id=request_id)

    with patch.object(providers, "get_current_app", return_value=fake_app):
        providers._caller_cdn_header(headers)

    assert headers["X-Fal-CDN-Token"] == cdn_token
    assert headers["X-Fal-Request-ID"] == request_id


@pytest.mark.parametrize(
    "repo_cls,token_manager,token",
    [
        (
            providers.FalFileRepository,
            providers.fal_v3_token_manager,
            providers.FalV3Token(
                token="test-token",
                token_type="Bearer",
                base_upload_url="https://upload",
                expires_at=datetime.now(timezone.utc),
            ),
        ),
        (
            providers.InternalFalFileRepositoryV3,
            providers.fal_v3_token_manager,
            providers.FalV3Token(
                token="test-token",
                token_type="Bearer",
                base_upload_url="https://upload",
                expires_at=datetime.now(timezone.utc),
            ),
        ),
        (
            providers.FalCDNFileRepository,
            providers.fal_v3_token_manager,
            providers.FalV3Token(
                token="test-token",
                token_type="Bearer",
                base_upload_url="https://upload",
                expires_at=datetime.now(timezone.utc),
            ),
        ),
    ],
)
def test_repository_auth_headers_include_cdn_token(repo_cls, token_manager, token):
    """Repository auth_headers should include CDN token when available."""
    cdn_token = "test-cdn-token"
    fake_app = _create_fake_app_with_cdn_token(cdn_token)

    with patch.object(token_manager, "get_token", return_value=token), patch.object(
        providers, "get_current_app", return_value=fake_app
    ):
        repo = repo_cls()
        headers = repo.auth_headers

    assert headers["X-Fal-CDN-Token"] == cdn_token
    assert "Authorization" in headers
    assert headers["User-Agent"] == providers.USER_AGENT


@pytest.mark.parametrize(
    "repo_cls,token_manager,token",
    [
        (
            providers.FalFileRepository,
            providers.fal_v3_token_manager,
            providers.FalV3Token(
                token="test-token",
                token_type="Bearer",
                base_upload_url="https://upload",
                expires_at=datetime.now(timezone.utc),
            ),
        ),
        (
            providers.InternalFalFileRepositoryV3,
            providers.fal_v3_token_manager,
            providers.FalV3Token(
                token="test-token",
                token_type="Bearer",
                base_upload_url="https://upload",
                expires_at=datetime.now(timezone.utc),
            ),
        ),
        (
            providers.FalCDNFileRepository,
            providers.fal_v3_token_manager,
            providers.FalV3Token(
                token="test-token",
                token_type="Bearer",
                base_upload_url="https://upload",
                expires_at=datetime.now(timezone.utc),
            ),
        ),
    ],
)
def test_repository_auth_headers_no_cdn_token_when_not_present(
    repo_cls, token_manager, token
):
    """Repository auth_headers should not include CDN token when not present."""
    with patch.object(token_manager, "get_token", return_value=token), patch.object(
        providers, "get_current_app", return_value=None
    ):
        repo = repo_cls()
        headers = repo.auth_headers

    assert "X-Fal-CDN-Token" not in headers
    assert "Authorization" in headers
    assert headers["User-Agent"] == providers.USER_AGENT


def test_fal_file_repository_v3_auth_headers_include_cdn_token():
    """FalFileRepositoryV3 auth_headers should include CDN token."""
    cdn_token = "test-cdn-token"
    fake_app = _create_fake_app_with_cdn_token(cdn_token)

    with patch.object(
        providers,
        "fetch_auth_credentials",
        return_value=AuthCredentials("Key", "key_id:key_secret"),
    ), patch.object(providers, "get_current_app", return_value=fake_app):
        repo = providers.FalFileRepositoryV3()
        headers = repo.auth_headers

    assert headers["X-Fal-CDN-Token"] == cdn_token
    assert headers["Authorization"] == "Key key_id:key_secret"
    assert headers["User-Agent"] == providers.USER_AGENT


def test_multipart_upload_v3_auth_headers_include_cdn_token():
    """MultipartUploadV3 auth_headers should include CDN token."""
    cdn_token = "test-cdn-token"
    fake_app = _create_fake_app_with_cdn_token(cdn_token)

    with patch.object(
        providers,
        "fetch_auth_credentials",
        return_value=AuthCredentials("Key", "key_id:key_secret"),
    ), patch.object(providers, "get_current_app", return_value=fake_app):
        multipart = providers.MultipartUploadV3("test.txt")
        headers = multipart.auth_headers

    assert headers["X-Fal-CDN-Token"] == cdn_token
    assert headers["Authorization"] == "Key key_id:key_secret"
    assert headers["User-Agent"] == providers.USER_AGENT


def test_fal_file_repository_v3_auth_headers_with_bearer_credentials():
    """FalFileRepositoryV3 forwards a bearer token verbatim as `Bearer <jwt>`."""
    with patch.object(
        providers,
        "fetch_auth_credentials",
        return_value=AuthCredentials("Bearer", "jwt-token"),
    ), patch.object(providers, "get_current_app", return_value=None):
        repo = providers.FalFileRepositoryV3()
        headers = repo.auth_headers

    assert headers["Authorization"] == "Bearer jwt-token"


def test_internal_multipart_upload_v3_auth_headers_include_cdn_token():
    """InternalMultipartUploadV3 auth_headers should include CDN token."""
    cdn_token = "test-cdn-token"
    fake_app = _create_fake_app_with_cdn_token(cdn_token)
    token = providers.FalV3Token(
        token="test-token",
        token_type="Bearer",
        base_upload_url="https://upload",
        expires_at=datetime.now(timezone.utc),
    )

    with patch.object(
        providers.fal_v3_token_manager, "get_token", return_value=token
    ), patch.object(providers, "get_current_app", return_value=fake_app):
        multipart = providers.InternalMultipartUploadV3("test.txt")
        headers = multipart.auth_headers

    assert headers["X-Fal-CDN-Token"] == cdn_token
    assert "Authorization" in headers
    assert headers["User-Agent"] == providers.USER_AGENT


@pytest.mark.parametrize(
    "host",
    [
        "https://fal.media",
        "fal.media",
        "https://fal.media/files",
        "https://v2.fal.media",
        "http://v2.fal.media:443",
    ],
)
def test_fal_cdn_host_env_warns_for_legacy_hosts(monkeypatch, host):
    monkeypatch.setenv("FAL_CDN_HOST", host)
    with pytest.warns(DeprecationWarning, match="FAL_CDN_HOST"):
        providers._warn_if_legacy_cdn_host_set()


@pytest.mark.parametrize(
    "host",
    ["https://v3.fal.media", "https://my-proxy.example.com", "my-proxy.example.com"],
)
def test_fal_cdn_host_env_silent_for_supported_hosts(monkeypatch, recwarn, host):
    monkeypatch.setenv("FAL_CDN_HOST", host)
    providers._warn_if_legacy_cdn_host_set()
    assert not [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]


def test_fal_cdn_host_env_silent_when_unset(monkeypatch, recwarn):
    monkeypatch.delenv("FAL_CDN_HOST", raising=False)
    providers._warn_if_legacy_cdn_host_set()
    assert not [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]
