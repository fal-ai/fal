from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from fal.toolkit.file.providers import fal as providers


class FakeRequest:
    """Fake request object with headers."""

    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}


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


def test_caller_cdn_token_header_no_current_app():
    """When there is no current app, headers should not be modified."""
    headers: dict[str, str] = {}

    with patch.object(providers, "get_current_app", return_value=None):
        providers._caller_cdn_token_header(headers)

    assert "X-Fal-CDN-Token" not in headers


def test_caller_cdn_token_header_no_request():
    """When current app has no request, headers should not be modified."""
    headers: dict[str, str] = {}
    fake_app = FakeApp(current_request=None)

    with patch.object(providers, "get_current_app", return_value=fake_app):
        providers._caller_cdn_token_header(headers)

    assert "X-Fal-CDN-Token" not in headers


def test_caller_cdn_token_header_no_cdn_token():
    """When request has no cdn token header, headers should not be modified."""
    headers: dict[str, str] = {}
    fake_request = FakeRequest(headers={"other-header": "value"})
    fake_app = FakeApp(current_request=fake_request)

    with patch.object(providers, "get_current_app", return_value=fake_app):
        providers._caller_cdn_token_header(headers)

    assert "X-Fal-CDN-Token" not in headers


def test_caller_cdn_token_header_adds_token():
    """When request has cdn token header, it should be added to headers."""
    headers: dict[str, str] = {}
    cdn_token = "test-cdn-token-12345"
    fake_request = FakeRequest(headers={"x-fal-cdn-token": cdn_token})
    fake_app = FakeApp(current_request=fake_request)

    with patch.object(providers, "get_current_app", return_value=fake_app):
        providers._caller_cdn_token_header(headers)

    assert headers["X-Fal-CDN-Token"] == cdn_token


def test_caller_cdn_token_header_preserves_existing():
    """Existing headers should be preserved when adding cdn token."""
    headers = {"Authorization": "Bearer token", "User-Agent": "test"}
    cdn_token = "test-cdn-token"
    fake_request = FakeRequest(headers={"x-fal-cdn-token": cdn_token})
    fake_app = FakeApp(current_request=fake_request)

    with patch.object(providers, "get_current_app", return_value=fake_app):
        providers._caller_cdn_token_header(headers)

    assert headers["Authorization"] == "Bearer token"
    assert headers["User-Agent"] == "test"
    assert headers["X-Fal-CDN-Token"] == cdn_token


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


def test_fal_file_repository_v3_auth_headers_include_cdn_token():
    """FalFileRepositoryV3 auth_headers should include CDN token."""
    cdn_token = "test-cdn-token"
    fake_app = _create_fake_app_with_cdn_token(cdn_token)

    with patch.object(
        providers, "key_credentials", return_value=("key_id", "key_secret")
    ), patch.object(providers, "get_current_app", return_value=fake_app):
        repo = providers.FalFileRepositoryV3()
        headers = repo.auth_headers

    assert headers["X-Fal-CDN-Token"] == cdn_token
    assert headers["Authorization"] == "Key key_id:key_secret"


def test_multipart_upload_v3_auth_headers_include_cdn_token():
    """MultipartUploadV3 auth_headers should include CDN token."""
    cdn_token = "test-cdn-token"
    fake_app = _create_fake_app_with_cdn_token(cdn_token)

    with patch.object(
        providers, "key_credentials", return_value=("key_id", "key_secret")
    ), patch.object(providers, "get_current_app", return_value=fake_app):
        multipart = providers.MultipartUploadV3("test.txt")
        headers = multipart.auth_headers

    assert headers["X-Fal-CDN-Token"] == cdn_token
    assert headers["Authorization"] == "Key key_id:key_secret"


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
