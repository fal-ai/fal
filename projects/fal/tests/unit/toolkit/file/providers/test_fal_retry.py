from __future__ import annotations

from functools import partial
from io import BytesIO
from types import SimpleNamespace
from unittest import mock
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from fal.toolkit.exceptions import FileUploadException
from fal.toolkit.file.providers import fal as providers
from fal.toolkit.file.providers.fal import _maybe_retry_request
from fal.toolkit.file.types import FileData


class MockResponse:
    """Mock response object that mimics urllib.response.addinfourl"""

    def __init__(self, data: str = '{"result": "success"}', status: int = 200):
        self.data = data.encode()
        self.status = status
        self.headers = {"Content-Type": "application/json"}

    def read(self):
        return self.data

    def close(self):
        pass


URLOPEN = "fal.toolkit.file.providers.fal.urlopen"
SLEEP = "fal.toolkit.utils.retry.time.sleep"


def test_retries_on_transient_error():
    """Smoke test: retry is wired up — a transient 500 is retried and succeeds."""
    request = Request("https://example.com/test")
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise HTTPError("https://example.com/test", 500, "Server Error", {}, None)
        return MockResponse()

    with mock.patch(URLOPEN, side_effect=side_effect):
        with mock.patch(SLEEP):
            with _maybe_retry_request(request) as response:
                assert response is not None

    assert call_count == 2


def test_retry_on_url_error():
    request = Request("https://example.com/test")
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise URLError("[SSL: UNEXPECTED_EOF_WHILE_READING]")
        return MockResponse()

    with mock.patch(URLOPEN, side_effect=side_effect):
        with mock.patch(SLEEP):
            with _maybe_retry_request(request) as response:
                assert response is not None

    assert call_count == 3


def test_retry_on_timeout_error():
    request = Request("https://example.com/test")
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise TimeoutError("Connection timed out")
        return MockResponse()

    with mock.patch(URLOPEN, side_effect=side_effect):
        with mock.patch(SLEEP):
            with _maybe_retry_request(request) as response:
                assert response is not None

    assert call_count == 2


def test_no_retry_on_client_error():
    """_should_retry negative path: a 400 is not retried."""
    request = Request("https://example.com/test")

    with mock.patch(URLOPEN) as mock_urlopen:
        mock_urlopen.side_effect = HTTPError(
            "https://example.com/test", 400, "Bad Request", {}, None
        )

        with pytest.raises(HTTPError) as exc_info:
            with _maybe_retry_request(request):
                pass

    assert exc_info.value.code == 400
    assert mock_urlopen.call_count == 1


def test_response_closed_on_caller_exception():
    """response.close() is called even when the caller raises inside the with block."""
    request = Request("https://example.com/test")

    with mock.patch(URLOPEN) as mock_urlopen:
        mock_response = MockResponse()
        mock_response.close = mock.Mock()
        mock_urlopen.return_value = mock_response

        with pytest.raises(RuntimeError):
            with _maybe_retry_request(request):
                raise RuntimeError("unexpected error occurred")

        mock_response.close.assert_called_once()


def test_custom_timeout_forwarded():
    request = Request("https://example.com/test")

    with mock.patch(URLOPEN) as mock_urlopen:
        mock_urlopen.return_value = MockResponse()

        with _maybe_retry_request(request, timeout=300) as response:
            assert response is not None

        mock_urlopen.assert_called_once_with(request, timeout=300)


def test_default_timeout():
    request = Request("https://example.com/test")

    with mock.patch(URLOPEN) as mock_urlopen:
        mock_urlopen.return_value = MockResponse()

        with _maybe_retry_request(request) as response:
            assert response is not None

        mock_urlopen.assert_called_once_with(request, timeout=10)


@pytest.fixture
def upload_auth(monkeypatch):
    monkeypatch.setattr(
        providers,
        "_require_auth_credentials",
        lambda: SimpleNamespace(header_value="test-key"),
    )
    monkeypatch.setattr(
        providers.fal_v3_token_manager,
        "get_token",
        lambda: SimpleNamespace(
            token_type="Bearer",
            token="test-token",
            base_upload_url="https://example.com",
        ),
    )


@pytest.mark.parametrize(
    ("repository_name", "storage_type", "multipart"),
    [
        ("FalFileRepository", "gcs", False),
        ("FalFileRepositoryV3", "fal-cdn-v3", False),
        ("InternalFalFileRepositoryV3", "fal-cdn-v3", False),
        ("MultipartUploadGCS", "gcs", True),
        ("MultipartUploadV3", "fal-cdn-v3", True),
        ("InternalMultipartUploadV3", "fal-cdn-v3", True),
    ],
)
def test_initiation_errors_include_server_response(
    upload_auth, repository_name, storage_type, multipart
):
    repository_type = getattr(providers, repository_name)
    if multipart:
        upload = repository_type("image.png", content_type="image/png").create
    else:
        upload = partial(
            repository_type().save,
            FileData(b"image", "image/png", "image.png"),
            multipart=False,
        )

    error = HTTPError(
        "https://example.com/upload?signature=test-signature",
        400,
        "Bad Request",
        {},
        BytesIO(b'{"detail":"Invalid storage type"}'),
    )
    with mock.patch(URLOPEN, side_effect=error) as mock_urlopen:
        with pytest.raises(FileUploadException) as exc_info:
            upload()

    assert str(exc_info.value) == (
        "Error initiating upload. Status 400: Bad Request. "
        f"Storage type: {storage_type}. "
        'Server response: {"detail":"Invalid storage type"}'
    )
    assert exc_info.value.__cause__ is error
    assert "test-key" not in str(exc_info.value)
    assert "test-token" not in str(exc_info.value)
    assert "test-signature" not in str(exc_info.value)
    assert mock_urlopen.call_count == 1


@pytest.mark.parametrize(
    ("body", "response_text"),
    [
        (b"Invalid storage type", "Invalid storage type"),
        (b'{"detail":', '{"detail":'),
        (b"Invalid \xff format", "Invalid \ufffd format"),
        (b"", None),
        (b" \n\t ", None),
        (None, None),
    ],
)
def test_initiation_error_body_formats(upload_auth, body, response_text):
    error = HTTPError(
        "https://example.com/upload",
        400,
        "Bad Request",
        {},
        BytesIO(body) if body is not None else None,
    )
    with mock.patch(URLOPEN, side_effect=error):
        with pytest.raises(FileUploadException) as exc_info:
            providers.FalFileRepositoryV3().save(FileData(b"image"), multipart=False)

    message = (
        "Error initiating upload. Status 400: Bad Request. Storage type: fal-cdn-v3"
    )
    if response_text is not None:
        message += f". Server response: {response_text}"
    assert str(exc_info.value) == message
    assert exc_info.value.__cause__ is error


def test_initiation_error_body_is_bounded(upload_auth):
    body = BytesIO(b"x" * (providers.MAX_ERROR_RESPONSE_BYTES + 100))
    error = HTTPError("https://example.com/upload", 400, "Bad Request", {}, body)

    with mock.patch(URLOPEN, side_effect=error):
        with pytest.raises(FileUploadException) as exc_info:
            providers.FalFileRepositoryV3().save(FileData(b"image"), multipart=False)

    assert str(exc_info.value).endswith(
        f"Server response: {'x' * providers.MAX_ERROR_RESPONSE_BYTES} [truncated]"
    )
    assert body.tell() == providers.MAX_ERROR_RESPONSE_BYTES + 1


def test_initiation_error_body_read_failure_preserves_upload_error(upload_auth):
    error = HTTPError("https://example.com/upload", 400, "Bad Request", {}, None)
    with mock.patch(URLOPEN, side_effect=error):
        with mock.patch.object(error, "read", side_effect=OSError("body unavailable")):
            with pytest.raises(FileUploadException) as exc_info:
                providers.FalFileRepositoryV3().save(
                    FileData(b"image"), multipart=False
                )

    assert str(exc_info.value) == (
        "Error initiating upload. Status 400: Bad Request. Storage type: fal-cdn-v3"
    )
    assert exc_info.value.__cause__ is error


@pytest.mark.parametrize(("status_code", "attempts"), [(400, 1), (422, 1), (503, 5)])
def test_initiation_diagnostics_preserve_retry_policy(
    upload_auth, status_code, attempts
):
    errors = [
        HTTPError(
            "https://example.com/upload",
            status_code,
            "Upload rejected",
            {},
            BytesIO(f"attempt {attempt}".encode()),
        )
        for attempt in range(1, attempts + 1)
    ]
    with mock.patch(URLOPEN, side_effect=errors) as mock_urlopen:
        with mock.patch(SLEEP):
            with pytest.raises(FileUploadException) as exc_info:
                providers.FalFileRepositoryV3().save(
                    FileData(b"image"), multipart=False
                )

    assert mock_urlopen.call_count == attempts
    assert str(exc_info.value).endswith(f"Server response: attempt {attempts}")
    assert exc_info.value.__cause__ is errors[-1]
