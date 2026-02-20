from __future__ import annotations

from unittest import mock
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from fal.toolkit.file.providers.fal import _maybe_retry_request


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


def test_successful_request_no_retry():
    request = Request("https://example.com/test")

    with mock.patch(URLOPEN) as mock_urlopen:
        mock_urlopen.return_value = MockResponse()

        with _maybe_retry_request(request) as response:
            assert response is not None

        assert mock_urlopen.call_count == 1


def test_retry_on_retryable_http_error():
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


def test_no_retry_on_non_retryable_http_error():
    request = Request("https://example.com/test")
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise HTTPError("https://example.com/test", 400, "Bad Request", {}, None)

    with mock.patch(URLOPEN, side_effect=side_effect):
        with pytest.raises(HTTPError) as exc_info:
            with _maybe_retry_request(request):
                pass

    assert exc_info.value.code == 400
    assert call_count == 1


def test_non_retryable_exception_not_retried():
    request = Request("https://example.com/test")
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise ValueError("unexpected")

    with mock.patch(URLOPEN, side_effect=side_effect):
        with pytest.raises(ValueError):
            with _maybe_retry_request(request):
                pass

    assert call_count == 1


def test_max_retries_exhausted():
    request = Request("https://example.com/test")
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise HTTPError("https://example.com/test", 500, "Server Error", {}, None)

    with mock.patch(URLOPEN, side_effect=side_effect):
        with mock.patch(SLEEP):
            with pytest.raises(HTTPError):
                with _maybe_retry_request(request):
                    pass

    assert call_count == 5


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
