from __future__ import annotations

from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from fal.toolkit.file.providers.fal import _maybe_retry_request

RETRY_CODES = [408, 409, 429, 500, 502, 503, 504]
NON_RETRY_CODES = [400, 401, 403, 404, 422]


class MockResponse:
    """Mock response object that mimics urllib.response.addinfourl"""

    def __init__(self, data: str = '{"result": "success"}', status: int = 200):
        self.data = data.encode()
        self.status = status
        self.headers = {"Content-Type": "application/json"}

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def test_successful_request_no_retry():
    request = Request("https://example.com/test")

    with mock.patch("fal.toolkit.file.providers.fal._urlopen") as mock_urlopen:
        mock_response = MockResponse()
        mock_context = mock.MagicMock()
        mock_context.__enter__.return_value = mock_response
        mock_context.__exit__.return_value = None
        mock_urlopen.return_value = mock_context

        with _maybe_retry_request(request) as response:
            assert response == mock_response

        assert mock_urlopen.call_count == 1


@pytest.mark.parametrize("error_code", RETRY_CODES)
def test_retry_on_retryable_http_codes(error_code):
    request = Request("https://example.com/test")

    call_count = 0

    def mock_urlopen_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            raise HTTPError(
                url="https://example.com/test",
                code=error_code,
                msg=f"HTTP {error_code} Error",
                hdrs={},
                fp=None,
            )
        else:
            mock_response = MockResponse()
            mock_context = mock.MagicMock()
            mock_context.__enter__.return_value = mock_response
            mock_context.__exit__.return_value = None
            return mock_context

    with mock.patch(
        "fal.toolkit.file.providers.fal._urlopen", side_effect=mock_urlopen_side_effect
    ):
        with mock.patch(
            "fal.toolkit.utils.retry.time.sleep"
        ):  # Mock sleep to speed up tests
            with _maybe_retry_request(request) as response:
                assert response is not None

    assert call_count == 2


@pytest.mark.parametrize("error_code", NON_RETRY_CODES)
def test_no_retry_on_non_retryable_http_codes(error_code):
    request = Request("https://example.com/test")

    call_count = 0

    def mock_urlopen_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise HTTPError(
            url="https://example.com/test",
            code=error_code,
            msg=f"HTTP {error_code} Error",
            hdrs={},
            fp=None,
        )

    with mock.patch(
        "fal.toolkit.file.providers.fal._urlopen", side_effect=mock_urlopen_side_effect
    ):
        with pytest.raises(HTTPError) as exc_info:
            with _maybe_retry_request(request):
                pass

        assert exc_info.value.code == error_code
        assert call_count == 1


def test_non_http_exception_not_retried():
    """Test that non-HTTP exceptions are not retried"""
    request = Request("https://example.com/test")

    call_count = 0

    def mock_urlopen_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise ValueError("Network error")

    with mock.patch(
        "fal.toolkit.file.providers.fal._urlopen", side_effect=mock_urlopen_side_effect
    ):
        with pytest.raises(ValueError) as exc_info:
            with _maybe_retry_request(request):
                pass

        assert str(exc_info.value) == "Network error"
        assert call_count == 1


def test_max_retries_exhausted_for_retryable_errors():
    """Test that retries are exhausted after MAX_ATTEMPTS for retryable HTTP errors"""
    request = Request("https://example.com/test")

    call_count = 0

    def mock_urlopen_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # Always fail with a retryable error
        raise HTTPError(
            url="https://example.com/test",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=None,
        )

    with mock.patch(
        "fal.toolkit.file.providers.fal._urlopen", side_effect=mock_urlopen_side_effect
    ):
        with mock.patch(
            "fal.toolkit.utils.retry.time.sleep"
        ):  # Mock sleep to speed up tests
            with pytest.raises(HTTPError) as exc_info:
                with _maybe_retry_request(request):
                    pass

        assert exc_info.value.code == 500
        assert call_count == 5
