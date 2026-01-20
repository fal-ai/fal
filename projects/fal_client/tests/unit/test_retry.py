"""Unit tests for retry logic in client.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx

from fal_client._headers import REQUEST_TIMEOUT_TYPE_HEADER
from fal_client.client import (
    RETRY_CODES,
    _is_ingress_error,
    _should_retry,
    _should_retry_response,
)


def _make_response(
    status_code: int,
    headers: dict[str, str] | None = None,
    text: str = "",
) -> httpx.Response:
    """Create a mock httpx.Response for testing."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.headers = headers or {}
    response.text = text
    return response


class TestIsIngressError:
    """Tests for _is_ingress_error function."""

    def test_non_ingress_status_code_returns_false(self) -> None:
        """Non-ingress status codes (not 502, 503, 504) return False."""
        response = _make_response(500)
        assert _is_ingress_error(response) is False

    def test_ingress_code_with_fal_request_id_returns_false(self) -> None:
        """Ingress codes with x-fal-request-id are from our server, not ingress."""
        response = _make_response(504, headers={"x-fal-request-id": "req-123"})
        assert _is_ingress_error(response) is False

    def test_ingress_code_with_nginx_in_body_returns_true(self) -> None:
        """Ingress codes with 'nginx' in body are ingress errors."""
        response = _make_response(502, text="<html>nginx error</html>")
        assert _is_ingress_error(response) is True

    def test_ingress_code_without_nginx_returns_false(self) -> None:
        """Ingress codes without 'nginx' in body are not ingress errors."""
        response = _make_response(503, text="Service unavailable")
        assert _is_ingress_error(response) is False


class TestShouldRetryResponse:
    """Tests for _should_retry_response function."""

    def test_ingress_error_with_user_timeout_type_does_not_retry(self) -> None:
        """Ingress error with REQUEST_TIMEOUT_TYPE_HEADER='user' should NOT retry."""
        response = _make_response(
            504,
            headers={REQUEST_TIMEOUT_TYPE_HEADER: "user"},
            text="nginx timeout",
        )
        assert _should_retry_response(response) is False

    def test_ingress_error_without_timeout_type_retries(self) -> None:
        """Ingress error without timeout type header should retry."""
        response = _make_response(502, text="nginx error")
        assert _should_retry_response(response) is True

    def test_retry_codes_always_retry(self) -> None:
        """Status codes in RETRY_CODES (408, 409, 429) should retry."""
        for code in RETRY_CODES:
            response = _make_response(code)
            assert _should_retry_response(response) is True

    def test_extra_retry_codes_retry(self) -> None:
        """Extra retry codes passed as argument should retry."""
        response = _make_response(418)  # I'm a teapot
        assert _should_retry_response(response) is False
        assert _should_retry_response(response, extra_retry_codes=[418]) is True

    def test_non_retryable_code_does_not_retry(self) -> None:
        """Non-retryable status codes should not retry."""
        response = _make_response(400)
        assert _should_retry_response(response) is False

    def test_504_from_our_server_does_not_retry(self) -> None:
        """504 from our server (has x-fal-request-id) is not ingress error."""
        response = _make_response(
            504,
            headers={"x-fal-request-id": "req-123"},
        )
        # Not an ingress error, and 504 not in RETRY_CODES
        assert _should_retry_response(response) is False


class TestShouldRetry:
    """Tests for _should_retry function."""

    def test_timeout_exception_with_user_timeout_does_not_retry(self) -> None:
        """TimeoutException with user timeout header should NOT retry.

        When user specifies timeout=30, both httpx client and server get 30s.
        If client times out first, we should honor user's intent and not retry.
        """
        from fal_client._headers import REQUEST_TIMEOUT_HEADER

        request = MagicMock(spec=httpx.Request)
        request.headers = {REQUEST_TIMEOUT_HEADER: "30"}  # User set timeout

        exc = httpx.TimeoutException("timeout", request=request)

        # Should NOT retry - user specified a timeout
        assert _should_retry(exc) is False

    def test_timeout_exception_without_user_timeout_retries(self) -> None:
        """TimeoutException without user timeout header should retry.

        If no user timeout was specified, treat as transient network issue.
        """
        request = MagicMock(spec=httpx.Request)
        request.headers = {}  # No user timeout

        exc = httpx.TimeoutException("timeout", request=request)

        # Should retry - no user timeout, could be transient
        assert _should_retry(exc) is True

    def test_transport_error_retries(self) -> None:
        """TransportError should always retry."""
        exc = httpx.TransportError("connection failed")

        assert _should_retry(exc) is True

    def test_http_status_error_with_user_timeout_does_not_retry(self) -> None:
        """HTTPStatusError with user timeout response should NOT retry."""
        response = _make_response(
            504,
            headers={REQUEST_TIMEOUT_TYPE_HEADER: "user"},
            text="nginx timeout",
        )
        request = MagicMock(spec=httpx.Request)
        exc = httpx.HTTPStatusError("error", request=request, response=response)

        assert _should_retry(exc) is False

    def test_http_status_error_with_retryable_code_retries(self) -> None:
        """HTTPStatusError with retryable code should retry."""
        response = _make_response(429)
        request = MagicMock(spec=httpx.Request)
        exc = httpx.HTTPStatusError("rate limited", request=request, response=response)

        assert _should_retry(exc) is True

    def test_generic_exception_does_not_retry(self) -> None:
        """Generic exceptions should not retry."""
        exc = ValueError("something went wrong")

        assert _should_retry(exc) is False


class TestUserTimeoutNoRetry:
    """
    Integration-style tests verifying user timeout doesn't trigger retry.

    These tests verify the complete behavior: when the server returns a 504
    with X-Fal-Request-Timeout-Type: user, the client should NOT retry.
    """

    def test_user_timeout_504_from_gateway_does_not_retry(self) -> None:
        """504 with user timeout type from gateway should not retry."""
        # Simulate gateway response for user-defined timeout
        response = _make_response(
            504,
            headers={
                REQUEST_TIMEOUT_TYPE_HEADER: "user",
                "x-fal-request-id": "req-123",  # Has request ID, so from our server
            },
            text="Request timeout exceeded",
        )
        request = MagicMock(spec=httpx.Request)
        exc = httpx.HTTPStatusError("timeout", request=request, response=response)

        # Should not retry because it has user timeout type
        assert _should_retry(exc) is False

    def test_infrastructure_timeout_504_without_user_header_may_retry(self) -> None:
        """504 without user timeout type is infrastructure timeout, may retry."""
        # Simulate ingress/infrastructure timeout (nginx)
        response = _make_response(
            504,
            headers={},
            text="<html>nginx gateway timeout</html>",
        )
        request = MagicMock(spec=httpx.Request)
        exc = httpx.HTTPStatusError("timeout", request=request, response=response)

        # Should retry because it's an ingress error without user timeout
        assert _should_retry(exc) is True

    def test_504_with_user_timeout_does_not_retry(self) -> None:
        """504 with user timeout header does not retry, regardless of source."""
        # 504 from nginx with user timeout
        response = _make_response(
            504,
            headers={REQUEST_TIMEOUT_TYPE_HEADER: "user"},
            text="nginx error",
        )
        request = MagicMock(spec=httpx.Request)
        exc = httpx.HTTPStatusError("error", request=request, response=response)

        assert _should_retry(exc) is False

    def test_502_503_ingress_errors_retry_even_with_timeout_header(self) -> None:
        """502/503 ingress errors retry even if timeout header present.

        Note: The server only sets X-Fal-Request-Timeout-Type on 504 responses,
        so 502/503 with this header is not a realistic scenario. However, the
        implementation only checks for 504 specifically.
        """
        for code in [502, 503]:
            response = _make_response(
                code,
                headers={REQUEST_TIMEOUT_TYPE_HEADER: "user"},
                text="nginx error",
            )
            request = MagicMock(spec=httpx.Request)
            exc = httpx.HTTPStatusError("error", request=request, response=response)

            # 502/503 are ingress errors and will retry (timeout check is only for 504)
            assert _should_retry(exc) is True, f"{code} should retry as ingress error"
