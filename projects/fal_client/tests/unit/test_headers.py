"""Unit tests for _headers.py module."""

from __future__ import annotations

import pytest

from fal_client._headers import (
    MIN_REQUEST_TIMEOUT_SECONDS,
    QUEUE_PRIORITY_HEADER,
    REQUEST_TIMEOUT_HEADER,
    RUNNER_HINT_HEADER,
    add_fal_app_context_headers,
    add_hint_header,
    add_priority_header,
    add_timeout_header,
    set_get_current_app,
)


class TestAddTimeoutHeader:
    """Tests for add_timeout_header function."""

    def test_adds_header_with_valid_integer(self) -> None:
        """Valid integer timeout adds the header correctly."""
        headers: dict[str, str] = {}
        add_timeout_header(30, headers)

        assert REQUEST_TIMEOUT_HEADER in headers
        assert headers[REQUEST_TIMEOUT_HEADER] == "30.0"

    def test_adds_header_with_valid_float(self) -> None:
        """Valid float timeout adds the header correctly."""
        headers: dict[str, str] = {}
        add_timeout_header(45.5, headers)

        assert REQUEST_TIMEOUT_HEADER in headers
        assert headers[REQUEST_TIMEOUT_HEADER] == "45.5"

    def test_raises_for_timeout_below_minimum(self) -> None:
        """Raises ValueError for timeout below MIN_REQUEST_TIMEOUT_SECONDS."""
        headers: dict[str, str] = {}

        with pytest.raises(ValueError) as exc_info:
            add_timeout_header(0.5, headers)

        assert "must be greater than" in str(exc_info.value)
        assert str(MIN_REQUEST_TIMEOUT_SECONDS) in str(exc_info.value)

    def test_raises_for_zero_timeout(self) -> None:
        """Raises ValueError for zero timeout."""
        headers: dict[str, str] = {}

        with pytest.raises(ValueError):
            add_timeout_header(0, headers)

    def test_raises_for_negative_timeout(self) -> None:
        """Raises ValueError for negative timeout."""
        headers: dict[str, str] = {}

        with pytest.raises(ValueError):
            add_timeout_header(-5, headers)

    def test_boundary_at_minimum_fails(self) -> None:
        """Timeout exactly at minimum (1s) fails since it must be greater than."""
        headers: dict[str, str] = {}

        with pytest.raises(ValueError):
            add_timeout_header(0.99, headers)

    def test_boundary_just_above_minimum_succeeds(self) -> None:
        """Timeout just above minimum succeeds."""
        headers: dict[str, str] = {}

        # 1.0 should succeed (1.0 < 1 is False)
        add_timeout_header(1.01, headers)

        assert REQUEST_TIMEOUT_HEADER in headers
        assert headers[REQUEST_TIMEOUT_HEADER] == "1.01"

    def test_preserves_existing_headers(self) -> None:
        """Adding timeout header preserves existing headers."""
        headers: dict[str, str] = {"X-Existing": "value"}
        add_timeout_header(30, headers)

        assert headers["X-Existing"] == "value"
        assert headers[REQUEST_TIMEOUT_HEADER] == "30.0"

    def test_overwrites_existing_timeout_header(self) -> None:
        """Adding timeout header overwrites existing timeout header."""
        headers: dict[str, str] = {REQUEST_TIMEOUT_HEADER: "10.0"}
        add_timeout_header(30, headers)

        assert headers[REQUEST_TIMEOUT_HEADER] == "30.0"


class TestAddHintHeader:
    """Tests for add_hint_header function."""

    def test_adds_hint_header(self) -> None:
        """Adds runner hint header correctly."""
        headers: dict[str, str] = {}
        add_hint_header("gpu-a100", headers)

        assert RUNNER_HINT_HEADER in headers
        assert headers[RUNNER_HINT_HEADER] == "gpu-a100"


class TestAddPriorityHeader:
    """Tests for add_priority_header function."""

    def test_adds_normal_priority_header(self) -> None:
        """Adds queue priority header correctly with 'normal' value."""
        headers: dict[str, str] = {}
        add_priority_header("normal", headers)

        assert QUEUE_PRIORITY_HEADER in headers
        assert headers[QUEUE_PRIORITY_HEADER] == "normal"

    def test_adds_low_priority_header(self) -> None:
        """Adds queue priority header correctly with 'low' value."""
        headers: dict[str, str] = {}
        add_priority_header("low", headers)

        assert QUEUE_PRIORITY_HEADER in headers
        assert headers[QUEUE_PRIORITY_HEADER] == "low"

    def test_raises_for_invalid_priority(self) -> None:
        """Raises ValueError for invalid priority value."""
        headers: dict[str, str] = {}

        with pytest.raises(ValueError) as exc_info:
            add_priority_header("high", headers)  # type: ignore[arg-type]

        assert "must be one of" in str(exc_info.value)
        assert "high" in str(exc_info.value)

    def test_raises_for_empty_priority(self) -> None:
        """Raises ValueError for empty priority value."""
        headers: dict[str, str] = {}

        with pytest.raises(ValueError):
            add_priority_header("", headers)  # type: ignore[arg-type]


class _FakeRequest:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


class _FakeApp:
    def __init__(self, request: _FakeRequest | None) -> None:
        self.current_request = request


class TestAddFalAppContextHeaders:
    """Tests for add_fal_app_context_headers function."""

    @pytest.fixture(autouse=True)
    def _reset_current_app(self):
        yield
        set_get_current_app(None)  # type: ignore[arg-type]

    def _set_request(self, headers: dict[str, str] | None) -> None:
        request = _FakeRequest(headers) if headers is not None else None
        set_get_current_app(lambda: _FakeApp(request))

    def test_no_app_context_leaves_headers_untouched(self) -> None:
        """Outside a fal app, no context headers are added."""
        set_get_current_app(None)  # type: ignore[arg-type]
        headers: dict[str, str] = {}

        add_fal_app_context_headers(headers)

        assert headers == {}

    def test_no_current_request_leaves_headers_untouched(self) -> None:
        """When there is no current request, no context headers are added."""
        self._set_request(None)
        headers: dict[str, str] = {}

        add_fal_app_context_headers(headers)

        assert headers == {}

    def test_propagates_cdn_token(self) -> None:
        """Forwards x-fal-cdn-token from the current request."""
        self._set_request({"x-fal-cdn-token": "tok"})
        headers: dict[str, str] = {}

        add_fal_app_context_headers(headers)

        assert headers["x-fal-cdn-token"] == "tok"

    def test_propagates_can_disable_filter(self) -> None:
        """Forwards x-app-fal-can-disable-filter from the current request."""
        self._set_request({"x-app-fal-can-disable-filter": "true"})
        headers: dict[str, str] = {}

        add_fal_app_context_headers(headers)

        assert headers["x-app-fal-can-disable-filter"] == "true"

    def test_propagates_all_context_headers(self) -> None:
        """Forwards every supported context header at once."""
        self._set_request(
            {
                "x-fal-cdn-token": "tok",
                "x-app-fal-can-disable-filter": "true",
            }
        )
        headers: dict[str, str] = {}

        add_fal_app_context_headers(headers)

        assert headers == {
            "x-fal-cdn-token": "tok",
            "x-app-fal-can-disable-filter": "true",
        }

    def test_absent_headers_are_not_added(self) -> None:
        """Context headers missing from the request are not added."""
        self._set_request({})
        headers: dict[str, str] = {}

        add_fal_app_context_headers(headers)

        assert headers == {}
