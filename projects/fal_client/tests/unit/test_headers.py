"""Unit tests for _headers.py module."""

from __future__ import annotations

import pytest

from fal_client._headers import (
    MIN_REQUEST_TIMEOUT_SECONDS,
    QUEUE_PRIORITY_HEADER,
    REQUEST_TIMEOUT_HEADER,
    RUNNER_HINT_HEADER,
    add_hint_header,
    add_priority_header,
    add_timeout_header,
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

    def test_adds_priority_header(self) -> None:
        """Adds queue priority header correctly."""
        headers: dict[str, str] = {}
        add_priority_header("high", headers)

        assert QUEUE_PRIORITY_HEADER in headers
        assert headers[QUEUE_PRIORITY_HEADER] == "high"
