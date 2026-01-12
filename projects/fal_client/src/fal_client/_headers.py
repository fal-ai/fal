from __future__ import annotations

from typing import Union

MIN_REQUEST_TIMEOUT_SECONDS = 1  # Minimum allowed request timeout in seconds

# Request headers
REQUEST_TIMEOUT_HEADER = "X-Fal-Request-Timeout"
REQUEST_TIMEOUT_TYPE_HEADER = "X-Fal-Request-Timeout-Type"
RUNNER_HINT_HEADER = "X-Fal-Runner-Hint"
QUEUE_PRIORITY_HEADER = "X-Fal-Queue-Priority"


def _add_header(key: str, value: str, headers: dict[str, str]) -> None:
    """Add a header to the headers dictionary."""
    headers[key] = value


def add_timeout_header(timeout: Union[int, float], headers: dict[str, str]) -> None:
    """
    Validates the timeout and adds the timeout header to the headers dictionary.
    """
    try:
        timeout = float(timeout)

    except ValueError:
        raise ValueError(f"Timeout must be a number, got {timeout}")

    if timeout < MIN_REQUEST_TIMEOUT_SECONDS:
        raise ValueError(
            f"Timeout must be greater than {MIN_REQUEST_TIMEOUT_SECONDS} seconds"
        )
    _add_header(REQUEST_TIMEOUT_HEADER, str(timeout), headers)


def add_hint_header(hint: str, headers: dict[str, str]) -> None:
    _add_header(RUNNER_HINT_HEADER, hint, headers)


def add_priority_header(priority: str, headers: dict[str, str]) -> None:
    _add_header(QUEUE_PRIORITY_HEADER, priority, headers)
