from __future__ import annotations

from typing import Literal, Union, get_args

MIN_REQUEST_TIMEOUT_SECONDS = 1  # Minimum allowed request timeout in seconds

# Request headers
REQUEST_TIMEOUT_HEADER = "X-Fal-Request-Timeout"
REQUEST_TIMEOUT_TYPE_HEADER = "X-Fal-Request-Timeout-Type"
RUNNER_HINT_HEADER = "X-Fal-Runner-Hint"
QUEUE_PRIORITY_HEADER = "X-Fal-Queue-Priority"

# Valid priority values
Priority = Literal["normal", "low"]


def add_timeout_header(timeout: Union[int, float], headers: dict[str, str]) -> None:
    """
    Validates the timeout and adds the timeout header to the headers dictionary.
    """
    try:
        timeout = float(timeout)

    except ValueError:
        raise ValueError(f"Timeout must be a number, got {timeout}")

    if timeout <= MIN_REQUEST_TIMEOUT_SECONDS:
        raise ValueError(
            f"Timeout must be greater than {MIN_REQUEST_TIMEOUT_SECONDS} seconds"
        )
    headers[REQUEST_TIMEOUT_HEADER] = str(timeout)


def add_hint_header(hint: str, headers: dict[str, str]) -> None:
    headers[RUNNER_HINT_HEADER] = hint


def add_priority_header(priority: Priority, headers: dict[str, str]) -> None:
    """
    Validates the priority and adds the priority header to the headers dictionary.

    Args:
        priority: Queue priority, must be "normal" or "low".
        headers: Headers dictionary to add the priority header to.

    Raises:
        ValueError: If priority is not a valid value.
    """
    valid_priorities = get_args(Priority)
    if priority not in valid_priorities:
        raise ValueError(
            f"Priority must be one of {valid_priorities}, got '{priority}'"
        )
    headers[QUEUE_PRIORITY_HEADER] = priority
