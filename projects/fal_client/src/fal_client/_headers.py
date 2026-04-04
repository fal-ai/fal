from __future__ import annotations

from typing import Literal, Union, get_args, Optional, Any

from httpx import Headers

try:
    from fal.ref import get_current_app
except ImportError:

    def get_current_app() -> Optional[Any]:
        return None


def _current_fal_app_request() -> Optional[Any]:
    """Get the current request if we are running in a fal app."""
    if (app := get_current_app()) is not None and app.current_request is not None:
        return app.current_request
    return None


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


def add_fal_app_context_headers(headers: dict[str, str]) -> None:
    if request := _current_fal_app_request():
        if cdn_token := request.headers.get("x-fal-cdn-token"):
            headers["x-fal-cdn-token"] = cdn_token


def handle_response_headers(response_headers: Headers) -> None:
    if request := _current_fal_app_request():
        if cdn_token := response_headers.get("x-fal-cdn-token"):
            request.headers["x-fal-cdn-token"] = cdn_token
