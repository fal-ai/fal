from __future__ import annotations

import re
from typing import Literal, Mapping, Union, get_args, Optional, Any, Callable

from httpx import Headers


_get_current_app: Optional[Callable[[], Any]] = None


def set_get_current_app(func: Callable[[], Any]):
    global _get_current_app
    _get_current_app = func


def get_current_app() -> Optional[Any]:
    if _get_current_app is None:
        return None
    return _get_current_app()


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
TAGS_HEADER = "X-Fal-Tags"

# Valid priority values
Priority = Literal["normal", "low"]

# Tag limits, mirroring the gateway's validator. The gateway fails open and
# drops pairs it rejects, so the client validates up front instead of silently
# sending tags that never make it into the account's usage breakdown.
MAX_TAG_PAIRS = 10
MAX_TAG_KEY_LENGTH = 64
MAX_TAG_VALUE_LENGTH = 256
MAX_TAGS_TOTAL_BYTES = 1024

# System-only tag namespace, rejected in caller input.
RESERVED_TAG_KEY_PREFIX = "fal."

_TAG_KEY_PATTERN = re.compile(r"^[a-z0-9._-]+$")


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


def _is_valid_tag_value(value: str) -> bool:
    # Printable ASCII (incl. space), excluding control chars and the "," separator.
    return all(char.isascii() and char.isprintable() and char != "," for char in value)


def add_tags_header(tags: Mapping[str, str], headers: dict[str, str]) -> None:
    """
    Validates the tags and adds the packed tags header to the headers dictionary.

    Keys and values are trimmed, keys are lowercased, and the pairs are packed
    into a single `key=value,key=value` header. An empty mapping adds no header.

    Args:
        tags: Tags to attach to the request, as a key to value mapping.
        headers: Headers dictionary to add the tags header to.

    Raises:
        ValueError: If a key or value is invalid, or a tag limit is exceeded.
    """
    if not tags:
        return

    if len(tags) > MAX_TAG_PAIRS:
        raise ValueError(f"At most {MAX_TAG_PAIRS} tags are allowed, got {len(tags)}")

    packed: dict[str, str] = {}
    total_bytes = 0

    for raw_key, raw_value in tags.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            raise ValueError(
                f"Tag keys and values must be strings, got {raw_key!r}: {raw_value!r}"
            )

        key = raw_key.strip().lower()
        value = raw_value.strip()

        if not _TAG_KEY_PATTERN.match(key):
            raise ValueError(
                f"Tag key must be non-empty and match [a-z0-9._-], got '{raw_key}'"
            )
        if not _is_valid_tag_value(value):
            raise ValueError(
                f"Tag value must be printable ASCII without ',', got '{raw_value}'"
            )
        if len(key) > MAX_TAG_KEY_LENGTH:
            raise ValueError(
                f"Tag key must be at most {MAX_TAG_KEY_LENGTH} characters, "
                f"got '{raw_key}'"
            )
        if len(value) > MAX_TAG_VALUE_LENGTH:
            raise ValueError(
                f"Tag value must be at most {MAX_TAG_VALUE_LENGTH} characters, "
                f"got '{raw_value}'"
            )
        if key.startswith(RESERVED_TAG_KEY_PREFIX):
            raise ValueError(
                f"Tag keys starting with '{RESERVED_TAG_KEY_PREFIX}' are reserved, "
                f"got '{raw_key}'"
            )

        # Both the pair count and this byte budget (key and value) are measured
        # before duplicate keys collapse -- the same way the gateway counts them.
        total_bytes += len(key) + len(value)
        if total_bytes > MAX_TAGS_TOTAL_BYTES:
            raise ValueError(
                f"Tags must be at most {MAX_TAGS_TOTAL_BYTES} bytes of keys and values"
            )

        packed[key] = value  # last-wins on duplicate keys

    headers[TAGS_HEADER] = ",".join(f"{key}={value}" for key, value in packed.items())


def add_fal_app_context_headers(headers: dict[str, str]) -> None:
    if request := _current_fal_app_request():
        if cdn_token := request.headers.get("x-fal-cdn-token"):
            headers["x-fal-cdn-token"] = cdn_token
        if can_disable_filter := request.headers.get("x-app-fal-can-disable-filter"):
            headers["x-app-fal-can-disable-filter"] = can_disable_filter


def handle_response_headers(response_headers: Headers) -> None:
    if request := _current_fal_app_request():
        if cdn_token := response_headers.get("x-fal-cdn-token"):
            request.headers["x-fal-cdn-token"] = cdn_token
