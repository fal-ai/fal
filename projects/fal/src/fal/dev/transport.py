"""Calls to routes of a box's app on fal.run."""

from __future__ import annotations

import httpx


def post_json(
    url: str, payload: dict, headers: dict[str, str], timeout: float = 60
) -> httpx.Response:
    """POST a JSON body to `url` with the given (authorization) headers.

    Request I/O storage is switched off, because payloads carry project source
    and box tokens that must not be kept.
    """
    return httpx.post(
        url,
        json=payload,
        headers={**headers, "X-Fal-Store-IO": "0"},
        timeout=timeout,
    )
