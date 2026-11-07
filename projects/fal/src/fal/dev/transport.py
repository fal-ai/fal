"""Calls to routes of a box's app on fal.run, with retries for connect failures."""

from __future__ import annotations

import os
import threading
import time
import urllib.parse
from typing import Callable, TypeVar

import httpx

from fal._proxy import PROXY_HINT, https_proxy_for, redact
from fal.dev.errors import DevError

T = TypeVar("T")

# Connecting to fal's edge fails now and then (dropped SYNs). A failed connect
# means the request never reached the server, so it is safe to send again.
CONNECT_TRIES = 4


def _connect_timeout() -> float:
    try:
        seconds = float(os.environ.get("FAL_DEV_CONNECT_TIMEOUT", ""))
    except ValueError:
        return 3.0
    return seconds if 0 < seconds < float("inf") else 3.0


# Seconds per connect; slow proxies may need FAL_DEV_CONNECT_TIMEOUT raised.
CONNECT_TIMEOUT = _connect_timeout()
# Sleeps between tries: about 18 s in all with the connect timeouts.
CONNECT_BACKOFF = (1.0, 2.0, 3.0)
CONNECT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)

_pool: list[httpx.Client] = []
_pool_lock = threading.Lock()


class BoxUnreachable(DevError):
    """A call to the box did not get an answer (network, gateway timeout or 5xx).

    Safe to wait out when polling: the box keeps working meanwhile.
    """


class ConnectFailed(BoxUnreachable):
    """The host could not be reached after every try."""


def client() -> httpx.Client:
    """One pooled client per process, so keep-alive avoids a fresh connect."""
    with _pool_lock:
        if not _pool:
            _pool.append(
                httpx.Client(timeout=httpx.Timeout(60, connect=CONNECT_TIMEOUT))
            )
        return _pool[0]


def host_of(url: str) -> str:
    return urllib.parse.urlsplit(url).hostname or url


def retry_connect(
    call: Callable[[], T],
    *,
    host: str,
    step: str,
    tries: int = CONNECT_TRIES,
    backoff: tuple[float, ...] = CONNECT_BACKOFF,
) -> T:
    """Run `call`, trying again when it fails to connect; never on anything else.

    Read timeouts and error responses are not retried: the server may have
    acted on the request.
    """
    for attempt in range(1, tries + 1):
        try:
            return call()
        except CONNECT_ERRORS as exc:
            if attempt == tries:
                proxy = https_proxy_for(host)
                via = f" through proxy {redact(proxy)}" if proxy else ""
                reason = str(exc) or type(exc).__name__
                raise ConnectFailed(
                    f"could not reach {host}{via} while {step} after {tries} tries: "
                    f"{reason}" + (f"\nhint: {PROXY_HINT}" if proxy else ""),
                    title="the proxy is not reachable"
                    if proxy
                    else "the network is flaky",
                ) from exc
            time.sleep(backoff[min(attempt - 1, len(backoff) - 1)])
    raise AssertionError("unreachable")


def post_json(
    url: str,
    payload: dict,
    headers: dict[str, str],
    timeout: float = 60,
    *,
    step: str = "calling the box",
) -> httpx.Response:
    """POST a JSON body to `url` with the given (authorization) headers.

    Request I/O storage is switched off, because payloads carry project source
    and box tokens that must not be kept. A failed connect is tried again
    (see `retry_connect`); anything else is raised as is.
    """
    return retry_connect(
        lambda: client().post(
            url,
            json=payload,
            headers={**headers, "X-Fal-Store-IO": "0"},
            timeout=httpx.Timeout(timeout, connect=CONNECT_TIMEOUT),
        ),
        host=host_of(url),
        step=step,
    )
