"""Deferred billing report plumbing for WMA sessions.

A WMA session bills once, at close: the ``/start-session`` response carries
``x-fal-billable-units-webhook: 1`` so the gateway parks the request as
WAITING, and the accumulated total is settled with one authenticated POST to
the fal REST API. The header helpers live in :mod:`fal.wma.sdk`; this module
owns the REST client and the deadline-bounded report loop.

``httpx`` is imported lazily so importing :mod:`fal.wma` never requires it in
the (cloudpickled) runner environment unless a report is actually sent.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from typing import TYPE_CHECKING, Optional

from fal.wma._errors import format_billable_units
from fal.wma._request_id import valid_fal_request_id

if TYPE_CHECKING:
    import httpx

STREAM_BILLING_TIMEOUT_SECONDS = 60.0
STREAM_BILLING_ATTEMPT_TIMEOUT_SECONDS = 10.0

logger = logging.getLogger(__name__)


def make_fal_rest_client() -> httpx.AsyncClient:
    """Build the long-lived fal REST client used for billing reports."""

    import httpx

    from fal._user_agent import USER_AGENT
    from fal.flags import REST_URL

    return httpx.AsyncClient(
        base_url=REST_URL,
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )


async def report_stream_billing_units(
    rest_client: httpx.AsyncClient,
    fal_request_id: Optional[str],
    billing_units: float,
    *,
    log_prefix: str,
    timeout: float = STREAM_BILLING_TIMEOUT_SECONDS,
) -> None:
    """Report authoritative units for one completed session, with retries.

    Never raises: a report that cannot be delivered within ``timeout`` is
    logged as an error and leaves the gateway request WAITING — the monitored
    unbilled-session signal — rather than failing session teardown.
    """

    request_id = valid_fal_request_id(fal_request_id)
    if request_id is None:
        logger.info(
            "[%s] no valid x-fal-request-id; skipping stream billing report",
            log_prefix,
        )
        return
    if not os.environ.get("FAL_APP_NAME"):
        logger.info(
            "[%s] not in a fal app; skipping stream billing for %s",
            log_prefix,
            request_id,
        )
        return
    fal_key = os.environ.get("FAL_KEY")
    if not fal_key:
        logger.warning(
            "[%s] FAL_KEY unavailable; skipping stream billing for %s",
            log_prefix,
            request_id,
        )
        return
    if not math.isfinite(billing_units) or billing_units < 0:
        logger.error(
            "[%s] non-finite or negative billing_units %r; "
            "skipping stream billing for %s",
            log_prefix,
            billing_units,
            request_id,
        )
        return

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    for attempt in range(10):
        remaining = deadline - loop.time()
        if remaining <= 0.0:
            break
        try:
            response = await asyncio.wait_for(
                rest_client.post(
                    f"/requests/billable-units/{request_id}",
                    json={"billable_units": format_billable_units(billing_units)},
                    headers={"Authorization": f"Key {fal_key}"},
                ),
                timeout=min(STREAM_BILLING_ATTEMPT_TIMEOUT_SECONDS, remaining),
            )
            response.raise_for_status()
            logger.info(
                "[%s] reported %s billable units for %s",
                log_prefix,
                billing_units,
                request_id,
            )
            return
        except Exception as exc:
            remaining = deadline - loop.time()
            if attempt == 9 or remaining <= 0.0:
                break
            delay = min(2**attempt, 16, remaining)
            logger.warning(
                "[%s] stream billing report failed for %s: %s; retrying in %ss",
                log_prefix,
                request_id,
                exc,
                delay,
            )
            await asyncio.sleep(delay)

    logger.error("[%s] failed to report stream billing for %s", log_prefix, request_id)
