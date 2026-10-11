from itertools import islice
from types import SimpleNamespace

import httpx
import pytest

from fal_client.client import (
    AsyncRequestHandle,
    Completed,
    PollBackoff,
    SyncRequestHandle,
)


def test_backoff_grows_from_initial_up_to_maximum():
    delays = PollBackoff(jitter=0).delays()
    assert [next(delays) for _ in range(3)] == pytest.approx([0.25, 0.2625, 0.275625])
    assert next(islice(delays, 1000, None)) == 30.0


def test_backoff_jitter_stays_within_bounds():
    first = {next(PollBackoff().delays()) for _ in range(100)}
    assert len(first) > 1
    assert all(0.2 <= delay <= 0.3 for delay in first)
    assert all(
        delay <= 30.0 for delay in islice(PollBackoff(multiplier=2).delays(), 100)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial": 0},
        {"initial": 31},
        {"maximum": float("inf")},
        {"multiplier": 0.9},
        {"jitter": 1.5},
    ],
)
def test_backoff_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        PollBackoff(**kwargs)


def _two_polls_then_done():
    remaining = 2

    def respond(request):
        nonlocal remaining
        if remaining > 0:
            remaining -= 1
            return httpx.Response(200, json={"status": "IN_QUEUE", "queue_position": 1})
        return httpx.Response(
            200, json={"status": "COMPLETED", "logs": [], "metrics": {}}
        )

    return respond


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []

    async def async_sleep(delay):
        recorded.append(delay)

    # Replace the module references rather than patching attributes of the
    # stdlib modules, which the running event loop also uses.
    monkeypatch.setattr(
        "fal_client.client.time", SimpleNamespace(sleep=recorded.append)
    )
    monkeypatch.setattr("fal_client.client.asyncio", SimpleNamespace(sleep=async_sleep))
    return recorded


DOUBLING = PollBackoff(initial=1.0, multiplier=2.0, jitter=0.0)
UNUSED = PollBackoff(initial=999.0, maximum=999.0, jitter=0.0)


def _sync_handle():
    http = httpx.Client(transport=httpx.MockTransport(_two_polls_then_done()))
    return SyncRequestHandle.from_request_id(http, "fal-ai/test", "request-id")


def _async_handle():
    http = httpx.AsyncClient(transport=httpx.MockTransport(_two_polls_then_done()))
    return AsyncRequestHandle.from_request_id(http, "fal-ai/test", "request-id")


def test_sync_iter_events_backs_off_per_attempt(sleeps):
    statuses = list(_sync_handle().iter_events(backoff=DOUBLING))

    assert sleeps == [1.0, 2.0]
    assert isinstance(statuses[-1], Completed)


def test_sync_iter_events_fixed_interval_ignores_backoff(sleeps):
    list(_sync_handle().iter_events(interval=0.5, backoff=UNUSED))

    assert sleeps == [0.5, 0.5]


async def test_async_iter_events_backs_off_per_attempt(sleeps):
    statuses = [s async for s in _async_handle().iter_events(backoff=DOUBLING)]

    assert sleeps == [1.0, 2.0]
    assert isinstance(statuses[-1], Completed)


async def test_async_iter_events_fixed_interval_ignores_backoff(sleeps):
    async for _ in _async_handle().iter_events(interval=0.5, backoff=UNUSED):
        pass

    assert sleeps == [0.5, 0.5]
