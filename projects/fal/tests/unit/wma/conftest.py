from __future__ import annotations

import asyncio
import sys

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "allow_real_sleep: this test intentionally performs real (network) "
        "waits, e.g. a real aiortc ICE negotiation",
    )


@pytest.fixture(autouse=True)
def _event_loop_for_sync_construction():
    """Keep a current event loop set on Python <= 3.9.

    ``asyncio.Event()`` binds ``get_event_loop()`` at construction before
    3.10, and an earlier test's ``asyncio.run`` clears the thread's loop, so
    ported sync tests that build events at function scope would otherwise
    raise "no current event loop". No-op on 3.10+.
    """
    if sys.version_info >= (3, 10):
        yield
        return
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield
    finally:
        asyncio.set_event_loop(None)
        loop.close()
