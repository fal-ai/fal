import asyncio

import pytest

from fal.toolkit.utils.endpoint import cancel_on_disconnect


class FakeRequest:
    def __init__(self, messages):
        self._messages = asyncio.Queue()
        for message in messages:
            self._messages.put_nowait(message)

    async def receive(self):
        return await self._messages.get()


@pytest.mark.asyncio
async def test_cancel_on_disconnect_cancels_inflight_work():
    request = FakeRequest([{"type": "http.disconnect"}])
    work_started = asyncio.Event()
    work_cancelled = asyncio.Event()

    async def run_work():
        async with cancel_on_disconnect(request):
            work_started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                work_cancelled.set()
                raise

    task = asyncio.create_task(run_work())
    await work_started.wait()
    await task

    assert work_cancelled.is_set()


@pytest.mark.asyncio
async def test_cancel_on_disconnect_ignores_non_disconnect_messages():
    request = FakeRequest([{"type": "http.request"}])

    async with cancel_on_disconnect(request):
        await asyncio.sleep(0)
