from contextlib import asynccontextmanager

from anyio import create_task_group
from fastapi import Request


@asynccontextmanager
async def cancel_on_disconnect(request: Request):
    """
    Async context manager for async code that needs to be cancelled if client
    disconnects prematurely.
    The client disconnect is monitored through the Request object.
    """
    async with create_task_group() as tg:

        async def watch_disconnect():
            while True:
                message = await request.receive()

                if message["type"] == "http.disconnect":
                    tg.cancel_scope.cancel()
                    break

        tg.start_soon(watch_disconnect)

        try:
            yield
        finally:
            tg.cancel_scope.cancel()
