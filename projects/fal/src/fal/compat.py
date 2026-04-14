from __future__ import annotations

import asyncio
from contextvars import copy_context
from functools import partial
from typing import Any, Callable


async def run_in_thread(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run sync code on a worker thread with Python 3.8+ support."""
    try:
        to_thread = asyncio.to_thread
    except AttributeError:
        loop = asyncio.get_running_loop()
        ctx = copy_context()
        bound = partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, ctx.run, bound)

    return await to_thread(func, *args, **kwargs)
