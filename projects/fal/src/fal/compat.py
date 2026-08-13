from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any


async def run_in_thread(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(func, *args, **kwargs)
