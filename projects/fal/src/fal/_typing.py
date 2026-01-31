from __future__ import annotations

from typing import Any, Callable, TypeVar

EndpointT = TypeVar("EndpointT", bound=Callable[..., Any])
