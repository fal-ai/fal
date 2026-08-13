from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

EndpointT = TypeVar("EndpointT", bound=Callable[..., Any])
