from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BillingDeclaration(Protocol):
    """Anything that can publish an endpoint's billable components."""

    def declaration(self) -> dict[str, Any]: ...
