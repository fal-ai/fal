from __future__ import annotations

import json
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

BILLING_COMPONENTS_VERSION = 1


class BillingComponents(TypedDict):
    """One endpoint's billable components, as deploy metadata publishes them
    under ``billing_components``, keyed by endpoint path.

    ``app`` is the JSON Schema of the report the endpoint sends: an object
    whose properties are the billable metrics. ``schema_id`` identifies that
    schema, so a report can say which declaration it follows.
    """

    version: Literal[1]
    schema_id: str
    app: dict[str, Any]


@runtime_checkable
class BillingDeclaration(Protocol):
    """Anything that can publish an endpoint's billable components."""

    def declaration(self) -> BillingComponents: ...


def check_declaration(path: str, value: object) -> BillingComponents:
    """Refuse a malformed declaration before it is published."""

    def fail(reason: str) -> ValueError:
        return ValueError(f"billing declaration for {path!r} {reason}")

    if not isinstance(value, dict):
        raise fail("must be a dict")
    if set(value) != {"version", "schema_id", "app"}:
        raise fail("must have exactly version, schema_id and app")
    version = value["version"]
    if isinstance(version, bool) or version != BILLING_COMPONENTS_VERSION:
        raise fail(f"must have version {BILLING_COMPONENTS_VERSION}")
    if not isinstance(value["schema_id"], str) or not value["schema_id"]:
        raise fail("must have a non-empty string schema_id")
    app = value["app"]
    if not isinstance(app, dict) or app.get("type") != "object":
        raise fail("must have an object JSON Schema as app")
    # Deploy metadata travels as JSON (a protobuf Struct, then JSONB): refuse
    # what JSON cannot carry, or would silently change (tuples, non-str keys).
    try:
        encoded = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        raise fail("must be JSON-serializable, without NaN or infinity") from None
    if json.loads(encoded) != value:
        raise fail("must survive a JSON round trip unchanged")
    return value  # type: ignore[return-value]
