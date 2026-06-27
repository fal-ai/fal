"""Caller-defined fallback chains, applied uniformly to every fal endpoint.

A caller may include a ``fal_fallback`` array in the request body: an ordered
list of alternative endpoints, each with its own input. When the primary
endpoint fails (5xx) or times out, the framework forwards the request to the
next eligible node. Because each node carries its own input, models with
different request schemas can be chained without server-side adapters.

This lives in the fal SDK (not the model registry) so it applies to ALL apps
without per-app wiring. Triggering is based on the primary's response status,
which already encodes each app's own error semantics (4xx surfaced to the
caller, 5xx retried), so no app-specific error taxonomy is needed here.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

#: Request-body key that holds the caller's fallback chain.
FALLBACK_FIELD = "fal_fallback"

#: Request-body key for the caller's timeout (seconds) on the primary attempt.
PRIMARY_TIMEOUT_FIELD = "fal_fallback_timeout"

#: Primary output fields, in priority order, used to infer a primary's output
#: contract for the result guard and normalization. Covers the four modalities
#: (image / video / audio / text). Text has no list/single variant, so it only
#: drives the output-shape guard, not normalization.
MEDIA_FIELDS = ("images", "image", "videos", "video", "audios", "audio", "text")

#: Statuses that map to a "timeout" trigger; other 5xx map to "error".
_TIMEOUT_STATUSES = {408, 504}

# Media output fields, mapped between their list and single-item forms, used to
# reconcile a fallback result to the primary's output contract (modality-level
# grouping allows, e.g., an ``image``-returning model to back an ``images`` one).
_PLURAL_BY_SINGULAR = {"image": "images", "video": "videos", "audio": "audios"}
_SINGULAR_BY_PLURAL = {plural: single for single, plural in _PLURAL_BY_SINGULAR.items()}

# An async callable that forwards (endpoint, input) to another fal endpoint.
ForwardFn = Callable[[str, dict], Awaitable[Any]]


class FallbackNode(BaseModel):
    """A single hop in a caller-defined fallback chain.

    Lenient by design: every field has a safe default and extra keys are
    ignored, so a malformed node can never reject a request whose primary would
    succeed. Unusable nodes (no ``endpoint``) and non-matching ``on`` values are
    skipped at runtime.
    """

    model_config = ConfigDict(extra="ignore")

    endpoint: Optional[str] = Field(
        default=None,
        description="fal endpoint id to fall back to, e.g. 'fal-ai/gpt-image-2'.",
    )
    input: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters shaped for ``endpoint`` (forwarded as-is).",
    )
    on: list[str] = Field(
        default_factory=lambda: ["error", "timeout"],
        description="Which outcomes of the preceding attempt trigger this node "
        "('error' and/or 'timeout'). Unknown values never match.",
    )
    timeout: Optional[float] = Field(
        default=None,
        description="Seconds budget for this node's own attempt; non-positive ignored.",
    )


def parse_fallback_chain(body: Any) -> list[FallbackNode]:
    """Extract a fallback chain from a parsed request body. Never raises.

    A missing, non-list, or malformed ``fal_fallback`` yields an empty chain so
    a bad spec can never break an otherwise-valid request.
    """
    if not isinstance(body, dict):
        return []
    raw = body.get(FALLBACK_FIELD)
    if not isinstance(raw, list):
        return []
    chain: list[FallbackNode] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            chain.append(FallbackNode(**item))
        except Exception:
            continue
    return chain


def primary_timeout_from_body(body: Any) -> Optional[float]:
    """Caller-chosen timeout (seconds) for the primary attempt. Never raises.

    Returns ``None`` when absent, non-numeric, or non-positive.
    """
    if not isinstance(body, dict):
        return None
    try:
        value = float(body.get(PRIMARY_TIMEOUT_FIELD))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def media_field(model: Any) -> Optional[str]:
    """The primary media field (e.g. ``images``) of a pydantic output model.

    Used to normalize a fallback result to the primary's output contract.
    Returns ``None`` when the model has no recognizable media field.
    """
    fields = getattr(model, "model_fields", None)
    if not fields:
        return None
    for name in MEDIA_FIELDS:
        if name in fields:
            return name
    return None


def trigger_for_status(status_code: int) -> Optional[str]:
    """Map a primary response status to a fallback trigger.

    Returns ``"timeout"`` / ``"error"`` when a fallback should be attempted, or
    ``None`` when the response should be surfaced to the caller (2xx/3xx/4xx).
    """
    if status_code in _TIMEOUT_STATUSES:
        return "timeout"
    if status_code >= 500:
        return "error"
    return None


def normalize_output(result: Any, expected_field: Optional[str]) -> Any:
    """Reconcile a fallback result to the primary's media field (single <-> list).

    Best-effort and never raises; anything it cannot map is returned untouched.
    """
    if not expected_field or not isinstance(result, dict) or expected_field in result:
        return result

    singular = _SINGULAR_BY_PLURAL.get(expected_field)
    if singular and result.get(singular) is not None:
        return {**result, expected_field: [result[singular]]}

    plural = _PLURAL_BY_SINGULAR.get(expected_field)
    if plural and isinstance(result.get(plural), list) and result[plural]:
        return {**result, expected_field: result[plural][0]}

    return result


_COUNTER: Any = None
_COUNTER_FAILED = False


def record_fallback(*, trigger: str, outcome: str, endpoint: Optional[str]) -> bool:
    """Increment the fallback counter; returns True if recorded.

    Defensive: if ``prometheus_client`` is unavailable or the counter can't be
    created/incremented, it silently no-ops (returns False) so metrics never
    affect request handling. ``outcome`` is ``served`` or ``exhausted``.
    """
    global _COUNTER, _COUNTER_FAILED
    if _COUNTER is None and not _COUNTER_FAILED:
        try:
            from prometheus_client import Counter  # noqa: PLC0415

            _COUNTER = Counter(
                "fal_caller_fallback_total",
                "Caller-defined fallback attempts",
                labelnames=["trigger", "outcome", "endpoint"],
            )
        except Exception:
            _COUNTER_FAILED = True
    if _COUNTER is None:
        return False
    try:
        _COUNTER.labels(
            trigger=trigger, outcome=outcome, endpoint=endpoint or "none"
        ).inc()
        return True
    except Exception:
        return False


def build_fallback_middleware(
    *,
    output_fields: Optional[dict] = None,
    forward: Optional[ForwardFn] = None,
):
    """Build the Starlette HTTP-middleware dispatch for caller-defined fallback.

    Extracted from the app wiring so it can be exercised with a FastAPI
    TestClient. ``forward`` defaults to ``fal_client.subscribe_async`` and is
    injectable for tests. ``output_fields`` maps a route path to its primary
    media output field for result normalization.

    Note on triggering: this middleware sits inside Starlette's
    ``ServerErrorMiddleware``, so an unhandled endpoint exception arrives here as
    a raised exception, while app-mapped failures arrive as a 5xx *response*.
    Both are handled -- exceptions and 5xx responses trigger ``error``, a blown
    primary timeout triggers ``timeout``.
    """
    output_fields = output_fields or {}

    async def _default_forward(endpoint: str, node_input: dict) -> Any:
        import fal_client  # noqa: PLC0415

        return await fal_client.subscribe_async(endpoint, arguments=node_input)

    fwd = forward or _default_forward

    async def dispatch(request, call_next):
        import json  # noqa: PLC0415

        from fastapi.responses import JSONResponse  # noqa: PLC0415

        if request.method != "POST":
            return await call_next(request)

        body_bytes = await request.body()
        try:
            payload = json.loads(body_bytes) if body_bytes else None
        except Exception:
            payload = None

        # Strip our reserved fields so the endpoint never sees them, regardless of
        # its input's `extra` policy (apps with extra="forbid" would otherwise
        # 422 the whole request on the unknown `fal_fallback` key). Re-expose the
        # (possibly trimmed) body so the endpoint still parses its own params.
        if isinstance(payload, dict) and (
            FALLBACK_FIELD in payload or PRIMARY_TIMEOUT_FIELD in payload
        ):
            stripped = {
                k: v
                for k, v in payload.items()
                if k not in (FALLBACK_FIELD, PRIMARY_TIMEOUT_FIELD)
            }
            request._body = json.dumps(stripped).encode("utf-8")
        else:
            request._body = body_bytes

        disabled = request.headers.get("x-app-fal-disable-fallback", "").lower() in (
            "true",
            "1",
            "yes",
            "y",
        )
        chain = parse_fallback_chain(payload)
        if disabled or not chain:
            return await call_next(request)

        primary_timeout = primary_timeout_from_body(payload)
        response = None
        primary_exc: Optional[BaseException] = None
        try:
            coro = call_next(request)
            if primary_timeout:
                response = await asyncio.wait_for(coro, primary_timeout)
            else:
                response = await coro
        except (asyncio.TimeoutError, TimeoutError) as exc:
            trigger, primary_exc = "timeout", exc
        except Exception as exc:
            trigger, primary_exc = "error", exc
        else:
            trigger = trigger_for_status(response.status_code)
            # Don't replace a streaming/SSE response with a JSON fallback -- it
            # would break the streaming contract (success streams are already 2xx;
            # this also guards the rare streamed error).
            is_streaming = "text/event-stream" in response.headers.get(
                "content-type", ""
            )
            if trigger is None or is_streaming:
                return response  # success, client error, or stream -> surface

        result, served = await run_fallback_chain(
            chain,
            trigger=trigger,
            forward=fwd,
            output_field=output_fields.get(request.url.path),
        )
        record_fallback(
            trigger=trigger,
            outcome="served" if served else "exhausted",
            endpoint=served,
        )
        if served is None:
            if response is not None:
                return response  # keep the primary's failed response
            raise primary_exc  # re-surface the original timeout/exception

        return JSONResponse(
            content=result,
            headers={
                "x-app-fal-api-fallback": "true",
                "x-app-fal-api-fallback-endpoint": served,
                "x-fal-bill-as": served,
            },
        )

    return dispatch


async def run_fallback_chain(
    chain: list[FallbackNode],
    *,
    trigger: str,
    forward: ForwardFn,
    output_field: Optional[str] = None,
) -> tuple[Any, Optional[str]]:
    """Walk a caller-defined fallback chain, trying nodes in order.

    A node is attempted only when the current trigger (``error``/``timeout``) is
    listed in its ``on``. The first node that succeeds with an output compatible
    with ``output_field`` wins. Returns ``(result, endpoint)`` on success, or
    ``(None, None)`` if the chain is exhausted (the caller keeps the primary's
    failed response). Never raises on a node failure -- it advances instead.
    """
    for node in chain:
        if not node.endpoint or trigger not in node.on:
            continue
        node_timeout = node.timeout if (node.timeout and node.timeout > 0) else None
        try:
            result = await asyncio.wait_for(
                forward(node.endpoint, node.input), node_timeout
            )
        except (asyncio.TimeoutError, TimeoutError):
            trigger = "timeout"
            continue
        except Exception:
            trigger = "error"
            continue

        result = normalize_output(result, output_field)

        # Output-shape guard: reject a result lacking the primary's media field
        # (e.g. a cross-modality target) so the caller never gets garbage.
        if output_field and isinstance(result, dict) and output_field not in result:
            trigger = "error"
            continue

        return result, node.endpoint

    return None, None
