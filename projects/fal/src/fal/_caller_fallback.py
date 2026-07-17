"""Caller-defined fallback chains, applied uniformly to every fal endpoint.

A caller may include a ``fal_fallback`` array in the request body: an ordered
list of alternative endpoints, each with its own input. Each node opts into
trigger categories via ``on``: ``server_error`` (any 5xx, 504 included),
``timeout`` (408 or a blown time budget), and ``policy`` (422 model/content
rejection -- opt-in only, never triggered by default). Because each node
carries its own input, models with different request schemas can be chained
without server-side adapters.

A node may instead set ``retry: true`` -- a shortcut that re-attempts the
previous attempt (with the single-node cap: the primary) with the same endpoint
and payload. Unlike fallback nodes, a retry node is policy-eligible BY DEFAULT:
content flags usually moderate stochastic output, a re-roll can pass, and the
same model re-applies its own moderation, so nothing is bypassed (decision
2026-07-15; the ``policy`` opt-in keeps gating only fallback nodes). The chain
is capped at ``MAX_CHAIN_LENGTH`` node(s), i.e. 2 total attempts including the
primary; extra nodes are leniently ignored.

This lives in the fal SDK (not the model registry) so it applies to ALL apps
without per-app wiring. Triggering is based on the primary's response status,
which already encodes each app's own error semantics (4xx surfaced to the
caller, 5xx retried), so no app-specific error taxonomy is needed here.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Request-body key that holds the caller's fallback chain.
FALLBACK_FIELD = "fal_fallback"

#: Request-body key for the caller's timeout (seconds) on the primary attempt.
PRIMARY_TIMEOUT_FIELD = "fal_fallback_timeout"

#: Request header carrying the primary endpoint id (set by the platform on
#: dispatch; same value as ``fal.app.REQUEST_ENDPOINT_KEY``). Used to resolve a
#: ``retry: true`` node back to the primary endpoint.
PRIMARY_ENDPOINT_HEADER = "x-fal-endpoint"

#: Maximum number of chain nodes considered per request: the primary plus ONE
#: further attempt (a fallback endpoint OR a retry) = 2 total attempts. Extra
#: nodes are leniently ignored, never an error. (Reduced from 2 nodes / 3 total
#: attempts on 2026-07-06 to bound worst-case cost and latency.)
MAX_CHAIN_LENGTH = 1

#: Reserved key for referencing a value from the PRIMARY input inside a
#: fallback step's ``input``: ``{"$ref": "dot.path"}``. Explicit by design (the
#: caller writes it); avoids duplicating large prompts or inline media in the
#: step input and enables cross-schema mapping (e.g. ``image_urls.0`` mapped
#: into a single ``image_url``).
REF_KEY = "$ref"

#: Sentinel for an unresolvable ``$ref`` path; the containing field/element is
#: dropped so the target model never receives a stray ref object.
_MISSING = object()

#: Public OpenAPI spec for an endpoint, used to resolve a fallback target's
#: output shape for pre-flight schema validation (before any billable work).
OPENAPI_URL = "https://fal.ai/api/openapi/queue/openapi.json?endpoint_id={endpoint}"

#: Machine-readable error type returned when a fallback target's output shape
#: does not match the primary's (rejected up front, no attempt is made).
SHAPE_MISMATCH_TYPE = "fallback_output_shape_mismatch"

# Resolved target output shapes, cached per endpoint for the process lifetime.
# Values: a MEDIA_FIELDS name, or None for a non-media output. Failed lookups
# are NOT cached so they can retry on a later request.
_SHAPE_CACHE: dict[str, Optional[str]] = {}

#: Primary output fields, in priority order, used to infer a primary's output
#: contract for the output-shape guard. Fallback grouping is by the EXACT output
#: field: ``images`` chains only with ``images``, single ``image`` only with
#: ``image`` (no single<->list bridging), so a served fallback always matches the
#: primary's declared output shape. Covers the four modalities
#: (image / video / audio / text).
MEDIA_FIELDS = ("images", "image", "videos", "video", "audios", "audio", "text")

#: Statuses that map to a "timeout" trigger. Deliberately only 408: the
#: "timeout" category is for blown time budgets (408, `fal_fallback_timeout`,
#: node `timeout`); every 5xx -- 504 (gateway timeout) included -- counts as
#: "server_error".
_TIMEOUT_STATUSES = {408}

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
    retry: bool = Field(
        default=False,
        description="Shortcut: re-attempt the PREVIOUS attempt in the chain (the "
        "primary when first) with the same endpoint and payload. When set, "
        "``endpoint``/``input`` on this node are ignored; ``on``/``timeout`` "
        "still apply to the retry attempt. A retry node's DEFAULT ``on`` also "
        "includes 'policy': re-rolling the same model re-applies its own "
        "moderation, so a retry can never bypass it (decision 2026-07-15).",
    )
    on: list[str] = Field(
        default_factory=lambda: ["server_error", "timeout"],
        description="Which outcomes of the preceding attempt trigger this node: "
        "'server_error' (any 5xx, 504 included), 'timeout' (408 or a blown "
        "time budget), and/or 'policy' (422 model/content rejection; opt-in "
        "for FALLBACK nodes -- not in their default, so a 422 never reaches "
        "another model unless explicitly listed. Retry nodes default to "
        "policy-eligible, see ``retry``). Unknown values never match.",
    )

    @model_validator(mode="after")
    def _retry_defaults_policy_eligible(self) -> "FallbackNode":
        # Retry re-attempts the SAME model, which re-applies its own moderation,
        # so policy eligibility cannot bypass anything; content flags usually
        # moderate stochastic output and a re-roll can pass. An EXPLICIT `on`
        # still wins (the caller narrowed it on purpose). Mirrors the workflow
        # orchestrator's RETRY_ON = (server_error, timeout, policy).
        if self.retry and "on" not in self.model_fields_set:
            self.on = ["server_error", "timeout", "policy"]
        return self
    timeout: Optional[float] = Field(
        default=None,
        description="Seconds budget for this node's own attempt; non-positive ignored.",
    )


def parse_fallback_chain(body: Any) -> list[FallbackNode]:
    """Extract a fallback chain from a parsed request body. Never raises.

    A missing, non-list, or malformed ``fal_fallback`` yields an empty chain so
    a bad spec can never break an otherwise-valid request. The chain is capped
    at ``MAX_CHAIN_LENGTH`` node(s) (2 total attempts including the primary);
    nodes past the cap are leniently ignored.
    """
    if not isinstance(body, dict):
        return []
    raw = body.get(FALLBACK_FIELD)
    if not isinstance(raw, list):
        return []
    chain: list[FallbackNode] = []
    for item in raw:
        if len(chain) >= MAX_CHAIN_LENGTH:
            break
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


def _lookup_path(root: Any, path: str) -> Any:
    """Walk a dot-path (dict keys and list indices) into ``root``.

    Returns ``_MISSING`` when the path does not resolve. Never raises.
    """
    current = root
    for segment in path.split("."):
        if isinstance(current, dict) and segment in current:
            current = current[segment]
        elif (
            isinstance(current, list)
            and segment.isdigit()
            and int(segment) < len(current)
        ):
            current = current[int(segment)]
        else:
            return _MISSING
    return current


def resolve_input_refs(value: Any, primary_input: Any) -> Any:
    """Resolve ``{"$ref": "dot.path"}`` placeholders in a fallback step's input
    against the PRIMARY input. Never raises.

    A dict of exactly ``{"$ref": <str>}`` is replaced by the referenced value;
    a dict carrying any other key alongside ``$ref`` is treated as a literal.
    Unresolvable refs are DROPPED (the containing field or list element is
    removed) so the target model sees its own default or fails honestly on a
    missing field, never a stray ref object. Everything else passes through
    untouched.
    """
    if isinstance(value, dict):
        if set(value) == {REF_KEY} and isinstance(value[REF_KEY], str):
            return _lookup_path(primary_input, value[REF_KEY])
        resolved: dict = {}
        for key, item in value.items():
            result = resolve_input_refs(item, primary_input)
            if result is not _MISSING:
                resolved[key] = result
        return resolved
    if isinstance(value, list):
        items = []
        for item in value:
            result = resolve_input_refs(item, primary_input)
            if result is not _MISSING:
                items.append(result)
        return items
    return value


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

    Returns a trigger category (``server_error`` / ``timeout`` / ``policy``)
    when a fallback may be attempted, or ``None`` when the response is surfaced
    to the caller.

    Mapping: all 5xx, 504 included -> ``server_error``; 408 -> ``timeout``; 422
    (model/content rejection) -> ``policy``. ``policy`` is OPT-IN: the default node ``on`` is
    ``["server_error", "timeout"]``, so a 422 only falls back when a node explicitly
    lists ``"policy"`` -- the caller consciously accepts retrying a rejection
    (including a content-policy one) on another model, so opting in can never be
    a silent moderation bypass. All other 4xx (400/401/403/404/413/415/429) are
    surfaced, never fall back (429 stays out of scope).
    """
    if status_code in _TIMEOUT_STATUSES:
        return "timeout"
    if status_code >= 500:
        return "server_error"
    if status_code == 422:
        return "policy"
    return None


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


async def resolve_output_field(endpoint: str) -> tuple[bool, Optional[str]]:
    """Resolve an endpoint's primary output media field from its public OpenAPI.

    Returns ``(known, field)``: ``known`` is False when the spec could not be
    fetched or parsed (callers must fail OPEN and skip validation); ``field``
    is the output's media field, or ``None`` for a non-media output.
    Successful resolutions are cached for the process lifetime. Never raises.
    """
    if endpoint in _SHAPE_CACHE:
        return True, _SHAPE_CACHE[endpoint]
    try:
        import httpx  # noqa: PLC0415

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(OPENAPI_URL.format(endpoint=endpoint))
            resp.raise_for_status()
            spec = resp.json()

        schemas = (spec.get("components") or {}).get("schemas") or {}

        def _field_of(schema_name: str) -> Optional[str]:
            props = (schemas.get(schema_name) or {}).get("properties") or {}
            for media in MEDIA_FIELDS:
                if media in props:
                    return media
            return None

        # Prefer the POST 200 response schema of the spec's path(s); fall back
        # to any *Output* component if the paths give nothing.
        field: Optional[str] = None
        for path_item in (spec.get("paths") or {}).values():
            ok = ((path_item.get("post") or {}).get("responses") or {}).get("200") or {}
            content = (ok.get("content") or {}).get("application/json") or {}
            ref = (content.get("schema") or {}).get("$ref") or ""
            if ref:
                field = _field_of(ref.rsplit("/", 1)[-1])
                if field:
                    break
        if field is None:
            for name in schemas:
                if "Output" in name:
                    field = _field_of(name)
                    if field:
                        break

        _SHAPE_CACHE[endpoint] = field
        return True, field
    except Exception:
        return False, None


#: Async resolver signature: endpoint -> (known, output_field).
ShapeResolver = Callable[[str], Awaitable[tuple[bool, Optional[str]]]]


async def validate_chain_shapes(
    chain: list[FallbackNode],
    primary_field: Optional[str],
    resolver: ShapeResolver,
) -> Optional[dict]:
    """Pre-flight schema validation: every fallback target must share the
    primary's output shape.

    Returns an error payload (fal error-taxonomy shaped) for the FIRST node
    whose resolved output shape is known and differs from the primary's, or
    ``None`` when the chain passes. Fail-open by design: an unknown primary
    field, retry nodes, and unresolvable targets are skipped -- the runtime
    output-shape guard still covers those.
    """
    if not primary_field:
        return None
    for index, node in enumerate(chain):
        if node.retry or not node.endpoint:
            continue  # a retry re-targets a prior attempt; always compatible
        known, node_field = await resolver(node.endpoint)
        if known and node_field != primary_field:
            produces = f"'{node_field}'" if node_field else "a non-media output"
            return {
                "detail": [
                    {
                        "loc": ["body", FALLBACK_FIELD, index, "endpoint"],
                        "msg": (
                            f"fallback endpoint '{node.endpoint}' produces "
                            f"{produces} but this endpoint produces "
                            f"'{primary_field}'; fallback targets must share "
                            "the primary's output shape."
                        ),
                        "type": SHAPE_MISMATCH_TYPE,
                    }
                ]
            }
    return None


_DISABLE_HEADER = b"x-app-fal-disable-fallback"


def _suppress_app_static_fallback(request) -> None:
    """Inject the disable-fallback header into the request scope so the app's own
    static ``@fallback_to`` short-circuits for this request (surfacing its
    primary error for the caller's chain to handle instead of serving the app
    author's curated fallback). Best-effort; never raises. No effect on apps
    without a static fallback -- the header is simply unread.
    """
    try:
        scope = request.scope
        headers = [
            (k, v)
            for (k, v) in scope.get("headers", [])
            if k.lower() != _DISABLE_HEADER
        ]
        headers.append((_DISABLE_HEADER, b"true"))
        scope["headers"] = headers
    except Exception:
        pass


def build_fallback_middleware(
    *,
    output_fields: Optional[dict] = None,
    forward: Optional[ForwardFn] = None,
    shape_resolver: Optional[ShapeResolver] = None,
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
    resolver = shape_resolver or resolve_output_field

    async def dispatch(request, call_next):
        import json  # noqa: PLC0415

        from fastapi.responses import JSONResponse  # noqa: PLC0415

        if request.method != "POST":
            return await call_next(request)

        # Only JSON bodies can carry `fal_fallback`. Skip non-JSON POSTs
        # (multipart/form uploads, binary) entirely -- don't buffer their body
        # or touch `_body`, which would risk breaking upload parsing.
        if "application/json" not in request.headers.get("content-type", "").lower():
            return await call_next(request)

        # Fail-safe pre-flight: read/parse/strip the body. A bug here must never
        # break a request that would otherwise succeed -- on any error we behave
        # as if the middleware were absent. We re-expose the original body first,
        # then strip our reserved fields (so apps with extra="forbid" inputs
        # aren't 422'd by the unknown `fal_fallback` key).
        stripped: Optional[dict] = None
        try:
            body_bytes = await request.body()
            request._body = body_bytes
            try:
                payload = json.loads(body_bytes) if body_bytes else None
            except Exception:
                payload = None
            if isinstance(payload, dict) and (
                FALLBACK_FIELD in payload or PRIMARY_TIMEOUT_FIELD in payload
            ):
                stripped = {
                    k: v
                    for k, v in payload.items()
                    if k not in (FALLBACK_FIELD, PRIMARY_TIMEOUT_FIELD)
                }
                request._body = json.dumps(stripped).encode("utf-8")
            disabled = request.headers.get(
                "x-app-fal-disable-fallback", ""
            ).lower() in ("true", "1", "yes", "y")
            chain = parse_fallback_chain(payload)
            primary_timeout = primary_timeout_from_body(payload)
        except Exception:
            return await call_next(request)

        if disabled or not chain:
            return await call_next(request)

        # Pre-flight schema validation: a fallback target must share the
        # primary's output shape. A KNOWN mismatch is a caller integration
        # error -- reject the whole request up front with a clear, typed 400
        # before any (billable) work happens. Unknown shapes (resolver
        # failure, non-media primary) fail OPEN; the runtime output-shape
        # guard still covers them. Wrapped so a bug in the validation itself
        # can never break a request (only the intentional 400 escapes).
        try:
            shape_error = await validate_chain_shapes(
                chain, output_fields.get(request.url.path), resolver
            )
        except Exception:
            shape_error = None
        if shape_error is not None:
            return JSONResponse(content=shape_error, status_code=400)

        # Caller intent wins (scenario 4): when the caller supplies a chain, it
        # is the authority on fallback, so suppress the app's own static
        # ``@fallback_to`` for this request. We inject the disable-fallback header
        # the static fallback already honors -- it then surfaces its primary
        # error instead of serving the app author's curated model, and that error
        # triggers OUR chain. So the caller falls back to THEIR chosen models, not
        # the app's. (Our own ``disabled`` was read from the original header
        # above, so this injection never short-circuits us.)
        _suppress_app_static_fallback(request)

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
            trigger, primary_exc = "server_error", exc
        else:
            trigger = trigger_for_status(response.status_code)
            # Don't replace a streaming/SSE response with a JSON fallback -- it
            # would break the streaming contract (success streams are already 2xx;
            # this also guards the rare streamed error).
            is_streaming = "text/event-stream" in response.headers.get(
                "content-type", ""
            )
            # A partner-backend failure surfaces as a 400 with error type
            # `downstream_service_error` (public error taxonomy). That is a
            # server-side failure of the target, not a caller mistake, so
            # upgrade it to a `server_error` trigger. Buffer + rebuild the
            # body to inspect it (kept intact when surfaced).
            if trigger is None and response.status_code == 400 and not is_streaming:
                from starlette.responses import Response as _Response  # noqa: PLC0415

                raw = b"".join([chunk async for chunk in response.body_iterator])
                response = _Response(
                    content=raw,
                    status_code=400,
                    headers={
                        k: v
                        for k, v in response.headers.items()
                        if k.lower() != "content-length"
                    },
                    media_type=response.media_type,
                )
                if b"downstream_service_error" in raw:
                    trigger = "server_error"
            if trigger is None or is_streaming:
                return response  # success, client error, or stream -> surface
            # 5xx ("server_error"), 408 / blown budget ("timeout"), and 422
            # ("policy") reach here.
            # "policy" is opt-in per node: with no node listing "policy" in its
            # `on`, the chain matches nothing and the 422 (including a
            # content-policy one) is surfaced intact -- falling back on a
            # rejection is always an explicit caller choice, never a default.

        # Fail-safe orchestration: a bug in our own fallback code must never lose
        # the primary's natural outcome. Any unexpected error here degrades to
        # returning the primary response (or re-raising the primary error).
        try:
            # `retry: true` nodes resolve to the primary via the platform's
            # endpoint header + the stripped payload. Note a primary retry is a
            # fresh queue call back to the same app; if the app is at capacity
            # it may wait in queue -- per-node `timeout` bounds that.
            result, served = await run_fallback_chain(
                chain,
                trigger=trigger,
                forward=fwd,
                output_field=output_fields.get(request.url.path),
                primary_endpoint=request.headers.get(PRIMARY_ENDPOINT_HEADER),
                primary_input=stripped,
            )
            record_fallback(
                trigger=trigger,
                outcome="served" if served else "exhausted",
                endpoint=served,
            )
            if served is not None:
                return JSONResponse(
                    content=result,
                    headers={
                        "x-app-fal-api-fallback": "true",
                        "x-app-fal-api-fallback-endpoint": served,
                        "x-fal-bill-as": served,
                    },
                )
        except Exception:
            pass  # fall through to the primary's outcome below

        if response is not None:
            return response  # keep the primary's failed response
        raise primary_exc  # re-surface the original timeout/exception

    return dispatch


async def run_fallback_chain(
    chain: list[FallbackNode],
    *,
    trigger: str,
    forward: ForwardFn,
    output_field: Optional[str] = None,
    primary_endpoint: Optional[str] = None,
    primary_input: Optional[dict] = None,
) -> tuple[Any, Optional[str]]:
    """Walk a caller-defined fallback chain, trying nodes in order.

    A node is attempted only when the current trigger (``server_error`` /
    ``timeout`` / ``policy``) is listed in its ``on``. A ``retry: true`` node re-attempts the PREVIOUS
    attempt -- the primary (``primary_endpoint`` + ``primary_input``) when it is
    the first attempt, otherwise the last chain node actually attempted -- with
    the same endpoint and payload. The first node that succeeds with an output
    compatible with ``output_field`` wins. Returns ``(result, endpoint)`` on
    success, or ``(None, None)`` if the chain is exhausted (the caller keeps the
    primary's failed response). Never raises on a node failure -- it advances
    instead.
    """
    # The (endpoint, input) a `retry` node re-attempts: starts at the primary,
    # then follows whatever was last attempted.
    last_endpoint = primary_endpoint
    last_input: dict = primary_input if isinstance(primary_input, dict) else {}
    for node in chain:
        if trigger not in node.on:
            continue
        if node.retry:
            endpoint, node_input = last_endpoint, last_input
        else:
            endpoint, node_input = node.endpoint, node.input
            # Resolve {"$ref": "dot.path"} placeholders against the primary
            # input, so large prompts / inline media need not be duplicated
            # and cross-schema mappings (image_urls.0 -> image_url) work.
            resolved = resolve_input_refs(node_input, primary_input)
            node_input = resolved if isinstance(resolved, dict) else {}
        if not endpoint:
            # Unusable: endpointless node, or a retry with no known prior
            # attempt (primary endpoint header absent). Skip, don't fail.
            continue
        # Strip reserved fields from the forwarded input: a nested
        # ``fal_fallback`` would otherwise be honored by the TARGET's own
        # middleware, letting a caller build chains-of-chains that multiply
        # attempts, runner holds, and cost beyond the cap.
        if FALLBACK_FIELD in node_input or PRIMARY_TIMEOUT_FIELD in node_input:
            node_input = {
                k: v
                for k, v in node_input.items()
                if k not in (FALLBACK_FIELD, PRIMARY_TIMEOUT_FIELD)
            }
        last_endpoint, last_input = endpoint, node_input
        node_timeout = node.timeout if (node.timeout and node.timeout > 0) else None
        try:
            result = await asyncio.wait_for(
                forward(endpoint, node_input), node_timeout
            )
        except (asyncio.TimeoutError, TimeoutError):
            trigger = "timeout"
            continue
        except Exception:
            trigger = "server_error"
            continue

        # Output-shape guard: the served result must carry the primary's EXACT
        # media field (grouping is by exact output shape -- ``images`` with
        # ``images``, single ``image`` with ``image``, no single<->list
        # bridging). A result lacking it (cross-modality, or the wrong list/single
        # form) is rejected and the chain advances, so the caller never gets a
        # shape that doesn't match what the primary endpoint declared.
        if output_field and isinstance(result, dict) and output_field not in result:
            trigger = "server_error"
            continue

        return result, endpoint

    return None, None
