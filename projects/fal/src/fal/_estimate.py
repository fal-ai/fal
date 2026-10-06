"""Estimate mode: run an endpoint's paired estimate function instead of its handler.

An endpoint declares what it bills with ``@fal.endpoint(..., billing=...)`` and
pairs a pure estimate function with ``estimate=...``. A request carrying the
:data:`ESTIMATE_HEADER` header runs the estimate instead of the handler.

Estimate requests reuse the endpoint's request parsing and validation, never run
the handler, and always report zero billable units.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Protocol, get_type_hints, runtime_checkable

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.params import Param
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

ESTIMATE_HEADER = "x-app-fal-estimate"
_REQUEST_PARAMETER = "_fal_estimate_request"
_TRUTHY = frozenset({"1", "true", "yes"})


@runtime_checkable
class BillingDeclaration(Protocol):
    """Anything that can publish an endpoint's billable components."""

    def declaration(self) -> dict[str, Any]: ...


def _resolved_parameters(func: Callable[..., Any]) -> list[inspect.Parameter]:
    raw = getattr(func, "__func__", func)
    hints = get_type_hints(raw, include_extras=True)
    return [
        parameter.replace(annotation=hints[name]) if name in hints else parameter
        for name, parameter in inspect.signature(func).parameters.items()
    ]


def _body_parameter(endpoint: Callable[..., Any], path: str) -> inspect.Parameter:
    candidates = [
        parameter
        for parameter in _resolved_parameters(endpoint)
        if inspect.isclass(parameter.annotation)
        and issubclass(parameter.annotation, BaseModel)
        and not isinstance(parameter.default, Param)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Endpoint {path!r} must take exactly one request body model "
            "to support estimates."
        )
    return candidates[0]


def validate_estimate(
    path: str, endpoint: Callable[..., Any], estimate: Callable[..., Any]
) -> str:
    """Check the estimate takes only the endpoint's input model; return its name."""
    body = _body_parameter(endpoint, path)
    parameters = _resolved_parameters(estimate)
    if (
        len(parameters) != 1
        or parameters[0].kind
        not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        or parameters[0].annotation is not body.annotation
    ):
        raise ValueError(
            f"Estimate for {path!r} must be a plain function taking only "
            f"`{body.name}: {body.annotation.__name__}`, the endpoint's input "
            f"model. Got {inspect.signature(estimate)}."
        )
    return body.name


def _estimate_requested(request: Request) -> bool:
    return request.headers.get(ESTIMATE_HEADER, "").strip().lower() in _TRUTHY


def _estimate_response(result: Any) -> Response:
    if isinstance(result, Response):
        response = result
    elif callable(getattr(result, "to_json", None)):
        response = Response(result.to_json(), media_type="application/json")
    else:
        response = JSONResponse(jsonable_encoder(result))
    response.headers["x-fal-billable-units"] = "0"
    return response


def with_estimate_mode(
    path: str, endpoint: Callable[..., Any], estimate: Callable[..., Any]
) -> Callable[..., Any]:
    """Wrap an endpoint so the estimate header dispatches to ``estimate``.

    The wrapper keeps the endpoint's signature, with annotations resolved, plus
    a hidden ``Request`` parameter, so FastAPI parses and validates the body
    exactly as before and the OpenAPI spec is unchanged.
    """
    input_name = validate_estimate(path, endpoint, estimate)
    is_async = inspect.iscoroutinefunction(getattr(endpoint, "__func__", endpoint))

    @functools.wraps(endpoint)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        request: Request = kwargs.pop(_REQUEST_PARAMETER)
        if _estimate_requested(request):
            result = estimate(kwargs[input_name])
            if inspect.isawaitable(result):
                result = await result
            return _estimate_response(result)
        if is_async:
            return await endpoint(*args, **kwargs)
        return await run_in_threadpool(endpoint, *args, **kwargs)

    signature = inspect.signature(endpoint)
    hints = get_type_hints(getattr(endpoint, "__func__", endpoint), include_extras=True)
    parameters = _resolved_parameters(endpoint)
    request_parameter = inspect.Parameter(
        _REQUEST_PARAMETER, inspect.Parameter.KEYWORD_ONLY, annotation=Request
    )
    # Keyword-only parameters must precede a trailing **kwargs.
    position = len(parameters)
    if parameters and parameters[-1].kind is inspect.Parameter.VAR_KEYWORD:
        position -= 1
    parameters.insert(position, request_parameter)
    wrapper.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=parameters,
        return_annotation=hints.get("return", signature.return_annotation),
    )
    return wrapper
