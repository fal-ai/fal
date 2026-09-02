"""Wire-compatible error responses for the WMA session endpoint.

These reproduce the exact error shapes the fal platform already emits for
WMA sessions (422 field errors and billing-safe 500s), so a WMA app ported
between the fal registry and ``fal.wma`` answers identically on error paths:

- ``detail`` items carry ``loc`` / ``msg`` / ``type`` / ``url`` / ``input``
  (plus ``ctx`` when non-empty), with ``url`` pointing at
  ``https://docs.fal.ai/errors#<type>``.
- Responses carry ``X-Fal-needs-retry`` and, when billing applies,
  ``x-fal-billable-units`` headers.

Only the two classes WMA needs are defined; this is deliberately not a
general error framework.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Union

import pydantic
from fastapi import HTTPException
from pydantic import BaseModel

# https://github.com/pydantic/pydantic/pull/2573
if not hasattr(pydantic, "__version__") or pydantic.__version__.startswith("1."):
    IS_PYDANTIC_V2 = False
else:
    IS_PYDANTIC_V2 = True

ERROR_URL = "https://docs.fal.ai/errors"


def format_billable_units(units: Union[int, float]) -> str:
    """Render billable units in the SDK's canonical fixed-point wire form.

    ``str()`` would emit scientific notation for small floats
    (``str(1.23e-05) == '1.23e-05'``); the platform's billing paths emit
    fixed-point (see the AppException handler in ``fal.api.api``), so this
    mirrors that renderer exactly.
    """
    return format(float(units), ".0f" if isinstance(units, int) else ".8f")


def _model_input_dict(model: BaseModel) -> Any:
    exclude = getattr(model, "SCHEMA_IGNORES", None) or None
    if IS_PYDANTIC_V2:
        return model.model_dump(exclude=exclude)
    return model.dict(exclude=exclude)


def json_safe(value: Any) -> Any:
    """Return a JSON-renderable echo of ``value`` for an error payload.

    Error ``detail`` is rendered by Starlette's ``JSONResponse`` as
    ``json.dumps(..., ensure_ascii=False, allow_nan=False)`` with no
    ``default=`` fallback, so a value it can't encode — a non-serializable
    object, a ``NaN`` / ``Inf`` float, raw ``bytes``, or a lone-surrogate
    string — would raise *inside* the exception handler and turn the intended
    4xx into a 500. Walk by type and sanitize only unsafe leaves.
    """
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        # Starlette renders with allow_nan=False, so NaN / Inf would raise.
        return value if math.isfinite(value) else f"<{type(value).__name__}>"
    if isinstance(value, str):
        # Lone surrogates break Starlette's ensure_ascii=False UTF-8 encode.
        # Drop them (surrogatepass + ignore) rather than substitute: the echo
        # may be forwarded onward, where replacement characters are worse than
        # omission, and this matches the platform's established behavior.
        # (Ordinal comparison: the repo bans non-ASCII string literals.)
        if any(0xD800 <= ord(ch) <= 0xDFFF for ch in value):
            return value.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        return value
    if isinstance(value, bytes):
        return f"<binary {len(value)} bytes>"
    if isinstance(value, dict):
        return {json_safe(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return f"<{type(value).__name__}>"


class Error(dict):
    """One wire-shaped ``detail`` item: loc / msg / type / url / input [/ ctx]."""

    def __init__(
        self,
        loc: List[Union[str, int]],
        input: Any,
        msg: str,
        type: str,
        ctx: Optional[dict] = None,
    ):
        if isinstance(input, BaseModel):
            input = _model_input_dict(input)

        # Guarantee the echoed input is JSON-renderable so a non-serializable
        # value can never 500 the error response itself.
        input = json_safe(input)

        super().__init__(
            loc=loc,
            msg=msg,
            type=type,
            url=f"{ERROR_URL}#{type}",
            ctx=ctx,
            input=input,
        )

        if not ctx:
            self.pop("ctx")


class AppError(HTTPException):
    """An HTTP error carrying fal's retry and billing response headers."""

    def __init__(
        self,
        status_code: int,
        errors: List[Error],
        retryable: bool = False,
        billing_units: Optional[float] = None,
    ):
        headers = {"X-Fal-needs-retry": "true" if retryable else "false"}

        if billing_units is not None:
            headers["x-fal-billable-units"] = format_billable_units(billing_units)

        super().__init__(status_code=status_code, detail=errors, headers=headers)


class InternalServerError(AppError):
    def __init__(
        self,
        errors: Optional[List[Error]] = None,
        retryable: bool = False,
        input: Any = None,
    ):
        if not errors:
            error = Error(
                loc=["body"],
                msg="Internal server error",
                type="internal_server_error",
                input=input,
            )
            errors = [error]

        # 500s are never charged, so the billing header is pinned to zero.
        super().__init__(
            status_code=500, errors=errors, retryable=retryable, billing_units=0
        )


class InputValueError(AppError):
    def __init__(
        self,
        *,
        errors: List[Error],
        retryable: bool = False,
        billing_units: Optional[float] = None,
    ):
        super().__init__(
            status_code=422,
            errors=errors,
            retryable=retryable,
            billing_units=billing_units,
        )

    @classmethod
    def from_generic_error(
        cls, msg: str, input: Any, billing_units: Optional[float] = 0
    ) -> InputValueError:
        return cls(
            errors=[
                Error(
                    loc=["body"],
                    msg=msg,
                    type="input_value_error",
                    input=input,
                )
            ],
            billing_units=billing_units,
        )

    @classmethod
    def from_field_error(
        cls,
        *,
        field: Union[str, List[Union[str, int]]],
        msg: str,
        input: Any = None,
        type: str = "value_error",
        billing_units: Optional[float] = 0,
    ) -> InputValueError:
        """Create a 422 validation error located at a specific body field."""
        if isinstance(field, list):
            loc: List[Union[str, int]] = ["body", *field]
        else:
            loc = ["body", field]

        return cls(
            errors=[Error(loc=loc, msg=msg, type=type, input=input)],
            billing_units=billing_units,
        )
