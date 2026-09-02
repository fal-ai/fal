"""Validation of fal request identifiers before REST path interpolation.

WMA sessions report deferred billing by ``POST``-ing to
``https://rest.alpha.fal.ai/requests/billable-units/{request_id}`` using the
app's ``FAL_KEY`` service-account credential. ``request_id`` originates from
the caller-controlled ``x-fal-request-id`` request header. Interpolating that
raw value into the URL path lets a caller smuggle ``..`` path segments or
``?`` / ``&`` query characters into the credential-bearing request (CWE-88,
argument injection) — ``httpx`` does not normalize ``..`` segments, so the
POST destination and query string can be shifted.

fal request identifiers are UUIDs: the upstream endpoint declares its path
parameter as ``request_id: UUID`` and looks the request up by UUID, so any
value that is not a canonical UUID could never match a real request. Coercing
the header to its canonical UUID form (or rejecting it) therefore closes the
injection surface with no billing impact — a canonical UUID string contains
only ``[0-9a-f-]`` and can never break out of a single path segment.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID


def valid_fal_request_id(request_id: Optional[str]) -> Optional[str]:
    """Return the canonical UUID string for ``request_id``, or ``None``.

    ``request_id`` is the caller-controlled ``x-fal-request-id`` header value.
    Returns the canonical (lowercase, hyphenated) UUID string when the value
    is a well-formed UUID, otherwise ``None``.

    Callers that build a REST path from a request id MUST use the returned
    value and skip the request when this returns ``None`` — never interpolate
    the raw header value, or a caller can perform path/argument injection into
    a ``FAL_KEY``-bearing request.
    """
    if not request_id:
        return None
    try:
        return str(UUID(request_id))
    except (ValueError, TypeError):
        return None
