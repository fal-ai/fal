"""Upload outputs to a caller-supplied S3 bucket instead of the fal CDN.

Private and meant to be temporary: this is the SDK-side stopgap until infra
ships a first-class upload destination, at which point this module goes away.
Hence the leading underscore -- nothing here is a supported public API.

The ``x-app-fal-upload-policy`` header carries an S3 pre-signed POST policy --
the JSON from boto3's ``generate_presigned_post()``::

    {"url": "https://bucket.s3.<region>.amazonaws.com/",
     "fields": {"key": "uploads/${filename}", "policy": "...", ...}}

Validation is synchronous and raises a 422; the POST is backgrounded and the
access URL returned immediately, matching ``registry/cdn.py``. Backgrounding is
about GPU cost -- the destination is a URL the caller chose, and blocking a
generation on it holds a runner hostage to someone else's storage.

The cost: on the fal CDN a read of a still-uploading object blocks rather than
404s, but an object POSTed straight to S3 has no fal-side metadata, so callers
here must tolerate a 404 window and get a permanently dead URL if the upload
fails. There is no fallback to the fal CDN -- a caller who asked for their own
bucket must not silently get fal-owned storage.

That guarantee is enforced, not assumed: the header is only honored where the
policy is parsed, so ``App`` refuses a WebSocket handshake carrying it rather
than let that surface write to fal storage (see
``fal.app._RejectUploadPolicyOnWebSocket``). A plain ``serve=True`` function is
the one surface that still ignores it, having no request context to reject from.

``tests/unit/toolkit/upload_policy_vectors.json`` pins the decisions this shares
with the registry, and lists the divergences.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import threading
import time
import unicodedata

# Qualified, never ``from urllib.parse import urlsplit``: urlsplit is an
# lru_cache wrapper, which cloudpickle cannot pickle by reference, so a module
# registered pickle-by-value ends up carrying urllib.parse's privates and fails
# to deserialize on a runner with a different CPython patch release.
import urllib.parse
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable, List, Tuple, Union
from uuid import uuid4

from fal.exceptions import AppException
from fal.ref import get_current_app
from fal.toolkit.exceptions import FileUploadException

if TYPE_CHECKING:
    # Annotations only, and the block never executes, so `httpx` is not bound at
    # module scope and cannot reach a pickle. See _new_client for why that matters.
    import httpx

UPLOAD_POLICY_KEY = "x-app-fal-upload-policy"
UPLOAD_POLICY_FILENAME_PLACEHOLDER = "${filename}"
# S3 browser-POST uploads are a single request capped at 5 GB.
UPLOAD_POLICY_MAX_BYTES = 5 * 1024 * 1024 * 1024

# Per socket operation, not wall-clock. The socket value is httpx's first
# positional, so it covers write too, which is the operative one for a 5 GB
# push. Plain floats, not an httpx.Timeout: _new_client reads these as globals,
# and an instance would pickle by reference and drag httpx in at load time.
UPLOAD_POLICY_TIMEOUT_SOCKET = 300.0
UPLOAD_POLICY_TIMEOUT_CONNECT = 10.0
# Bounds retries of one file, checked between attempts. Not a per-request cap.
UPLOAD_POLICY_TOTAL_DEADLINE = 300.0
# How long teardown waits for in-flight uploads before abandoning them.
UPLOAD_POLICY_DRAIN_TIMEOUT = 5.0
_MAX_ATTEMPTS = 5
_BASE_DELAY = 1
_MAX_DELAY = 30

_S3_ERROR_CODE_RE = re.compile(r"<Code>([A-Za-z0-9]{1,64})</Code>")

# urlsplit silently strips ASCII tab, CR and LF from anywhere in a URL and trims
# leading C0-or-space, so a URL carrying them validates as its stripped form and
# is then refused by httpx, which strips nothing. The upload dies on the first
# attempt with the caller already holding a success URL, so reject them here.
_URL_FORBIDDEN_RE = re.compile(r"[\x00-\x20\x7f-\x9f\u2028\u2029]")
# Long enough for any real presigned POST endpoint; short enough that a header
# cannot carry a megabyte of URL into every log line and error body.
_URL_MAX_LENGTH = 2048

# A multipart part's Content-Type is written to the wire verbatim, so every
# character a header parser may treat as a delimiter has to go, not just CR/LF.
# NEL (U+0085) and LS/PS (U+2028/9) are line breaks to some parsers.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u2028\u2029]")

# S3 folds field names on case and surrounding space. These are invisible or
# compatibility-equivalent, so a name carrying them is the same field to a human
# reader and a different one to a naive comparison: exactly how a second "key"
# slips past duplicate detection.
_ZERO_WIDTH_RE = re.compile(r"[\u00ad\u200b-\u200f\u2060\ufeff]")

_UNSET = object()

# Anchored: an unanchored ".s3." also admits PrivateLink names
# (vpce-….s3.<region>.vpce.amazonaws.com), whose zone resolves to a
# caller-chosen VPC CIDR, i.e. RFC1918 target selection through the allowlist.
# The single-label vpce forms still match here and are denied by label in
# _match_s3_host. The bucket group is captured because its presence is what
# separates virtual-hosted addressing (path must be empty) from path style
# (bucket is the first segment).
_S3_HOST_RE = re.compile(
    r"(?P<bucket>[a-z0-9][a-z0-9.-]*\.)?"  # optional bucket
    r"(?P<endpoint>s3(?:express-[a-z0-9-]+)?"  # Express glues the az on, no sep
    r"(?:[.-](?:dualstack|accelerate|accesspoint|object-lambda|outposts"
    r"|fips|website))?"  # optional qualifier
    r"(?:[.-][a-z0-9-]+)?"  # optional region
    r"\.amazonaws\.com(?:\.cn)?)"
)


class UploadPolicyInputError(AppException):
    """Malformed policy header, or a policy the output cannot be sent through.

    Deliberately not a ``FieldException``: that type's whole payload is
    ``loc: ["body", field]`` and this validates a header, so the body is just
    ``{"detail": message}``. 422 rather than 424 because nothing was attempted
    against the caller's bucket.

    Billing follows where it surfaces: the middleware answers a bad header
    before inference and bills zero, while anything ``_prepare_upload`` can only
    discover once the output exists is billed.
    """

    def __init__(self, message: str) -> None:
        AppException.__init__(self, message=message, status_code=422)
        # Dataclass exception, so Exception.args stays empty and str(exc) would
        # render as "" in tracebacks and logs without this.
        self.args = (message,)


class UploadPolicyError(FileUploadException, AppException):
    """The upload to the caller's bucket failed, or the queue refused it.

    Only the queue-refusal path reaches a caller -- a failed POST happens on a
    background thread and is logged, not raised. 424 rather than 5xx because
    the fault is a caller-supplied destination: fal-js retries 502 by default
    (re-running the inference), and 5xx counts toward the app's public status.

    Also a ``FileUploadException`` so ``except FileUploadException`` still works.
    """

    def __init__(self, message: str) -> None:
        AppException.__init__(self, message=message, status_code=424)
        self.args = (message,)


# An S3 POST field value: a JSON scalar, or a flat list/tuple of them. httpx
# repeats a flat list as repeated form fields; parse_upload_policy rejects
# nested containers.
PolicyFieldScalar = Union[str, int, float, bool, None]
PolicyFieldValue = Union[
    PolicyFieldScalar, List[PolicyFieldScalar], Tuple[PolicyFieldScalar, ...]
]


@dataclass(frozen=True)
class UploadPolicy:
    url: str
    # As parsed; httpx encodes them. 'key' and any 'Content-Type' are str,
    # enforced by parse_upload_policy.
    fields: dict[str, PolicyFieldValue]


def _match_s3_host(host: str) -> re.Match[str] | None:
    """The allowlist match for ``host``, or ``None``. The ``bucket`` group tells
    callers whether the bucket is in the host, which bounds the path.

    Anchored; see ``_S3_HOST_RE`` for why. The region label is not validated, so
    an unreachable name under AWS DNS passes -- validating it would break the
    first customer in a newly launched region, which is worse.

    Callers need the match itself, not just a verdict: the ``bucket`` group
    decides how deep a path the URL may carry.
    """
    match = _S3_HOST_RE.fullmatch(host)
    if match is None:
        return None
    # Reject the S3 PrivateLink zone. Its DNS is publicly delegated and resolves
    # to a caller-chosen VPC CIDR (RFC1918), which is the SSRF this allowlist
    # exists to block. Checked by label on the endpoint only, so a bucket that is
    # legitimately named "vpce-something" is still allowed to upload.
    endpoint = match.group("endpoint")
    if "vpce" in re.split(r"[.-]", endpoint):
        return None
    return match


def _reserved_name(name: str) -> str:
    """Fold a field name for reserved-name matching.

    S3 folds case and surrounding space. This also folds NFKC and strips
    zero-width characters, which is stricter than S3 and therefore fails closed:
    a name that looks identical to whoever reads the policy cannot pass duplicate
    detection and go on the wire un-substituted.
    """
    folded = unicodedata.normalize("NFKC", name)
    return _ZERO_WIDTH_RE.sub("", folded).strip().lower()


def _require_encodable(what: str, value: str) -> None:
    """Reject strings that cannot be encoded, e.g. lone surrogates.

    ``json.loads`` happily turns the ASCII escape ``\\ud800`` into a real lone
    surrogate. Left alone it survives host validation and then raises
    ``UnicodeEncodeError`` out of ``quote()`` -- or, worse, reaches the response
    as an un-encodable URL and kills the JSON encoder. Both are 500s on input
    that should be a 422.
    """
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} {what}: not encodable as UTF-8"
        ) from exc


def _validate_policy_url(url: str) -> None:
    """Reject any destination that is not an HTTPS S3 endpoint on port 443."""
    _require_encodable("'url'", url)
    if len(url) > _URL_MAX_LENGTH:
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} 'url': longer than "
            f"{_URL_MAX_LENGTH} characters"
        )
    # Checked on the raw string: by the time urlsplit has run, a URL carrying
    # these looks clean. See _URL_FORBIDDEN_RE.
    if _URL_FORBIDDEN_RE.search(url):
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} 'url': must not contain control "
            "characters or spaces"
        )
    try:
        # urlsplit, not urlparse: urlparse strips ";params" off the last path
        # segment, so the checks below would read a path the wire never sees and
        # ".../;extra" would pass as the bucket root.
        parsed_url = urllib.parse.urlsplit(url)
        host = (parsed_url.hostname or "").lower().rstrip(".")
        port = parsed_url.port
    except ValueError as exc:
        # urlsplit raises on, among others, hostnames that change under NFKC
        # normalization. Bad client input, not a server fault.
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} 'url': could not be parsed ({exc})"
        ) from exc

    host_match = _match_s3_host(host)
    if parsed_url.scheme.lower() != "https" or host_match is None:
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} 'url': must be an HTTPS S3 upload URL"
        )
    if port is not None and port != 443:
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} 'url': port {port} is not allowed"
        )
    if parsed_url.username or parsed_url.password:
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} 'url': must not contain credentials"
        )
    # A '.'/'..' segment here collapses under URL resolution, so the returned
    # access URL would name a different object. Decode first and treat "\" as "/"
    # (browsers do): else "%2e", "%2f", or "\" smuggle traversal past the check.
    decoded_path = urllib.parse.unquote(parsed_url.path).replace("\\", "/")
    raw_segments = decoded_path.split("/")
    segments = [segment for segment in raw_segments if segment]
    if any(seg in (".", "..") for seg in segments):
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} 'url': must not contain '.' or '..' "
            "path segments"
        )
    # An empty interior segment survives that filter ("//bucket", or a leading
    # "%2f"), and _prepare_upload appends the key to policy.url as given, so the
    # access URL would carry the doubled slash too and name nothing. A real
    # trailing "/" is the one empty tail that is fine, because _prepare_upload
    # rstrips it; a percent-encoded one it cannot.
    interior = raw_segments[1:-1] if parsed_url.path.endswith("/") else raw_segments[1:]
    if "" in interior:
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} 'url': must not contain empty path segments"
        )
    # A pre-signed POST addresses a bucket, never a key: S3 answers 405
    # MethodNotAllowed to a POST at any deeper path, and _prepare_upload appends
    # the key to this URL, so a deeper path also returns an access URL naming an
    # object that was never stored. Virtual-hosted names the bucket in the host,
    # so its path must be empty; path style carries exactly the bucket.
    bucket_label = host_match.group("bucket")
    if bucket_label:
        if segments:
            raise UploadPolicyInputError(
                f"Invalid {UPLOAD_POLICY_KEY} 'url': must point at the bucket "
                f"root, not {decoded_path!r}"
            )
        # The wildcard certificate covers one label, so a dotted bucket cannot be
        # reached virtual-hosted over HTTPS: the handshake fails on a subject-name
        # mismatch and the upload dies behind a 200. boto3 switches such buckets
        # to path style.
        #
        # MRAP and S3 on Outposts mint dotted names with no path-style form, so
        # the rule cannot apply there. Both are matched on the endpoint *and* the
        # label shape, so a bucket merely named "evil.mrap.accesspoint" in a
        # regular zone is still rejected.
        bucket = bucket_label.rstrip(".")
        endpoint = host_match.group("endpoint")
        dotted_by_design = (
            endpoint == "s3-global.amazonaws.com"
            and re.fullmatch(r"[a-z0-9-]+\.mrap\.accesspoint", bucket) is not None
        ) or (
            endpoint.startswith("s3-outposts.")
            and re.fullmatch(r"[a-z0-9-]+\.[a-z0-9-]+", bucket) is not None
        )
        if "." in bucket and not dotted_by_design:
            raise UploadPolicyInputError(
                f"Invalid {UPLOAD_POLICY_KEY} 'url': bucket {bucket!r} contains "
                "a dot, so it must be addressed path style; virtual-hosted "
                "HTTPS cannot match it"
            )
    elif not segments:
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} 'url': must name a bucket, either in "
            "the host or as the first path segment"
        )
    elif len(segments) > 1:
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} 'url': must point at the bucket root, "
            f"not {decoded_path!r}"
        )
    if "?" in url or "#" in url:
        # The access URL is built by appending the key, so a query or fragment
        # would land in the middle of it.
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} 'url': must not have a query or fragment"
        )


def _validate_multipart_value(name: str, value: str) -> None:
    # httpx percent-encodes field *names* and filenames, but writes field values
    # and a part's Content-Type verbatim. Values are safe only because the
    # boundary is unguessable; a Content-Type is a real header, so a CRLF here
    # injects headers into the part.
    if "\r" in value or "\n" in value:
        raise UploadPolicyInputError(f"Invalid multipart {name}: contains CR/LF")
    # CR/LF is the only sequence that injects a header outright, but a bare NUL,
    # VT, FF, NEL or LS/PS is a line break or a terminator to some parser between
    # here and S3, and none of them belong in a media type.
    if _CONTROL_RE.search(value):
        raise UploadPolicyInputError(
            f"Invalid multipart {name}: contains a control character"
        )


def _headers_get(headers: Mapping[str, Any] | None, key: str) -> Any:
    if headers is None:
        return None
    value = headers.get(key)
    if value is not None or not isinstance(headers, dict):
        return value

    # Starlette's Headers is already case-insensitive and RequestContext.headers
    # is built from it; this rescan only covers hand-built dicts.
    key_lower = key.lower()
    for header_key, header_value in headers.items():
        if isinstance(header_key, str) and header_key.lower() == key_lower:
            return header_value
    return None


def parse_upload_policy(headers: Mapping[str, Any] | None) -> UploadPolicy | None:
    """Parse and validate the policy header.

    Returns ``None`` only when the header is absent. Raises
    ``UploadPolicyInputError`` (422) on anything malformed -- including a blank
    value, which is a policy the caller failed to build rather than a request
    to use the fal CDN. Silently falling back there would be the one failure
    mode nobody can detect afterwards.
    """
    raw = _headers_get(headers, UPLOAD_POLICY_KEY)
    if raw is None:
        return None
    if not isinstance(raw, (str, bytes)):
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} header: must be a string, got "
            f"{type(raw).__name__}. A stubbed request answers every header with a "
            "stub: have its headers return None for this one."
        )

    try:
        policy = json.loads(raw)
    except Exception as exc:
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} header: not valid JSON ({exc})"
        ) from exc

    if not isinstance(policy, dict):
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} header: expected a JSON object"
        )

    url = policy.get("url")
    fields = policy.get("fields")

    if not isinstance(url, str):
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} header: must contain string 'url'"
        )
    if not isinstance(fields, dict):
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} header: must contain object 'fields'"
        )

    _validate_policy_url(url)

    # Passed through untouched: httpx renders True as "true" where str(True) is
    # "True", and a signed field that changes shape fails the signature.
    validated: dict[str, PolicyFieldValue] = {}
    for name, value in fields.items():
        if not isinstance(name, str):
            raise UploadPolicyInputError(
                f"Invalid {UPLOAD_POLICY_KEY} header: field names must be strings"
            )
        # Before it can reach any message: an un-encodable name in an error
        # string breaks the JSON encoder and turns this 422 into a 500.
        _require_encodable("field name", name)

        # key and Content-Type must be real strings: they are compared and
        # substituted, not just forwarded.
        strict = _reserved_name(name) in ("key", "content-type")
        if isinstance(value, (dict, set)) or (strict and not isinstance(value, str)):
            raise UploadPolicyInputError(
                f"Invalid {UPLOAD_POLICY_KEY} header: fields.{name!r} must be a string"
            )

        # httpx repeats a flat list/tuple as multiple form fields but rejects a
        # nested container -- and it does so at upload time, after the app has
        # already done its work, as an uncaught TypeError.
        items = value if isinstance(value, (list, tuple)) else (value,)
        for item in items:
            if isinstance(item, (dict, set, list, tuple)):
                raise UploadPolicyInputError(
                    f"Invalid {UPLOAD_POLICY_KEY} header: fields.{name!r} must not "
                    "contain nested values"
                )
            if isinstance(item, str):
                _require_encodable(f"fields.{name!r}", item)
        validated[name] = value

    # S3 matches field names case-insensitively. Any duplicate "key" is rejected:
    # _prepare_upload substitutes only the exact-lowercase one, so the variant
    # would go on the wire un-substituted and collide across outputs.
    for reserved, allow_agreeing in (("key", False), ("content-type", True)):
        matches = [name for name in validated if _reserved_name(name) == reserved]
        if len(matches) < 2:
            continue
        # Safe in a set: both reserved names are strict-checked to str above.
        if allow_agreeing and len({validated[name] for name in matches}) == 1:
            continue
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} header: duplicate {reserved!r} "
            f"fields {sorted(matches)!r}"
        )

    _require_key_template(validated)

    for name, value in validated.items():
        if _reserved_name(name) == "content-type":
            _validate_multipart_value("content type", value)

    # A name that folds to a reserved field but is not that field to S3 cannot be
    # honoured: it travels as written, where no signed condition covers it, while
    # also suppressing the injection or substitution the real name triggers. The
    # key path fails closed the same way, by looking up the exact name.
    for name in validated:
        reserved = _reserved_name(name)
        if reserved in ("key", "content-type") and name.lower() != reserved:
            raise UploadPolicyInputError(
                f"Invalid {UPLOAD_POLICY_KEY} header: fields.{name!r} reads as "
                f"{reserved!r} but S3 would not treat it as that field"
            )

    return UploadPolicy(url=url, fields=validated)


def _require_key_template(fields: dict[str, PolicyFieldValue]) -> str:
    """The key must be a template, so concurrent outputs cannot collide."""
    key_template = fields.get("key")
    if (
        not isinstance(key_template, str)
        or UPLOAD_POLICY_FILENAME_PLACEHOLDER not in key_template
    ):
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} header: fields.key must contain "
            f"{UPLOAD_POLICY_FILENAME_PLACEHOLDER!r}"
        )
    # S3 stores these literally, but URL path resolution collapses them, so the
    # access URL would name a different object than the one stored.
    segments = key_template.split("/")
    if key_template.startswith("/") or "." in segments or ".." in segments:
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} header: fields.key must not start with "
            "'/' or contain '.' or '..' path segments"
        )
    return key_template


def _current_request() -> Any:
    current_app = get_current_app()
    return None if current_app is None else current_app.current_request


def get_upload_policy(request: Any = None) -> UploadPolicy | None:
    """The policy for the in-flight request, or ``None``.

    Reads the cache the middleware fills, falling back to header parsing only
    for a bare request. ``current_request`` is a ContextVar, so an app's own
    ``ThreadPoolExecutor`` does not see it and output there goes to the fal CDN,
    documented as unsupported. WebSocket endpoints never reach this:
    ``fal.app._RejectUploadPolicyOnWebSocket`` refuses the handshake.
    """
    if request is None:
        request = _current_request()

    if request is None:
        return None
    # Middleware (RequestContext) parses once per request and caches the result
    # here, as None or an UploadPolicy. A bare fastapi Request has no such
    # attribute and falls through to parsing its headers.
    cached = getattr(request, "upload_policy", _UNSET)
    if cached is None or isinstance(cached, UploadPolicy):
        return cached
    return parse_upload_policy(getattr(request, "headers", {}))


def _validate_size(nbytes: int) -> None:
    # Reached after the generation, so billed at the endpoint's default rather
    # than zeroed: the output's size, name and signed Content-Type are not
    # knowable up front. A malformed header is, and the middleware bills it zero.
    if nbytes > UPLOAD_POLICY_MAX_BYTES:
        raise UploadPolicyInputError(
            f"Upload via {UPLOAD_POLICY_KEY} is {nbytes} bytes, which exceeds the "
            f"{UPLOAD_POLICY_MAX_BYTES}-byte S3 POST limit; this path cannot chunk "
            "large uploads."
        )


def _prepare_upload(
    policy: UploadPolicy,
    file_name: str,
    content_type: str,
) -> tuple[str, dict[str, PolicyFieldValue]]:
    """Resolve the destination URL and the POST form fields.

    The uuid prefix keeps concurrent uploads from colliding on a shared policy,
    which is signed once and typically reused for a whole request.
    """
    _validate_multipart_value("content type", content_type)
    # An app may forward a user-supplied content type straight in; an
    # un-encodable one reaches httpx's header encoder as an uncaught 500.
    _require_encodable("content type", content_type)
    _require_encodable("file name", file_name)
    if "/" in file_name or "\\" in file_name:
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} file name {file_name!r}: "
            "must not contain a path separator"
        )

    # Not a re-parse: the middleware already rejected a malformed policy before
    # generation. Re-checked here because upload_* is exported, so a caller can
    # hand-build an UploadPolicy that never passed parse_upload_policy.
    _validate_policy_url(policy.url)
    fields = dict(policy.fields)
    key_template = _require_key_template(fields)

    upload_file_name = f"{uuid4().hex}-{file_name}"
    final_key = key_template.replace(
        UPLOAD_POLICY_FILENAME_PLACEHOLDER, upload_file_name
    )
    fields["key"] = final_key

    # Signed fields cannot be rewritten: inject Content-Type only if the caller
    # left it out, otherwise require agreement or S3 answers an opaque 403.
    existing = [name for name in fields if _reserved_name(name) == "content-type"]
    if not existing:
        fields["Content-Type"] = content_type
    elif fields[existing[0]] != content_type:
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} header: fields.{existing[0]} is "
            f"{fields[existing[0]]!r} but the file being uploaded is "
            f"{content_type!r}; a signed Content-Type cannot be changed"
        )

    encoded_key = urllib.parse.quote(final_key.lstrip("/"), safe="/~")
    access_url = f"{policy.url.rstrip('/')}/{encoded_key}"
    return access_url, fields


def _should_retry(exc: Exception, deadline: float) -> bool:
    """Retry transport errors and 5xx; give up on a terminal 3xx/4xx.

    A rejected policy (403 from an expired signature, 400 from a condition
    mismatch) is rejected identically every time, and a 301 means a wrong-region
    bucket -- ``follow_redirects`` is off, and retrying resolves neither.
    """
    import httpx  # noqa: PLC0415 -- see _new_client

    if time.monotonic() >= deadline:
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if 300 <= status < 500:
            return status in (408, 429)
        return True
    return isinstance(exc, httpx.HTTPError)


def _s3_error_code(response: httpx.Response) -> str:
    """The ``<Code>`` from an S3 error body, or ``""``.

    Only the code, never the whole body: S3's error XML carries ``BucketName``,
    ``RequestId`` and ``HostId``, and the code alone is what's diagnostic. It
    feeds the server-side log, not the caller-facing exception.
    """
    match = _S3_ERROR_CODE_RE.search((response.text or "")[:4096])
    return match.group(1) if match else ""


def _new_client() -> httpx.Client:
    # Local, not module scope: `fal` is pickled by value, so a module-level
    # import becomes a load-time requirement of every isolated environment and
    # any @fal.function returning a File fails to deserialize. The noqa is
    # legitimate here: PLC0415 is enforced on this file to stop lazy imports of
    # `fal` itself, which may be absent on a runner. httpx is a declared
    # dependency, kept in SERVE_REQUIREMENTS, so on a served app, the only place
    # this runs, the import cannot fail.
    import httpx  # noqa: PLC0415

    return httpx.Client(
        timeout=httpx.Timeout(
            UPLOAD_POLICY_TIMEOUT_SOCKET, connect=UPLOAD_POLICY_TIMEOUT_CONNECT
        ),
        follow_redirects=False,
    )


@dataclass
class _UploadFailure:
    """One shape for every *upload* failure record, so its JSON has a fixed key
    set. The other records here (cleanup, drain) carry their own fields."""

    request_id: str | None
    status: int | None = None
    s3_error: str | None = None
    error: str | None = None
    error_type: str | None = None


def report_json_line(event: str, **fields: Any) -> None:
    """Write one JSON record to the runner's stdout, or nothing.

    ``fal.logging`` is not an option: its processor raises ``DropEvent`` unless
    ``set_debug_logging(True)``, which has no callers. Serialize before writing
    and write once, since ``print(obj)`` emits text and newline separately and a
    concurrent write from the request thread would split the record.
    """
    try:
        print(json.dumps({event: fields}) + "\n", end="", flush=True)
    except Exception:
        pass


def _report_failure(failure: _UploadFailure) -> None:
    """Record a failed upload. Only bounded fields we produced go in, never the
    S3 response body, which carries the caller's bucket name and host id."""
    report_json_line("upload_policy_failure", **asdict(failure))


def _attempt_upload(
    post: Callable[[httpx.Client], httpx.Response],
    request_id: str | None = None,
) -> None:
    """One upload, with retries. Raises ``UploadPolicyError`` on failure.

    The client is built per upload: this module is cloudpickled by value and a
    live ``httpx.Client`` holds an unpicklable ``SSLContext``, so a module global
    would make ``File`` unserializable. Hand-rolled rather than
    ``fal.toolkit.utils.retry``, whose stdout output would be filed under
    whichever request happens to be in flight.
    """
    import httpx  # noqa: PLC0415 -- see _new_client

    deadline = time.monotonic() + UPLOAD_POLICY_TOTAL_DEADLINE
    delay = _BASE_DELAY

    with _new_client() as client:
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = post(client)
                # A policy carrying success_action_redirect -- a normal field of
                # generate_presigned_post -- makes S3 answer 303 *after* storing
                # the object, and raise_for_status treats any non-2xx as error.
                if response.status_code != 303:
                    response.raise_for_status()
                return
            except (httpx.HTTPError, httpx.InvalidURL, OSError) as exc:
                last = exc
                if attempt == _MAX_ATTEMPTS or not _should_retry(exc, deadline):
                    break
                time.sleep(delay * random.uniform(0.5, 1.5))
                delay = min(delay * 2, _MAX_DELAY)

    # Keep the message generic: this runs fire-and-forget (_run swallows the
    # exception; the detail is reported just below), and the exported upload_*
    # entry points mean caller-chosen foreign text -- status line, reason phrase,
    # error body -- must not be baked into it.
    if isinstance(last, httpx.HTTPStatusError):
        status = last.response.status_code
        _report_failure(
            _UploadFailure(
                request_id=request_id,
                status=status,
                s3_error=_s3_error_code(last.response) or None,
            )
        )
        raise UploadPolicyError(
            f"Upload via {UPLOAD_POLICY_KEY} failed with status {status}."
        ) from last
    _report_failure(_UploadFailure(request_id=request_id, error=str(last)))
    raise UploadPolicyError(f"Upload via {UPLOAD_POLICY_KEY} failed.") from last


# --- background execution ---------------------------------------------
# Daemon threads, not a ThreadPoolExecutor: its workers are non-daemon and
# concurrent.futures joins them at exit with no timeout, so a stalled
# destination would block interpreter exit indefinitely.

# Two bounds on queued work: a thread count (fds, OS threads) and a byte total
# (RAM). Only large from_path outputs stream a disk-staged file and stay
# budget-free. Refused rather than queued when full; waiting would restore the
# runner-hold that backgrounding exists to avoid.
UPLOAD_POLICY_MAX_PENDING = 64
UPLOAD_POLICY_MAX_PENDING_BYTES = 512 * 1024 * 1024

_pending_bytes = 0
_state_lock = threading.Lock()
_inflight: set[threading.Thread] = set()


def _reset_after_fork() -> None:
    """A forked child inherits this module's locks and bookkeeping but none of
    the threads that would clear them, so the budgets could arrive already
    spent and ``drain`` would wait on threads that do not exist."""
    global _pending_bytes, _state_lock  # noqa: PLW0603 -- after fork
    _pending_bytes = 0
    _state_lock = threading.Lock()
    _inflight.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _submit(
    post: Callable[[httpx.Client], httpx.Response],
    request_id: str | None,
    nbytes: int,
    cleanup: Callable[[], None] | None = None,
) -> None:
    global _pending_bytes  # noqa: PLW0603 -- module-level queue budget

    def _run() -> None:
        global _pending_bytes  # noqa: PLW0603 -- module-level queue budget
        try:
            _attempt_upload(post, request_id)
        except UploadPolicyError:
            # Expected failure, already reported by _attempt_upload as one JSON
            # line. Swallowed so a multi-line traceback does not follow it: these
            # records do land on whichever request's stdout is open, so each one
            # carries its own request_id to be re-attributed later.
            pass
        except Exception as exc:
            # A bug here, not a rejected upload, so it carries the type.
            _report_failure(
                _UploadFailure(
                    request_id=request_id,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            )
        finally:
            # After every attempt, not each one: a retry rewinds the staged handle.
            _safe_cleanup(cleanup, request_id)
            with _state_lock:
                _pending_bytes -= nbytes
                _inflight.discard(threading.current_thread())

    thread = threading.Thread(target=_run, name="fal-upload-policy", daemon=True)

    with _state_lock:
        if len(_inflight) >= UPLOAD_POLICY_MAX_PENDING:
            refusal: str | None = "too many uploads still in flight"
        # nbytes=0 (disk-staged) and lone uploads are always admitted; the byte
        # budget only sheds additional concurrent in-memory work.
        elif (
            nbytes
            and _pending_bytes
            and _pending_bytes + nbytes > UPLOAD_POLICY_MAX_PENDING_BYTES
        ):
            refusal = "too much upload data still in flight"
        else:
            refusal = None
            _pending_bytes += nbytes
            _inflight.add(thread)

    if refusal is not None:
        _safe_cleanup(cleanup, request_id)
        raise UploadPolicyError(
            f"{UPLOAD_POLICY_KEY}: {refusal}; the destination is not keeping up."
        )

    try:
        thread.start()
    except RuntimeError as exc:
        # Runner out of thread capacity. Return the budget and drop the
        # never-started thread, else drain() raises joining it and skips
        # teardown. Same 424 as the queue-full refusal: not keeping up, not a 500.
        with _state_lock:
            _inflight.discard(thread)
            _pending_bytes -= nbytes
        _safe_cleanup(cleanup, request_id)
        raise UploadPolicyError(
            f"{UPLOAD_POLICY_KEY}: could not start upload; the runner is out of "
            "thread capacity."
        ) from exc


def _safe_cleanup(cleanup: Callable[[], None] | None, request_id: str | None) -> None:
    """Best-effort staged-file cleanup that never raises: a failed unlink must
    not skip a slot release or mask the UploadPolicyError a caller is raising."""
    if cleanup is None:
        return
    try:
        cleanup()
    except Exception as exc:
        report_json_line(
            "upload_policy_cleanup_failed", request_id=request_id, error=str(exc)
        )


def drain(timeout: float | None = UPLOAD_POLICY_DRAIN_TIMEOUT) -> None:
    """Give in-flight uploads a bounded chance to finish. Called at teardown.

    Best effort by construction: the threads are daemons, so whatever has not
    finished when the process goes away is lost. The alternative -- non-daemon
    threads -- trades that for an unbounded hang at interpreter exit.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    with _state_lock:
        threads = list(_inflight)
    for thread in threads:
        if not thread.ident:
            continue  # never started; joining would raise
        if deadline is None:
            thread.join()
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        thread.join(remaining)
    # Daemon threads: anything still running is dropped when the process exits.
    # Reported so a customer's "my output never arrived" has a breadcrumb, with
    # the pending count as its denominator.
    still_running = [thread for thread in threads if thread.is_alive()]
    if still_running:
        report_json_line(
            "upload_policy_abandoned_at_shutdown",
            unfinished=len(still_running),
            pending=len(threads),
            timeout=timeout,
        )


def _current_request_id() -> str | None:
    return getattr(_current_request(), "request_id", None)


def upload_bytes_with_policy(
    policy: UploadPolicy,
    file_name: str,
    data: bytes,
    content_type: str,
) -> str:
    """Queue ``data`` for upload and return the URL it will appear at.

    Validation raises inline; the POST does not. See the module docstring for
    what that means for the caller.
    """
    _validate_size(len(data))
    access_url, fields = _prepare_upload(policy, file_name, content_type)

    def _post(client: httpx.Client) -> httpx.Response:
        return client.post(
            policy.url,
            data=fields,
            files={"file": (file_name, data, content_type)},
        )

    _submit(_post, _current_request_id(), nbytes=len(data))
    return access_url


def upload_path_with_policy(
    policy: UploadPolicy,
    file_path: Path,
    file_name: str,
    content_type: str,
) -> str:
    """As :func:`upload_bytes_with_policy`, for a file on disk.

    Staged with a copy, not a hardlink: the caller may overwrite the path in
    place (``open(path, "wb")``) before the background upload runs, which a
    shared inode would let corrupt it. A copy snapshots the bytes now.

    The retained handle, not the name, is what the POST reads, because apps
    routinely drop the ``TemporaryDirectory`` they wrote into the moment
    ``from_path`` returns; reopening by name would leave a 200 with a dead URL.
    Runners are POSIX, where the snapshot is unlinked as soon as it is staged,
    so a runner killed during the upload strands nothing. Windows cannot unlink
    an open file, so there the name lingers until the handle closes.
    """
    _validate_size(file_path.stat().st_size)
    access_url, fields = _prepare_upload(policy, file_name, content_type)
    request_id = _current_request_id()

    # Stage on the source's own volume, not the default temp dir: the payload can
    # be large (up to 5 GB) and belongs next to where it was written rather than
    # on a possibly small /tmp.
    handle = NamedTemporaryFile(dir=file_path.parent)

    def _cleanup() -> None:
        try:
            handle.close()
        except FileNotFoundError:
            # Already unlinked, by us below or by a caller that dropped the
            # directory; the closer unlinks unguarded before 3.12.
            pass

    try:
        with open(file_path, "rb") as source:
            shutil.copyfileobj(source, handle)
        handle.flush()
        if os.name != "nt":
            os.unlink(handle.name)
    except Exception:
        _safe_cleanup(_cleanup, request_id)
        raise

    def _post(client: httpx.Client) -> httpx.Response:
        # Rewound per attempt: a retry cannot reuse a consumed file object.
        handle.seek(0)
        return client.post(
            policy.url,
            data=fields,
            files={"file": (file_name, handle, content_type)},
        )

    try:
        _submit(
            _post,
            request_id,
            # Staged on disk and streamed, so it costs no queue budget.
            nbytes=0,
            cleanup=_cleanup,
        )
    except Exception:
        _safe_cleanup(_cleanup, request_id)
        raise
    return access_url


__all__ = [
    "UPLOAD_POLICY_KEY",
    "UPLOAD_POLICY_MAX_BYTES",
    "UploadPolicy",
    "UploadPolicyError",
    "UploadPolicyInputError",
    "drain",
    "get_upload_policy",
    "parse_upload_policy",
    "upload_bytes_with_policy",
    "upload_path_with_policy",
]
