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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, List, Tuple, Union
from urllib.parse import quote, unquote, urlparse
from uuid import uuid4

import httpx

from fal.exceptions import AppException, FieldException
from fal.logging import get_logger
from fal.ref import get_current_app
from fal.toolkit.exceptions import FileUploadException

logger = get_logger(__name__)

UPLOAD_POLICY_KEY = "x-app-fal-upload-policy"
UPLOAD_POLICY_FILENAME_PLACEHOLDER = "${filename}"
# S3 browser-POST uploads are a single request capped at 5 GB.
UPLOAD_POLICY_MAX_BYTES = 5 * 1024 * 1024 * 1024

# Per socket operation, not a wall-clock cap: a destination trickling bytes
# under the read timeout is bounded by neither this nor the deadline below.
UPLOAD_POLICY_TIMEOUT = httpx.Timeout(300.0, connect=10.0)
# Bounds retries of one file, checked between attempts. Not a per-request cap.
UPLOAD_POLICY_TOTAL_DEADLINE = 300.0
# How long teardown waits for in-flight uploads before abandoning them.
UPLOAD_POLICY_DRAIN_TIMEOUT = 5.0

# Same retry shape the registry uses for this POST.
_MAX_ATTEMPTS = 5
_BASE_DELAY = 1
_MAX_DELAY = 30

_S3_ERROR_CODE_RE = re.compile(r"<Code>([A-Za-z0-9]{1,64})</Code>")

_UNSET = object()

_ERROR_TYPE = "input_value_error"
_ERROR_DOC_URL = f"https://docs.fal.ai/errors#{_ERROR_TYPE}"

# Anchored so that only a real S3 endpoint matches. An unanchored ".s3." test
# also admits S3 PrivateLink names (``vpce-….s3.<region>.vpce.amazonaws.com``),
# whose zone is publicly delegated and resolves to a *caller-chosen* VPC CIDR --
# i.e. arbitrary RFC1918 target selection through the allowlist that exists to
# prevent exactly that. The registry's copy is still unanchored and needs the
# same fix.
# The qualifier list is explicit rather than a wildcard label, so the regex
# rejects the two-label vpce forms (…s3.<region>.vpce.amazonaws.com); the
# single-label form (s3.vpce.amazonaws.com) is caught by the denylist in
# _is_s3_upload_policy_host.
_S3_HOST_RE = re.compile(
    r"(?:[a-z0-9][a-z0-9.-]*\.)?"  # optional bucket
    r"s3(?:express-[a-z0-9-]+)?"  # S3 Express glues the az on with no separator
    r"(?:[.-](?:dualstack|accelerate|accesspoint|object-lambda|outposts"
    r"|fips|website))?"  # optional qualifier
    r"(?:[.-][a-z0-9-]+)?"  # optional region
    r"\.amazonaws\.com(?:\.cn)?"
)


class UploadPolicyInputError(FieldException):
    """Malformed policy header. 422, with registry's body shape."""

    def __init__(self, message: str) -> None:
        super().__init__(
            field=UPLOAD_POLICY_KEY,
            message=message,
            status_code=422,
            type=_ERROR_TYPE,
        )
        # FieldException is kw-only, so Exception.args stays empty and str(exc)
        # would render as "" in tracebacks/logs without this.
        self.args = (message,)

    def to_pydantic_format(self) -> dict[str, list[dict]]:
        # Matches registry's InputValueError exactly: loc is ["body"] with no
        # field element, plus the url/input keys.
        return {
            "detail": [
                {
                    "loc": ["body"],
                    "msg": self.message,
                    "type": self.type,
                    "url": _ERROR_DOC_URL,
                    "input": None,
                }
            ]
        }


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


def _is_s3_upload_policy_host(host: str) -> bool:
    """Whether ``host`` is an S3 endpoint we will POST caller data to.

    Anchored allowlist; see ``_S3_HOST_RE`` for why. The region label is not
    validated, so an unreachable name under AWS DNS passes -- validating it
    would break the first customer in a newly launched region, which is worse.
    """
    # Reject the S3 PrivateLink zone outright. Its DNS is publicly delegated and
    # resolves to a caller-chosen VPC CIDR (RFC1918) -- the SSRF this allowlist
    # exists to block -- and the regex's region label would otherwise admit a
    # bare "vpce" (e.g. s3.vpce.amazonaws.com).
    if host.endswith(".vpce.amazonaws.com") or host.endswith(".vpce.amazonaws.com.cn"):
        return False
    return bool(_S3_HOST_RE.fullmatch(host))


def _reserved_name(name: str) -> str:
    """Fold a field name the way S3 matches them: case- and space-insensitive."""
    return name.strip().lower()


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
    try:
        parsed_url = urlparse(url)
        host = (parsed_url.hostname or "").lower().rstrip(".")
        port = parsed_url.port
    except ValueError as exc:
        # urlparse raises on, among others, hostnames that change under NFKC
        # normalization. Bad client input, not a server fault.
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} 'url': could not be parsed ({exc})"
        ) from exc

    if parsed_url.scheme.lower() != "https" or not _is_s3_upload_policy_host(host):
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
    decoded_path = unquote(parsed_url.path).replace("\\", "/")
    if any(seg in (".", "..") for seg in decoded_path.split("/")):
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} 'url': must not contain '.' or '..' "
            "path segments"
        )
    if "?" in url or "#" in url:
        # The access URL is built by appending the key, so a query or fragment
        # would land in the middle of it.
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} 'url': must not have a query or fragment"
        )


def _validate_multipart_value(name: str, value: str) -> None:
    # Load-bearing. httpx percent-encodes field *names* and filenames, but
    # writes field values and a part's Content-Type verbatim. Values are safe
    # only because the boundary is unguessable; a Content-Type is a real header,
    # so a CRLF here injects headers into the part.
    if "\r" in value or "\n" in value:
        raise UploadPolicyInputError(f"Invalid multipart {name}: contains CR/LF")


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

    Returns ``None`` when the header is absent, or when its value is not a
    string at all (see below). Raises
    ``UploadPolicyInputError`` (422) on anything malformed -- including a blank
    value, which is a policy the caller failed to build rather than a request
    to use the fal CDN. Silently falling back there would be the one failure
    mode nobody can detect afterwards.
    """
    raw = _headers_get(headers, UPLOAD_POLICY_KEY)
    if raw is None:
        return None
    if not isinstance(raw, (str, bytes)):
        # A Mock stands in for a request in app test suites; treating it as a
        # malformed policy would 422 every one of them. Anything else non-string
        # is real malformed input and must not silently reach the fal CDN.
        if type(raw).__module__.startswith("unittest.mock"):
            return None
        raise UploadPolicyInputError(
            f"Invalid {UPLOAD_POLICY_KEY} header: must be a string"
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

        # key and Content-Type must be real strings -- they are compared and
        # substituted, not just forwarded, and registry requires the same.
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
    # URL we return would name a different object -- a 200 with a dead URL, the
    # exact failure this module exists to avoid.
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

    The middleware parses and caches the policy upfront; this reads that cache,
    falling back to header parsing only for a bare request (see the inline note).

    ``current_request`` is a ContextVar, propagated by ``fal.compat.run_in_thread``
    (so the async constructors resolve). An app's own ``ThreadPoolExecutor`` and
    WebSocket/realtime endpoints (no HTTP middleware) don't see it -- output goes
    to the fal CDN. Both are documented as unsupported.
    """
    if request is None:
        request = _current_request()

    if request is None:
        return None
    # Middleware (RequestContext) parses once per request and caches the result
    # here, as None or an UploadPolicy. A bare fastapi Request has no such
    # attribute and falls through to parsing its headers. A Mock answers getattr
    # with another Mock, which is neither -- also fall through, where its Mock
    # headers parse to None.
    cached = getattr(request, "upload_policy", _UNSET)
    if cached is None or isinstance(cached, UploadPolicy):
        return cached
    return parse_upload_policy(getattr(request, "headers", {}))


def _validate_size(nbytes: int) -> None:
    # This and the other checks in _prepare_upload run inside from_bytes/from_path,
    # i.e. after the generation. They raise UploadPolicyInputError (422), which
    # bills the caller by default -- intended: the GPU work was done, and only
    # the output size / filename / signed Content-Type, none knowable earlier,
    # is wrong. Pre-generation malformed headers are caught in the middleware and
    # billed zero.
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

    # Not a re-parse of the header: the middleware already parsed it upfront and
    # rejected a malformed policy before generation. This re-checks only the URL
    # allowlist (SSRF) and key template because the upload_* entry points are
    # exported -- a caller can hand-build an UploadPolicy that never went through
    # parse_upload_policy, and those two must hold however it got here.
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

    encoded_key = quote(final_key.lstrip("/"), safe="/~")
    access_url = f"{policy.url.rstrip('/')}/{encoded_key}"
    return access_url, fields


def _should_retry(exc: Exception, deadline: float) -> bool:
    """Retry transport errors and 5xx; give up on a terminal 3xx/4xx.

    A rejected policy (403 from an expired signature, 400 from a condition
    mismatch) is rejected identically every time, and a 301 means a wrong-region
    bucket -- ``follow_redirects`` is off, and retrying resolves neither.
    """
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
    return httpx.Client(timeout=UPLOAD_POLICY_TIMEOUT, follow_redirects=False)


def _attempt_upload(
    post: Callable[[httpx.Client], httpx.Response],
    request_id: str | None = None,
) -> None:
    """One upload, with retries. Raises ``UploadPolicyError`` on failure.

    The client is built per upload, not at module scope: this module is
    cloudpickled by value (``include_module("fal")``) and a live ``httpx.Client``
    holds an unpicklable ``SSLContext``, so a module global would make ``File``
    unserializable. One client per file still reuses the connection across that
    file's retries.

    Hand-rolled, not ``fal.toolkit.utils.retry``: that decorator prints progress
    and tracebacks to stdout, and on this background thread that output would be
    filed under whichever request is in flight (see ``_submit``).
    """
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
    # exception; the detail is logged just below), and the exported upload_* entry
    # points mean caller-chosen foreign text -- status line, reason phrase, error
    # body -- must not be baked into it.
    if isinstance(last, httpx.HTTPStatusError):
        status = last.response.status_code
        logger.warning(
            "upload policy upload failed",
            request_id=request_id,
            status=status,
            s3_error=_s3_error_code(last.response) or None,
        )
        raise UploadPolicyError(
            f"Upload via {UPLOAD_POLICY_KEY} failed with status {status}."
        ) from last
    logger.warning(
        "upload policy upload failed",
        request_id=request_id,
        error=str(last),
    )
    raise UploadPolicyError(f"Upload via {UPLOAD_POLICY_KEY} failed.") from last


# --- background execution ---------------------------------------------
# Daemon threads, not a ThreadPoolExecutor: its workers are non-daemon and
# concurrent.futures joins them at exit with no timeout, so a stalled
# destination would block interpreter exit indefinitely.

# Two independent bounds on queued work: a thread count (fds and OS threads,
# both paths) and a byte total (RAM). in-memory uploads count against the byte
# total -- from_bytes, and small from_path outputs, which the caller keeps
# resident in File.file_data anyway; only large from_path outputs stream a
# disk-staged file and stay budget-free. Refused rather than queued when full;
# waiting would put back the runner-hold that backgrounding exists to avoid.
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
            # Expected failure, already logged in _attempt_upload. Swallowed so a
            # daemon-thread traceback doesn't hit stdout and mis-file under the
            # in-flight request (often a different tenant).
            pass
        except Exception as exc:
            # structlog, not print(): same stdout-attribution reason as above.
            # structlog drops it unless debugging.
            logger.warning(
                "upload policy upload crashed",
                request_id=request_id,
                error=str(exc),
            )
        finally:
            # After every attempt, not each one: a retry reopens the staged file.
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
        # "can't start new thread" -- a runner out of thread capacity. Return
        # the budget (else it leaks) and drop the never-started thread (else
        # drain() would raise joining it, escaping the lifespan and skipping
        # teardown). Surface it as the same 424 the queue-full refusal uses,
        # since it is the same "not keeping up" condition -- not a 500.
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
        logger.warning(
            "upload policy staged-file cleanup failed",
            request_id=request_id,
            error=str(exc),
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
    if threads:
        logger.info(
            "upload policy draining in-flight uploads",
            pending=len(threads),
            timeout=timeout,
        )
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
    # Log it so a customer's "my output never arrived" has a breadcrumb.
    still_running = [thread for thread in threads if thread.is_alive()]
    if still_running:
        logger.warning(
            "upload policy drain left uploads unfinished at shutdown",
            unfinished=len(still_running),
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
    """
    _validate_size(file_path.stat().st_size)
    access_url, fields = _prepare_upload(policy, file_name, content_type)

    # Stage on the source's own volume, not the default temp dir: the payload can
    # be large (up to 5 GB) and belongs next to where it was written rather than
    # on a possibly small /tmp.
    with NamedTemporaryFile(dir=file_path.parent, delete=False) as handle:
        staged = Path(handle.name)
    try:
        shutil.copyfile(file_path, staged)
    except Exception:
        staged.unlink(missing_ok=True)
        raise

    def _post(client: httpx.Client) -> httpx.Response:
        # Reopened per attempt; a retry cannot reuse a consumed file object.
        with open(staged, "rb") as source:
            return client.post(
                policy.url,
                data=fields,
                files={"file": (file_name, source, content_type)},
            )

    try:
        _submit(
            _post,
            _current_request_id(),
            # Staged on disk and streamed, so it costs no queue budget.
            nbytes=0,
            cleanup=lambda: staged.unlink(missing_ok=True),
        )
    except Exception:
        staged.unlink(missing_ok=True)
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
