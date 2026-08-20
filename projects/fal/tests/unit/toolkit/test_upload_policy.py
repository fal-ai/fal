from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from fal.exceptions import AppException
from fal.toolkit.exceptions import FileUploadException
from fal.toolkit.file import _upload_policy as up
from fal.toolkit.file._upload_policy import (
    UPLOAD_POLICY_KEY,
    UploadPolicy,
    UploadPolicyError,
    UploadPolicyInputError,
    drain,
    parse_upload_policy,
    upload_bytes_with_policy,
    upload_path_with_policy,
)

VECTORS = json.loads((Path(__file__).parent / "upload_policy_vectors.json").read_text())
HEADER_VECTORS = [
    case for case in VECTORS["parse"] + VECTORS["prepare"] if "header" in case
]

VALID_POLICY = UploadPolicy(
    url="https://bucket.s3.us-west-1.amazonaws.com/",
    fields={"key": "uploads/${filename}", "policy": "b64", "x-amz-signature": "sig"},
)


class _StubClient:
    """Stand-in for the per-upload httpx.Client: a context manager whose .post
    is the supplied fake."""

    def __init__(self, post):
        self.post = post
        self.follow_redirects = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """Keep retry tests instant."""
    monkeypatch.setattr(up, "_BASE_DELAY", 0)


@pytest.mark.parametrize(
    "case", HEADER_VECTORS, ids=[c["name"] for c in HEADER_VECTORS]
)
def test_shared_vectors(case):
    """The decision table shared with registry/cdn.py's implementation.

    The SDK enforces all of these at parse; registry splits them across parse
    and prepare, hence the sections in the JSON. Cases flagged ``sdk_only``
    differ from registry in final outcome and are the live divergence list.
    """
    headers = {UPLOAD_POLICY_KEY: case["header"]}
    if case["accepted"]:
        assert parse_upload_policy(headers) is not None
    else:
        # The messages are literals, not patterns -- several contain regex
        # metacharacters (`${filename}`).
        with pytest.raises(UploadPolicyInputError, match=re.escape(case["message"])):
            parse_upload_policy(headers)


def test_every_reject_vector_pins_a_message():
    """Otherwise a case can pass for an unrelated reason."""
    for case in HEADER_VECTORS:
        if not case["accepted"]:
            assert case.get("message"), case["name"]


def test_only_an_absent_header_is_ignored():
    """A blank value is a policy the caller failed to build, not a request for
    the fal CDN. Silently using fal storage is the one failure mode nobody can
    detect afterwards."""
    assert parse_upload_policy({}) is None
    assert parse_upload_policy(None) is None

    for blank in ("", "   "):
        with pytest.raises(UploadPolicyInputError):
            parse_upload_policy({UPLOAD_POLICY_KEY: blank})


def test_a_non_string_header_is_rejected_not_ignored():
    """Only a Mock gets the pass; anything else non-string is real malformed
    input and must not silently fall through to the fal CDN."""
    with pytest.raises(UploadPolicyInputError, match="must be a string"):
        parse_upload_policy({UPLOAD_POLICY_KEY: 42})

    assert parse_upload_policy({UPLOAD_POLICY_KEY: MagicMock()}) is None


def test_header_lookup_is_case_insensitive():
    header = json.dumps(
        {
            "url": "https://bucket.s3.us-west-1.amazonaws.com/",
            "fields": {"key": "uploads/${filename}"},
        }
    )
    assert parse_upload_policy({UPLOAD_POLICY_KEY.upper(): header}) is not None


class TestHostAllowlist:
    @pytest.mark.parametrize(
        "host",
        [
            "s3.amazonaws.com",
            "bucket.s3.amazonaws.com",
            "s3.us-west-1.amazonaws.com",
            "bucket.s3.us-west-1.amazonaws.com",
            "bucket.s3.dualstack.us-west-1.amazonaws.com",
            "my.dotted.bucket.s3.us-east-1.amazonaws.com",
            "bucket.s3.cn-north-1.amazonaws.com.cn",
            "bucket.s3-accelerate.amazonaws.com",
            "s3-us-west-2.amazonaws.com",
            # generate_presigned_post emits these for an access-point ARN and
            # for S3 Express; rejecting them turns away valid destinations.
            "my-ap-123456.s3-accesspoint.us-east-1.amazonaws.com",
            "bkt--use1-az4--x-s3.s3express-use1-az4.us-east-1.amazonaws.com",
            "bucket.s3-object-lambda.us-east-1.amazonaws.com",
        ],
    )
    def test_accepts_real_s3_endpoints(self, host):
        assert up._is_s3_upload_policy_host(host)

    @pytest.mark.parametrize(
        "host",
        [
            # PrivateLink: a publicly delegated zone resolving to a
            # caller-chosen VPC CIDR, i.e. arbitrary RFC1918 target selection
            # through the allowlist that exists to prevent exactly that.
            "vpce-0a1b2c3d4e5f-ghijk.s3.us-east-1.vpce.amazonaws.com",
            "bucket.vpce-0a1b-ghij.s3.us-east-1.vpce.amazonaws.com",
            "bucket.s3.amazonaws.com.evil.com",
            "evil.example.com",
            "169.254.169.254",
            "s3.amazonaws.com.cn.evil.com",
            "s3.us-east-1.vpce.amazonaws.com",
            # Single-label vpce forms: the regex region group admits a bare
            # "vpce", so the explicit zone denylist -- not the regex -- rejects
            # these.
            "s3.vpce.amazonaws.com",
            "bucket.s3.vpce.amazonaws.com",
            "bucket.s3.accesspoint.vpce.amazonaws.com",
            # "$" would match before a trailing newline; the anchor is fullmatch.
            "s3.amazonaws.com\n",
        ],
    )
    def test_rejects_everything_else(self, host):
        assert not up._is_s3_upload_policy_host(host)


class TestErrorShape:
    def test_is_a_422(self):
        with pytest.raises(UploadPolicyInputError) as excinfo:
            parse_upload_policy({UPLOAD_POLICY_KEY: "{not json"})

        assert excinfo.value.status_code == 422

    def test_body_matches_registrys_exactly(self):
        """Callers parse these bodies; two shapes for one header is a break."""
        with pytest.raises(UploadPolicyInputError) as excinfo:
            parse_upload_policy({UPLOAD_POLICY_KEY: "{not json"})

        detail = excinfo.value.to_pydantic_format()["detail"]
        assert len(detail) == 1
        assert detail[0]["loc"] == ["body"]
        assert detail[0]["type"] == "input_value_error"
        assert detail[0]["input"] is None
        assert detail[0]["url"].endswith("#input_value_error")
        assert "not valid JSON" in detail[0]["msg"]

    def test_str_is_not_empty(self):
        """A dataclass exception with no args logs as '' in tracebacks."""
        assert "not valid JSON" in str(
            UploadPolicyInputError("Invalid header: not valid JSON")
        )

    @pytest.mark.parametrize(
        "header",
        [
            '{"url": "https://b.s3.us-east-1.amazonaws.com/",'
            ' "fields": {"key": "u/${filename}", "x-\\ud800-meta": "v"}}',
            '{"url": "https://b.s3.us-east-1.amazonaws.com/",'
            ' "fields": {"key": "u/${filename}", "x-\\ud800-meta": 5}}',
            '{"url": "https://b.s3.us-east-1.amazonaws.com/",'
            ' "fields": {"key": "u/${filename}", "x": ["\\ud800"]}}',
        ],
        ids=["surrogate name", "surrogate name + bad value", "surrogate nested"],
    )
    def test_the_error_itself_is_always_renderable(self, header):
        """An un-encodable field name must not reach the message.

        JSONResponse encodes with ensure_ascii=False, so a surrogate in the
        message kills the exception handler and turns this 422 into a 500 --
        the exact fault _require_encodable exists to prevent.
        """
        with pytest.raises(UploadPolicyInputError) as excinfo:
            parse_upload_policy({UPLOAD_POLICY_KEY: header})

        json.dumps(excinfo.value.to_pydantic_format(), ensure_ascii=False).encode(
            "utf-8"
        )


class TestUploadFailureSurfacing:
    """The error object itself. Under fire-and-forget it does not reach the
    caller for an upload failure -- it is logged -- so these exercise
    ``_attempt_upload`` directly. The status still matters: ``_submit`` raises
    synchronously when the queue is full."""

    def test_is_a_424_carrying_the_reason(self):
        exc = UploadPolicyError("bucket said no")

        assert exc.status_code == 424
        assert exc.message == "bucket said no"
        assert str(exc) == "bucket said no"

    def test_is_still_catchable_as_a_file_upload_exception(self):
        assert isinstance(UploadPolicyError("x"), FileUploadException)

    def test_the_app_renders_a_synchronous_one_as_a_424(self):
        """isinstance alone does not prove dispatch: dropping the AppException
        base leaves status_code working while the handler stops matching."""
        app = FastAPI()

        @app.exception_handler(AppException)
        async def _handler(request, exc: AppException):  # matches api.py
            return JSONResponse({"detail": exc.message}, exc.status_code)

        @app.get("/")
        def _endpoint():
            raise UploadPolicyError("bucket said no")

        response = TestClient(app, raise_server_exceptions=False).get("/")

        assert response.status_code == 424
        assert response.json() == {"detail": "bucket said no"}

    def test_the_message_carries_only_the_status_no_foreign_text(self):
        """The message reflects nothing but the bare status -- no S3 code, no
        body identifiers."""
        body = (
            "<Error><Code>AccessDenied</Code><BucketName>fal-internal</BucketName>"
            "<RequestId>ABC123</RequestId><HostId>SECRET==</HostId></Error>"
        )

        def fake_post(client):
            return httpx.Response(
                403, text=body, request=httpx.Request("POST", VALID_POLICY.url)
            )

        with pytest.raises(UploadPolicyError) as excinfo:
            up._attempt_upload(fake_post)

        message = str(excinfo.value)
        assert message == f"Upload via {UPLOAD_POLICY_KEY} failed with status 403."
        for foreign in ("AccessDenied", "fal-internal", "ABC123", "SECRET=="):
            assert foreign not in message

    def test_the_diagnostic_detail_lands_in_the_server_side_log(self, monkeypatch):
        """The other half of the relocation: what the message drops, the log must
        keep, or a debugger is left with only a bare status."""
        events = MagicMock()
        monkeypatch.setattr(up, "logger", events)

        def fake_post(client):
            return httpx.Response(
                403,
                text="<Error><Code>AccessDenied</Code></Error>",
                request=httpx.Request("POST", VALID_POLICY.url),
            )

        with pytest.raises(UploadPolicyError):
            up._attempt_upload(fake_post)

        assert any(
            call.kwargs.get("status") == 403
            and call.kwargs.get("s3_error") == "AccessDenied"
            for call in events.warning.call_args_list
        )

    def test_does_not_echo_the_destinations_reason_phrase(self):
        """httpx takes reason_phrase off the wire, so a destination could smuggle
        bytes into it; the message must not reflect it."""

        def fake_post(client):
            return httpx.Response(
                403,
                extensions={"reason_phrase": b"Forbidden LEAKED-SECRET AKIAX"},
                request=httpx.Request("POST", VALID_POLICY.url),
            )

        with pytest.raises(UploadPolicyError) as excinfo:
            up._attempt_upload(fake_post)

        assert "LEAKED-SECRET" not in str(excinfo.value)
        assert str(excinfo.value) == (
            f"Upload via {UPLOAD_POLICY_KEY} failed with status 403."
        )

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            ("<Error><Code></Code></Error>", ""),
            (
                "<Error><Code>XAmzContentSHA256Mismatch</Code></Error>",
                "XAmzContentSHA256Mismatch",
            ),
            ("x" * 5000 + "<Error><Code>TooLate</Code></Error>", ""),
        ],
        ids=["empty code", "digit-bearing code", "code past the scan window"],
    )
    def test_error_code_extraction_edges(self, body, expected):
        # Extraction is unit-tested directly: the code feeds the server-side log,
        # not the caller-facing message.
        response = httpx.Response(
            403, text=body, request=httpx.Request("POST", VALID_POLICY.url)
        )
        assert up._s3_error_code(response) == expected


class TestPrepareUpload:
    def test_substitutes_filename_and_builds_url(self):
        url, fields = up._prepare_upload(VALID_POLICY, "cat.png", "image/png")

        assert fields["key"].startswith("uploads/")
        assert fields["key"].endswith("-cat.png")
        assert "${filename}" not in fields["key"]
        assert url == f"https://bucket.s3.us-west-1.amazonaws.com/{fields['key']}"

    def test_preserves_signed_fields_and_does_not_mutate_the_policy(self):
        _, fields = up._prepare_upload(VALID_POLICY, "cat.png", "image/png")

        assert fields["policy"] == "b64"
        assert fields["x-amz-signature"] == "sig"
        assert VALID_POLICY.fields["key"] == "uploads/${filename}"

    def test_injects_content_type_when_absent(self):
        _, fields = up._prepare_upload(VALID_POLICY, "cat.png", "image/png")

        assert fields["Content-Type"] == "image/png"

    def test_rejects_mismatched_explicit_content_type(self):
        """Overwriting a signed field would earn an opaque 403 from S3."""
        policy = UploadPolicy(
            url=VALID_POLICY.url,
            fields={"key": "uploads/${filename}", "Content-Type": "image/png"},
        )
        with pytest.raises(UploadPolicyInputError, match="cannot be changed"):
            up._prepare_upload(policy, "cat.jpg", "image/jpeg")

    def test_rejects_crlf_in_content_type(self):
        with pytest.raises(UploadPolicyInputError, match="CR/LF"):
            up._prepare_upload(VALID_POLICY, "cat.png", "image/png\r\nX-Evil: 1")

    def test_rejects_unencodable_content_type(self):
        """Apps forward user-supplied content types straight in."""
        with pytest.raises(UploadPolicyInputError, match="not encodable"):
            up._prepare_upload(VALID_POLICY, "cat.png", "image/\ud800")

    @pytest.mark.parametrize("file_name", ["../evil.png", "a/b.png", "a\\b.png"])
    def test_rejects_path_separators_in_the_file_name(self, file_name):
        """Client-side URL normalisation would collapse these, so the key we
        store and the URL we return would disagree."""
        with pytest.raises(UploadPolicyInputError, match="path separator"):
            up._prepare_upload(VALID_POLICY, file_name, "image/png")

    def test_rejects_a_key_template_with_a_leading_slash(self):
        """S3 stores the slash literally but URL resolution drops it, so the
        URL we return would name a different object -- 200 plus a dead URL."""
        with pytest.raises(UploadPolicyInputError, match="must not start with"):
            up._prepare_upload(
                UploadPolicy(
                    url=VALID_POLICY.url, fields={"key": "/uploads/${filename}"}
                ),
                "cat.png",
                "image/png",
            )

    @pytest.mark.parametrize("name", [" Content-Type", "content-type ", " KEY "])
    def test_folds_padded_and_cased_reserved_names(self, name):
        """One folding rule everywhere: a padded name must not slip past the
        CRLF and signed-value checks that the exact-case name gets."""
        with pytest.raises(UploadPolicyInputError):
            parse_upload_policy(
                {
                    UPLOAD_POLICY_KEY: json.dumps(
                        {
                            "url": VALID_POLICY.url,
                            "fields": {
                                "key": "uploads/${filename}",
                                name: "image/png\r\nX-Evil: 1",
                            },
                        }
                    )
                }
            )

    def test_rechecks_the_key_template_for_hand_built_policies(self):
        """The upload_* entry points are exported, so a caller can arrive with
        a policy that never went through parse_upload_policy."""
        for fields in ({}, {"key": "fixed.png"}):
            with pytest.raises(UploadPolicyInputError, match=re.escape("${filename}")):
                up._prepare_upload(
                    UploadPolicy(url=VALID_POLICY.url, fields=fields),
                    "cat.png",
                    "image/png",
                )

    @pytest.mark.parametrize(
        ("value", "on_the_wire"),
        [(True, b"true"), (False, b"false"), (5, b"5"), (1.5, b"1.5")],
        ids=repr,
    )
    def test_field_values_reach_the_wire_as_httpx_encodes_them(
        self, value, on_the_wire
    ):
        """A pre-signed policy signs its field values.

        Normalising with str() would send "True" where httpx sends "true"; the
        changed value fails the signature and S3 returns 403 for a policy that
        works against the registry. Asserted against the encoded body, not the
        dict -- an identity check would not catch the difference.
        """
        policy = UploadPolicy(
            url=VALID_POLICY.url,
            fields={"key": "uploads/${filename}", "x-amz-meta-v": value},
        )
        _, fields = up._prepare_upload(policy, "cat.png", "image/png")

        request = httpx.Request(
            "POST",
            policy.url,
            data=fields,
            files={"file": ("cat.png", b"x", "image/png")},
        )
        request.read()
        assert on_the_wire in request.content.split(b"\r\n")

    @pytest.mark.parametrize(
        "value", [["a", ["b"]], ["a", {"b": 1}], [[["deep"]]], {"a": 1}], ids=repr
    )
    def test_rejects_values_httpx_cannot_encode(self, value):
        """These raise TypeError inside httpx at upload time -- i.e. after the
        app has already done its work -- as an uncaught 500."""
        with pytest.raises(UploadPolicyInputError):
            parse_upload_policy(
                {
                    UPLOAD_POLICY_KEY: json.dumps(
                        {
                            "url": VALID_POLICY.url,
                            "fields": {"key": "uploads/${filename}", "x": value},
                        }
                    )
                }
            )

    def test_generates_distinct_keys_for_the_same_policy(self):
        _, first = up._prepare_upload(VALID_POLICY, "cat.png", "image/png")
        _, second = up._prepare_upload(VALID_POLICY, "cat.png", "image/png")

        assert first["key"] != second["key"]


class TestUploadBytes:
    """The upload is backgrounded, so every assertion drains first."""

    def test_returns_the_url_and_uploads_in_the_background(self, monkeypatch):
        posted = {}
        released = threading.Event()

        def fake_post(url, **kwargs):
            released.wait(timeout=5)
            posted["url"] = url
            posted["fields"] = kwargs["data"]
            return httpx.Response(204, request=httpx.Request("POST", url))

        monkeypatch.setattr(up, "_new_client", lambda: _StubClient(fake_post))
        url = upload_bytes_with_policy(VALID_POLICY, "cat.png", b"bytes", "image/png")

        # The URL comes back before the POST has even been allowed to start.
        assert url.startswith("https://bucket.s3.us-west-1.amazonaws.com/uploads/")
        assert "url" not in posted

        released.set()
        drain(timeout=5)
        assert posted["url"] == VALID_POLICY.url
        assert url.endswith(posted["fields"]["key"])

    def test_client_does_not_follow_redirects(self):
        """Following an S3 redirect would be an SSRF hole; off by default in
        httpx, but it must not silently become on."""
        with up._new_client() as client:
            assert client.follow_redirects is False

    def test_a_failed_upload_does_not_fail_the_request(self, monkeypatch):
        """Registry parity, and the accepted cost of backgrounding: the caller
        already has the URL, so the object simply never appears."""

        def fake_post(url, **kwargs):
            return httpx.Response(
                403, text="AccessDenied", request=httpx.Request("POST", url)
            )

        monkeypatch.setattr(up, "_new_client", lambda: _StubClient(fake_post))
        url = upload_bytes_with_policy(VALID_POLICY, "cat.png", b"bytes", "image/png")
        drain(timeout=5)

        assert url.startswith("https://bucket.s3.us-west-1.amazonaws.com/")

    @pytest.mark.parametrize("status", [403, 503])
    def test_a_background_failure_is_not_printed_to_stdout(
        self, monkeypatch, capsys, status
    ):
        """Runner stdout is attributed to whichever request is in flight, and
        that is frequently a different tenant. The structlog logger drops the
        event unless debug logging is on; print() would not."""

        def fake_post(url, **kwargs):
            return httpx.Response(
                status,
                text="<Error><Code>AccessDenied</Code><BucketName>tenant-a-secret"
                "</BucketName></Error>",
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(up, "_new_client", lambda: _StubClient(fake_post))
        upload_bytes_with_policy(VALID_POLICY, "cat.png", b"bytes", "image/png")
        drain(timeout=5)

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "tenant-a-secret" not in combined
        assert "AccessDenied" not in combined
        # fal.toolkit.utils.retry prints "Retrying N of M" and a traceback; a
        # retryable status must not route through it.
        assert "Retrying" not in combined
        assert "Traceback" not in combined

    @pytest.mark.parametrize("status", [301, 400, 403, 404])
    def test_terminal_statuses_are_not_retried(self, monkeypatch, status):
        calls = []

        def fake_post(url, **kwargs):
            calls.append(url)
            return httpx.Response(status, request=httpx.Request("POST", url))

        monkeypatch.setattr(up, "_new_client", lambda: _StubClient(fake_post))
        upload_bytes_with_policy(VALID_POLICY, "cat.png", b"bytes", "image/png")
        drain(timeout=5)

        assert len(calls) == 1

    @pytest.mark.parametrize("status", [408, 429, 500, 503])
    def test_transient_statuses_are_retried_to_exhaustion(self, monkeypatch, status):
        calls = []

        def fake_post(url, **kwargs):
            calls.append(url)
            return httpx.Response(status, request=httpx.Request("POST", url))

        monkeypatch.setattr(up, "_new_client", lambda: _StubClient(fake_post))
        upload_bytes_with_policy(VALID_POLICY, "cat.png", b"bytes", "image/png")
        drain(timeout=10)

        assert len(calls) == up._MAX_ATTEMPTS

    def test_recovers_when_a_retry_succeeds(self, monkeypatch):
        calls = []

        def fake_post(url, **kwargs):
            calls.append(url)
            status = 503 if len(calls) < 3 else 204
            return httpx.Response(status, request=httpx.Request("POST", url))

        monkeypatch.setattr(up, "_new_client", lambda: _StubClient(fake_post))
        upload_bytes_with_policy(VALID_POLICY, "cat.png", b"bytes", "image/png")
        drain(timeout=10)

        assert len(calls) == 3

    def test_303_is_success_not_failure(self, monkeypatch):
        """success_action_redirect makes S3 answer 303 after storing."""
        calls = []

        def fake_post(url, **kwargs):
            calls.append(url)
            return httpx.Response(
                303,
                headers={"Location": "https://example.com/done"},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(up, "_new_client", lambda: _StubClient(fake_post))
        upload_bytes_with_policy(VALID_POLICY, "cat.png", b"bytes", "image/png")
        drain(timeout=5)

        assert len(calls) == 1

    def test_oversized_payload_is_rejected_synchronously(self, monkeypatch):
        """Validation still raises inline -- only the POST is backgrounded."""
        posts = []
        monkeypatch.setattr(
            up,
            "_new_client",
            lambda: _StubClient(lambda *a, **k: posts.append(1)),
        )
        monkeypatch.setattr(up, "UPLOAD_POLICY_MAX_BYTES", 4)

        with pytest.raises(UploadPolicyInputError, match="exceeds"):
            upload_bytes_with_policy(
                VALID_POLICY, "cat.png", b"too many bytes", "image/png"
            )
        assert not posts

    def test_upload_threads_are_daemons(self, monkeypatch):
        """Non-daemon threads would let a stalled destination block interpreter
        exit with no timeout -- concurrent.futures' atexit hook joins them."""
        release = threading.Event()
        seen = {}

        def fake_post(url, **kwargs):
            seen["daemon"] = threading.current_thread().daemon
            release.wait(timeout=5)
            return httpx.Response(204, request=httpx.Request("POST", url))

        monkeypatch.setattr(up, "_new_client", lambda: _StubClient(fake_post))
        upload_bytes_with_policy(VALID_POLICY, "cat.png", b"bytes", "image/png")
        time.sleep(0.05)
        release.set()
        drain(timeout=5)

        assert seen["daemon"] is True


class TestBackgroundMachinery:
    def test_drain_without_a_timeout_waits_for_everything(self, monkeypatch):
        release = threading.Event()
        done = []

        def fake_post(url, **kwargs):
            release.wait(timeout=5)
            done.append(url)
            return httpx.Response(204, request=httpx.Request("POST", url))

        monkeypatch.setattr(up, "_new_client", lambda: _StubClient(fake_post))
        upload_bytes_with_policy(VALID_POLICY, "cat.png", b"bytes", "image/png")
        release.set()
        drain()

        assert len(done) == 1

    def test_drain_returns_when_the_timeout_expires(self, monkeypatch):
        release = threading.Event()

        def fake_post(url, **kwargs):
            release.wait(timeout=10)
            return httpx.Response(204, request=httpx.Request("POST", url))

        monkeypatch.setattr(up, "_new_client", lambda: _StubClient(fake_post))
        upload_bytes_with_policy(VALID_POLICY, "cat.png", b"bytes", "image/png")
        started = time.monotonic()
        drain(timeout=0.1)
        elapsed = time.monotonic() - started

        # Bounded: teardown must not be held open by a stalled destination.
        assert elapsed < 2
        release.set()
        drain(timeout=10)

    def test_drain_logs_uploads_it_abandons_at_shutdown(self, monkeypatch):
        """A runner can be killed before drain finishes; the abandoned uploads
        must leave a breadcrumb for when a customer reports a missing output."""
        release = threading.Event()
        events = MagicMock()
        monkeypatch.setattr(up, "logger", events)

        def fake_post(url, **kwargs):
            release.wait(timeout=10)
            return httpx.Response(204, request=httpx.Request("POST", url))

        monkeypatch.setattr(up, "_new_client", lambda: _StubClient(fake_post))
        upload_bytes_with_policy(VALID_POLICY, "cat.png", b"bytes", "image/png")
        drain(timeout=0.1)

        warned = [call for call in events.warning.call_args_list if call.args]
        assert any(
            "unfinished" in call.args[0] and call.kwargs.get("unfinished") == 1
            for call in warned
        )

        release.set()
        drain(timeout=10)

    def test_drain_does_not_warn_when_uploads_finish(self, monkeypatch):
        """The abandoned-upload warning must fire on a real timeout only, not on
        every clean shutdown."""
        events = MagicMock()
        monkeypatch.setattr(up, "logger", events)

        def fake_post(url, **kwargs):
            return httpx.Response(204, request=httpx.Request("POST", url))

        monkeypatch.setattr(up, "_new_client", lambda: _StubClient(fake_post))
        upload_bytes_with_policy(VALID_POLICY, "cat.png", b"bytes", "image/png")
        drain(timeout=5)

        assert not any(
            call.args and "unfinished" in call.args[0]
            for call in events.warning.call_args_list
        )

    def test_fork_reset_returns_the_queue_budget(self, monkeypatch):
        """A forked child inherits the spent budget but none of the threads
        that would return it, so uploads would refuse forever. The reset also
        drops the inherited client so the child builds its own."""
        monkeypatch.setattr(up, "_pending_bytes", up.UPLOAD_POLICY_MAX_PENDING_BYTES)

        up._reset_after_fork()

        assert up._pending_bytes == 0
        assert not up._inflight

    def test_disk_uploads_are_exempt_from_the_byte_budget(self, monkeypatch):
        """nbytes=0 uploads are budget-free even when the byte total is over
        cap; an in-memory upload in that state is still shed."""
        monkeypatch.setattr(
            up,
            "_new_client",
            lambda: _StubClient(
                lambda url, **kw: httpx.Response(
                    204, request=httpx.Request("POST", url)
                )
            ),
        )
        monkeypatch.setattr(
            up, "_pending_bytes", up.UPLOAD_POLICY_MAX_PENDING_BYTES + 1
        )

        up._submit(lambda client: client.post(VALID_POLICY.url), None, nbytes=0)
        with pytest.raises(UploadPolicyError, match="too much upload data"):
            up._submit(lambda client: client.post(VALID_POLICY.url), None, nbytes=1)
        up.drain(timeout=5)

    def test_cleanup_failure_still_releases_the_slot(self, monkeypatch):
        """A raised cleanup must not skip the slot release; otherwise a finished
        thread leaks its pending slot and enough failures wedge the queue."""
        monkeypatch.setattr(
            up,
            "_new_client",
            lambda: _StubClient(
                lambda url, **kw: httpx.Response(
                    204, request=httpx.Request("POST", url)
                )
            ),
        )

        called = []

        def boom():
            called.append(True)
            raise OSError("unlink failed")

        start_bytes = up._pending_bytes
        start_inflight = len(up._inflight)
        up._submit(
            lambda client: client.post(VALID_POLICY.url), None, nbytes=5, cleanup=boom
        )
        up.drain(timeout=5)

        assert called  # the raising-cleanup path was actually exercised
        assert up._pending_bytes == start_bytes
        assert len(up._inflight) == start_inflight

    def test_a_raising_cleanup_does_not_mask_the_refusal(self, monkeypatch):
        """On the synchronous refusal path a raised cleanup must not surface as
        an OSError/500 -- the caller must still see the 424 UploadPolicyError."""
        monkeypatch.setattr(
            up, "_pending_bytes", up.UPLOAD_POLICY_MAX_PENDING_BYTES + 1
        )

        called = []

        def boom():
            called.append(True)
            raise OSError("unlink failed")

        with pytest.raises(UploadPolicyError, match="too much upload data"):
            up._submit(
                lambda client: client.post(VALID_POLICY.url),
                None,
                nbytes=1,
                cleanup=boom,
            )
        assert called  # the cleanup on the refusal path actually ran

    def test_a_raising_cleanup_does_not_mask_a_thread_start_failure(self, monkeypatch):
        """Same contract on the thread-start-failure path: a raised cleanup must
        still surface the 424, and the reserved budget must be returned."""

        def no_start(self):
            raise RuntimeError("can't start new thread")

        monkeypatch.setattr(threading.Thread, "start", no_start)
        called = []

        def boom():
            called.append(True)
            raise OSError("unlink failed")

        start_bytes = up._pending_bytes
        start_inflight = len(up._inflight)
        with pytest.raises(UploadPolicyError, match="thread capacity"):
            up._submit(
                lambda client: client.post(VALID_POLICY.url),
                None,
                nbytes=5,
                cleanup=boom,
            )
        assert called
        assert up._pending_bytes == start_bytes  # budget returned, not leaked
        assert len(up._inflight) == start_inflight

    def test_transport_errors_are_wrapped(self):
        def fake_post(client):
            raise httpx.ConnectError("no route")

        with pytest.raises(UploadPolicyError) as excinfo:
            up._attempt_upload(fake_post)

        # Wrapped, but the transport error's text (which can carry the
        # caller-chosen host) stays out of the message and only the cause holds it.
        assert str(excinfo.value) == f"Upload via {UPLOAD_POLICY_KEY} failed."
        assert isinstance(excinfo.value.__cause__, httpx.ConnectError)

    def test_stops_retrying_once_the_total_deadline_passes(self, monkeypatch):
        calls = []

        def fake_post(client):
            calls.append(1)
            return httpx.Response(503, request=httpx.Request("POST", VALID_POLICY.url))

        monkeypatch.setattr(up, "UPLOAD_POLICY_TOTAL_DEADLINE", -1)
        with pytest.raises(UploadPolicyError):
            up._attempt_upload(fake_post)

        assert len(calls) == 1


class TestUploadPath:
    def test_posts_file_contents(self, tmp_path, monkeypatch):
        source = tmp_path / "cat.png"
        source.write_bytes(b"file bytes")
        seen = {}

        def fake_post(url, **kwargs):
            seen["body"] = kwargs["files"]["file"][1].read()
            return httpx.Response(204, request=httpx.Request("POST", url))

        monkeypatch.setattr(up, "_new_client", lambda: _StubClient(fake_post))
        upload_path_with_policy(VALID_POLICY, source, "cat.png", "image/png")
        drain(timeout=5)

        assert seen["body"] == b"file bytes"

    def test_survives_the_caller_deleting_the_file(self, tmp_path, monkeypatch):
        """The upload outlives the call, and apps routinely delete the temp
        file they just handed us."""
        source = tmp_path / "cat.png"
        source.write_bytes(b"file bytes")
        release = threading.Event()
        seen = {}

        def fake_post(url, **kwargs):
            release.wait(timeout=5)
            seen["body"] = kwargs["files"]["file"][1].read()
            return httpx.Response(204, request=httpx.Request("POST", url))

        monkeypatch.setattr(up, "_new_client", lambda: _StubClient(fake_post))
        upload_path_with_policy(VALID_POLICY, source, "cat.png", "image/png")
        source.unlink()  # gone before the upload has started
        release.set()
        drain(timeout=5)

        assert seen["body"] == b"file bytes"

    def test_survives_the_caller_overwriting_the_file_in_place(
        self, tmp_path, monkeypatch
    ):
        """A hardlink would share the source inode, so an in-place overwrite of
        the path would corrupt an in-flight upload; the copy must not."""
        source = tmp_path / "cat.png"
        source.write_bytes(b"first output")
        release = threading.Event()
        seen = {}

        def fake_post(url, **kwargs):
            release.wait(timeout=5)
            seen["body"] = kwargs["files"]["file"][1].read()
            return httpx.Response(204, request=httpx.Request("POST", url))

        monkeypatch.setattr(up, "_new_client", lambda: _StubClient(fake_post))
        upload_path_with_policy(VALID_POLICY, source, "cat.png", "image/png")
        source.write_bytes(b"second output, overwriting the same inode")
        release.set()
        drain(timeout=5)

        assert seen["body"] == b"first output"

    def test_staged_file_is_cleaned_up(self, tmp_path, monkeypatch):
        source = tmp_path / "cat.png"
        source.write_bytes(b"file bytes")
        staged = []

        def fake_post(url, **kwargs):
            staged.append(Path(kwargs["files"]["file"][1].name))
            return httpx.Response(204, request=httpx.Request("POST", url))

        monkeypatch.setattr(up, "_new_client", lambda: _StubClient(fake_post))
        upload_path_with_policy(VALID_POLICY, source, "cat.png", "image/png")
        drain(timeout=5)

        assert staged and not staged[0].exists()

    def test_reopens_the_file_for_each_attempt(self, tmp_path, monkeypatch):
        source = tmp_path / "cat.png"
        source.write_bytes(b"file bytes")
        bodies = []

        def fake_post(url, **kwargs):
            bodies.append(kwargs["files"]["file"][1].read())
            status = 503 if len(bodies) < 2 else 204
            return httpx.Response(status, request=httpx.Request("POST", url))

        monkeypatch.setattr(up, "_new_client", lambda: _StubClient(fake_post))
        upload_path_with_policy(VALID_POLICY, source, "cat.png", "image/png")
        drain(timeout=10)

        # A consumed handle would make the retry upload zero bytes.
        assert bodies == [b"file bytes", b"file bytes"]

    def test_oversized_file_is_rejected_synchronously(self, tmp_path, monkeypatch):
        source = tmp_path / "cat.png"
        source.write_bytes(b"too many bytes")
        posts = []
        monkeypatch.setattr(
            up,
            "_new_client",
            lambda: _StubClient(lambda *a, **k: posts.append(1)),
        )
        monkeypatch.setattr(up, "UPLOAD_POLICY_MAX_BYTES", 4)

        with pytest.raises(UploadPolicyInputError, match="exceeds"):
            upload_path_with_policy(VALID_POLICY, source, "cat.png", "image/png")
        assert not posts
