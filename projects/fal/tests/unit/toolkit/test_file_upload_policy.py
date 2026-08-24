"""How File.from_bytes / File.from_path behave when a policy header is present.

test_upload_policy.py covers the module in isolation; this covers the wiring --
that the branch is taken at all, that the fal CDN is never used as a fallback,
and that the two constructors agree.
"""

from __future__ import annotations

import asyncio
import json
from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import cloudpickle
import httpx
import pytest
from fastapi import WebSocket, WebSocketDisconnect
from starlette.testclient import TestClient

import fal
import fal.ref
from fal.app import _RejectUploadPolicyOnWebSocket
from fal.ref import set_current_app
from fal.toolkit.file import _upload_policy as up
from fal.toolkit.file._upload_policy import (
    UPLOAD_POLICY_KEY,
    UploadPolicyInputError,
    upload_bytes_with_policy,
    upload_path_with_policy,
)
from fal.toolkit.file.file import DEFAULT_REPOSITORY, File
from fal.toolkit.file.providers.fal import FalFileRepositoryV3

POLICY = json.dumps(
    {
        "url": "https://bucket.s3.us-west-1.amazonaws.com/",
        "fields": {"key": "uploads/${filename}"},
    }
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


class FakeRequest:
    """Stands in for a fastapi Request; only .headers is read."""

    def __init__(self, headers: dict[str, str]):
        self.headers = headers


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(up, "_BASE_DELAY", 0)


@pytest.fixture
def accepting_s3(monkeypatch):
    """Record every POST and answer 204."""
    posts: list[dict[str, Any]] = []

    def fake_post(url, **kwargs):
        posts.append({"url": url, **kwargs})
        return httpx.Response(204, request=httpx.Request("POST", url))

    monkeypatch.setattr(up, "_new_client", lambda: _StubClient(fake_post))
    return posts


@pytest.fixture
def repository_spy(monkeypatch):
    """Fails loudly if anything reaches a fal storage repository."""
    calls: list[str] = []

    def fake_try_with_fallback(func, *args, **kwargs):
        calls.append(func)
        raise AssertionError(
            f"fal CDN repository was used ({func}); the policy path must not "
            "fall back to fal-owned storage"
        )

    monkeypatch.setattr(
        "fal.toolkit.file.file._try_with_fallback", fake_try_with_fallback
    )
    return calls


def test_from_bytes_uploads_to_the_policy_destination(accepting_s3, repository_spy):
    request = FakeRequest({UPLOAD_POLICY_KEY: POLICY})

    file = File.from_bytes(b"payload", content_type="text/plain", request=request)

    assert file.url.startswith("https://bucket.s3.us-west-1.amazonaws.com/uploads/")
    up.drain(timeout=5)
    assert len(accepting_s3) == 1
    assert not repository_spy


def test_from_path_uploads_to_the_policy_destination(
    tmp_path, accepting_s3, repository_spy
):
    source = tmp_path / "out.txt"
    source.write_bytes(b"payload")
    request = FakeRequest({UPLOAD_POLICY_KEY: POLICY})

    file = File.from_path(source, content_type="text/plain", request=request)

    assert file.url.startswith("https://bucket.s3.us-west-1.amazonaws.com/uploads/")
    assert file.url.endswith("-out.txt")
    up.drain(timeout=5)
    assert len(accepting_s3) == 1
    assert not repository_spy


def test_a_rejected_upload_never_falls_back_to_the_fal_cdn(monkeypatch, repository_spy):
    """The headline guarantee: a caller who asked for their own bucket must
    never silently get fal-owned storage instead.

    The upload is backgrounded, so the request still succeeds -- what must not
    happen is a retry against fal's own CDN.
    """
    attempted: list[str] = []

    def fake_post(url, **kwargs):
        attempted.append(url)
        return httpx.Response(
            403, text="AccessDenied", request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(up, "_new_client", lambda: _StubClient(fake_post))
    request = FakeRequest({UPLOAD_POLICY_KEY: POLICY})

    file = File.from_bytes(b"payload", content_type="text/plain", request=request)
    up.drain(timeout=5)

    assert attempted
    assert not repository_spy
    assert file.url.startswith("https://bucket.s3.us-west-1.amazonaws.com/")


def test_both_constructors_agree_on_file_data(tmp_path, accepting_s3, repository_spy):
    """as_bytes() must not start failing just because a header was sent."""
    source = tmp_path / "out.txt"
    source.write_bytes(b"payload")
    request = FakeRequest({UPLOAD_POLICY_KEY: POLICY})

    from_bytes = File.from_bytes(b"payload", content_type="text/plain", request=request)
    from_path = File.from_path(source, content_type="text/plain", request=request)

    assert from_bytes.as_bytes() == b"payload"
    assert from_path.as_bytes() == b"payload"


def test_file_size_and_name_are_populated(tmp_path, accepting_s3, repository_spy):
    source = tmp_path / "out.txt"
    source.write_bytes(b"payload")
    request = FakeRequest({UPLOAD_POLICY_KEY: POLICY})

    file = File.from_path(source, content_type="text/plain", request=request)

    assert file.file_size == len(b"payload")
    assert file.file_name == "out.txt"
    assert file.content_type == "text/plain"


def test_absent_header_still_uses_the_repository():
    """The default path must be untouched."""
    file = File.from_bytes(
        b"payload",
        content_type="text/plain",
        repository="in_memory",
        request=FakeRequest({}),
    )

    assert file.url.startswith("data:text/plain;base64,")


def test_malformed_header_raises_before_any_upload(accepting_s3, repository_spy):
    request = FakeRequest({UPLOAD_POLICY_KEY: "{not json"})

    with pytest.raises(UploadPolicyInputError):
        File.from_bytes(b"payload", content_type="text/plain", request=request)

    up.drain(timeout=5)
    assert not accepting_s3
    assert not repository_spy


class _ContextVarApp:
    """Holds current_request in a real ContextVar, like fal.App does.

    A plain attribute would make the async tests below pass without exercising
    any context propagation at all -- which is the thing they exist to prove.
    """

    def __init__(self, context):
        self._var: ContextVar = ContextVar("_test_request_context", default=None)
        self._var.set(context)

    @property
    def current_request(self):
        return self._var.get()


@pytest.fixture
def current_app_with_policy(monkeypatch):
    """Resolve the policy the way production does: off the ContextVar.

    Real apps call File.from_bytes(data) with no request= argument, so the
    header is found via get_current_app().current_request. Passing request=
    explicitly, as the tests above do, bypasses that branch entirely.
    """
    context = SimpleNamespace(
        headers={UPLOAD_POLICY_KEY: POLICY}, lifecycle_preference=None
    )
    app = _ContextVarApp(context)
    monkeypatch.setattr("fal.toolkit.file._upload_policy.get_current_app", lambda: app)
    monkeypatch.setattr("fal.toolkit.file.file.get_current_app", lambda: app)
    return app


def test_resolves_the_policy_from_the_current_request(
    current_app_with_policy, accepting_s3, repository_spy
):
    """The production path: no request= argument."""
    file = File.from_bytes(b"payload", content_type="text/plain")

    assert file.url.startswith("https://bucket.s3.us-west-1.amazonaws.com/uploads/")
    up.drain(timeout=5)
    assert len(accepting_s3) == 1
    assert not repository_spy


def test_no_current_request_falls_through_to_the_repository(monkeypatch):
    """Outside a request there is no policy, so the CDN path must still work."""
    monkeypatch.setattr("fal.toolkit.file._upload_policy.get_current_app", lambda: None)
    monkeypatch.setattr("fal.toolkit.file.file.get_current_app", lambda: None)

    file = File.from_bytes(
        b"payload", content_type="text/plain", repository="in_memory"
    )

    assert file.url.startswith("data:text/plain;base64,")


@pytest.mark.asyncio
async def test_async_constructors_resolve_the_policy_through_the_contextvar(
    tmp_path, current_app_with_policy, accepting_s3, repository_spy
):
    """run_in_thread propagates the ContextVar, so the async variants work.

    Deliberately no request= argument -- passing one would make this pass
    without ever exercising the propagation it claims to test.
    """
    source = tmp_path / "out.txt"
    source.write_bytes(b"payload")

    from_bytes = await File.from_bytes_async(b"payload", content_type="text/plain")
    from_path = await File.from_path_async(source, content_type="text/plain")

    assert from_bytes.url.startswith("https://bucket.s3.us-west-1.amazonaws.com/")
    assert from_path.url.startswith("https://bucket.s3.us-west-1.amazonaws.com/")


def test_a_mock_request_object_is_not_treated_as_a_policy(accepting_s3):
    """App test suites routinely pass a bare MagicMock as request=."""
    file = File.from_bytes(
        b"payload",
        content_type="text/plain",
        repository="in_memory",
        request=MagicMock(),
    )

    assert file.url.startswith("data:text/plain;base64,")
    up.drain(timeout=5)
    assert not accepting_s3


class TestMiddleware:
    """The header is parsed once at request entry, not once per output file."""

    @pytest.fixture(autouse=True)
    def _release_current_app(self):
        """set_current_app refuses to overwrite, so leaving it set here fails
        whatever runs next in the same process."""
        yield
        fal.ref.current_app = None

    def _app(self):
        class _App(fal.App):
            @fal.endpoint("/")
            def gen(self) -> dict:
                # Three outputs from one request: the header must be validated
                # once, before this runs, not three times inside it.
                for i in range(3):
                    File.from_bytes(b"x", content_type="text/plain")
                return {"ok": True}

        return _App

    def _client(self, app_cls=None):
        # A context manager, so the lifespan runs and the request-context
        # middleware is armed (it no-ops otherwise).
        fal.ref.current_app = None
        app = (app_cls or self._app())(_allow_init=True)
        set_current_app(app)
        return TestClient(app._build_app(), raise_server_exceptions=False)

    def test_malformed_policy_is_422_before_the_endpoint_runs(self, monkeypatch):
        posts = []
        monkeypatch.setattr(
            up,
            "_new_client",
            lambda: _StubClient(lambda *a, **k: posts.append(1)),
        )

        with self._client() as client:
            resp = client.post("/", json={}, headers={UPLOAD_POLICY_KEY: "{not json"})

        assert resp.status_code == 422
        # A plain message, no invented body location, and nothing was uploaded.
        assert "not valid JSON" in resp.json()["detail"]
        assert not posts
        # Pre-generation: must bill zero, or the platform charges the default.
        assert resp.headers["x-fal-billable-units"] == "0"

    def test_upload_time_failure_is_a_422_that_still_bills(self, monkeypatch):
        """_prepare_upload's checks fire inside from_bytes, after the middleware
        has already accepted the header, so they surface through AppException's
        handler rather than the middleware.

        Unlike a header that was malformed from the start, this is only knowable
        once the output exists, so the generation really happened and billing is
        left at the endpoint's default.
        """
        posts = []
        monkeypatch.setattr(
            up,
            "_new_client",
            lambda: _StubClient(lambda *a, **k: posts.append(1)),
        )
        monkeypatch.setattr(up, "UPLOAD_POLICY_MAX_BYTES", 0)

        with self._client() as client:
            resp = client.post("/", json={}, headers={UPLOAD_POLICY_KEY: POLICY})

        assert resp.status_code == 422
        assert "exceeds" in resp.json()["detail"]
        assert not posts
        # Not zeroed: the GPU work was done before this was knowable.
        assert "x-fal-billable-units" not in resp.headers

    def test_valid_policy_is_parsed_once_and_serves_every_output(self, monkeypatch):
        posts = []

        def fake_post(url, **kwargs):
            posts.append(url)
            return httpx.Response(204, request=httpx.Request("POST", url))

        monkeypatch.setattr(up, "_new_client", lambda: _StubClient(fake_post))
        parses = []
        real_parse = up.parse_upload_policy
        # Patch the name the middleware imported into app.py.
        monkeypatch.setattr(
            "fal.app.parse_upload_policy",
            lambda h: parses.append(1) or real_parse(h),
        )

        with self._client() as client:
            resp = client.post("/", json={}, headers={UPLOAD_POLICY_KEY: POLICY})
        up.drain(timeout=5)

        assert resp.status_code == 200
        assert len(posts) == 3  # three outputs uploaded
        assert len(parses) == 1  # header parsed once, not per output

    def _ws_app(self):
        class _WsApp(fal.App):
            @fal.endpoint("/")
            def gen(self) -> dict:
                return {"ok": True}

            @fal.endpoint("/ws", is_websocket=True)
            async def gen_ws(self, websocket: WebSocket) -> None:
                await websocket.accept()
                await websocket.send_json({"ok": True})
                await websocket.close()

        return _WsApp

    def test_a_websocket_carrying_a_policy_is_refused(self):
        """A WebSocket never reaches the BaseHTTPMiddleware that parses the
        policy, so honoring the connection would write the caller's output to
        fal storage while they believe it went to their bucket.

        The close code is deliberately not asserted here: it is a TestClient
        artifact. uvicorn renders a pre-accept close as a bare 403, and fal's
        gateway reports 1011, so no caller ever sees 1008. What matters is that
        the endpoint never runs. The exact ASGI message is pinned by the sibling
        test against the guard in isolation.
        """
        with self._client(self._ws_app()) as client:
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(
                    "/ws", headers={UPLOAD_POLICY_KEY: POLICY}
                ):
                    pass

    @pytest.mark.parametrize(
        "header_name",
        [UPLOAD_POLICY_KEY, UPLOAD_POLICY_KEY.title()],
        ids=["lowercase", "titlecase"],
    )
    def test_the_refusal_ignores_header_case(self, header_name):
        """Only uvicorn's wsproto impl lowercases scope header names, and the
        default ws="auto" resolves to the sansio one, which does not. A titlecased
        header slipping through would silently write to fal storage."""
        sent: list[dict] = []
        guard = _RejectUploadPolicyOnWebSocket(app=None)
        scope = {
            "type": "websocket",
            "path": "/ws",
            "headers": [(header_name.encode(), POLICY.encode())],
            "extensions": {},
        }

        async def send(message):
            sent.append(message)

        asyncio.run(guard(scope, None, send))

        assert sent == [{"type": "websocket.close", "code": 1008}]

    def test_a_websocket_without_the_header_still_connects(self):
        """The guard must not break realtime endpoints generally."""
        with self._client(self._ws_app()) as client:
            with client.websocket_connect("/ws") as ws:
                assert ws.receive_json() == {"ok": True}


def test_an_explicit_repository_is_not_silently_overridden(
    accepting_s3, repository_spy
):
    """An app that names its own destination, often a fal-owned bucket its
    catalogue reads back from (registry passes GoogleStorageRepository for
    trained LoRA weights), must not have it moved by a caller header. The id
    is enough: the conflict check fires before get_builtin_repository would
    construct anything, so no service-account secret is needed."""
    request = FakeRequest({UPLOAD_POLICY_KEY: POLICY})

    with pytest.raises(UploadPolicyInputError, match="cannot override"):
        File.from_bytes(
            b"payload",
            content_type="text/plain",
            repository="gcp_storage",
            request=request,
        )

    assert not accepting_s3
    assert not repository_spy


def test_an_explicit_repository_is_not_silently_overridden_from_path(
    tmp_path, accepting_s3, repository_spy
):
    source = tmp_path / "out.txt"
    source.write_bytes(b"payload")
    request = FakeRequest({UPLOAD_POLICY_KEY: POLICY})

    with pytest.raises(UploadPolicyInputError, match="cannot override"):
        File.from_path(source, repository="gcp_storage", request=request)

    assert not accepting_s3
    assert not repository_spy


@dataclass
class _RepositoryWithSecret:
    """Shaped like the real ones: GoogleStorageRepository and R2Repository are
    dataclasses whose fields hold credentials."""

    gcp_account_json: str = "-----BEGIN PRIVATE KEY-----abc123"


def test_the_conflict_error_names_the_type_not_its_credentials(accepting_s3):
    """UploadPolicyInputError is an AppException, so api.py returns its message
    to the caller verbatim. repr() of a repository would hand a caller the app's
    service-account key."""
    request = FakeRequest({UPLOAD_POLICY_KEY: POLICY})

    with pytest.raises(UploadPolicyInputError) as excinfo:
        File.from_bytes(
            b"payload",
            content_type="text/plain",
            repository=_RepositoryWithSecret(),
            request=request,
        )

    assert "_RepositoryWithSecret" in excinfo.value.message
    assert "PRIVATE KEY" not in excinfo.value.message
    assert "abc123" not in excinfo.value.message


@pytest.mark.parametrize(
    "repository",
    ["cdn", "fal_v2", pytest.param(FalFileRepositoryV3(), id="instance")],
)
def test_a_deprecated_alias_for_the_default_still_honors_the_policy(
    repository, accepting_s3, repository_spy
):
    """get_builtin_repository folds both onto fal_v3, so they name the default
    destination and are not a conflict."""
    request = FakeRequest({UPLOAD_POLICY_KEY: POLICY})

    file = File.from_bytes(
        b"payload", content_type="text/plain", repository=repository, request=request
    )
    up.drain(timeout=5)

    assert file.url.startswith("https://bucket.s3.us-west-1.amazonaws.com/")
    assert not repository_spy


def test_passing_the_default_repository_explicitly_still_honors_the_policy(
    accepting_s3, repository_spy
):
    """Only a *different* destination conflicts; the default is what the policy
    is meant to replace."""
    request = FakeRequest({UPLOAD_POLICY_KEY: POLICY})

    file = File.from_bytes(
        b"payload",
        content_type="text/plain",
        repository=DEFAULT_REPOSITORY,
        request=request,
    )
    up.drain(timeout=5)

    assert file.url.startswith("https://bucket.s3.us-west-1.amazonaws.com/")
    assert len(accepting_s3) == 1
    assert not repository_spy


def test_an_upload_failure_is_reported_where_the_runner_captures_it(monkeypatch, capfd):
    """The only evidence a caller's bucket is empty. structlog drops it unless
    debug logging is on, and nothing turns that on, so it has to reach stdout."""

    def failing_post(url, **kwargs):
        request = httpx.Request("POST", url)
        raise httpx.HTTPStatusError(
            "403",
            request=request,
            response=httpx.Response(
                403,
                request=request,
                text="<Error><Code>AccessDenied</Code></Error>",
            ),
        )

    monkeypatch.setattr(up, "_new_client", lambda: _StubClient(failing_post))
    request = FakeRequest({UPLOAD_POLICY_KEY: POLICY})

    File.from_bytes(b"payload", content_type="text/plain", request=request)
    up.drain(timeout=5)

    reports = [
        json.loads(line)
        for line in capfd.readouterr().out.strip().splitlines()
        if "upload_policy_failure" in line
    ]
    assert len(reports) == 1
    failure = reports[0]["upload_policy_failure"]
    assert failure["status"] == 403
    assert failure["s3_error"] == "AccessDenied"


def test_large_from_path_output_keeps_file_data_none(
    tmp_path, accepting_s3, monkeypatch
):
    """The chunked branch (file over the multipart threshold) returns
    file_data=None, mirroring the repository path; as_bytes() then raises."""
    monkeypatch.setattr(
        "fal.toolkit.file.file.MultipartUploadV3.MULTIPART_THRESHOLD", 4
    )
    source = tmp_path / "big.bin"
    source.write_bytes(b"payload-over-threshold")
    request = FakeRequest({UPLOAD_POLICY_KEY: POLICY})

    file = File.from_path(source, content_type="text/plain", request=request)
    up.drain(timeout=5)

    assert file.file_data is None
    with pytest.raises(ValueError, match="not been downloaded"):
        file.as_bytes()


def test_httpx_stays_declared_for_served_apps():
    """_upload_policy imports httpx inside its functions, so the dependency is
    invisible to whoever maintains SERVE_REQUIREMENTS. Assert the declaration,
    and the floor rather than mere presence: _new_client passes
    Client(follow_redirects=), which needs httpx 0.20, and lowering the floor
    would raise TypeError only on a real upload, never on a test machine with a
    modern httpx installed.
    """
    from packaging.requirements import Requirement  # noqa: PLC0415

    from fal.api.api import SERVE_REQUIREMENTS  # noqa: PLC0415

    declared = [
        Requirement(raw)
        for raw in SERVE_REQUIREMENTS
        if Requirement(raw).name == "httpx"
    ]

    assert declared, SERVE_REQUIREMENTS
    assert all(
        not requirement.specifier.contains("0.15") for requirement in declared
    ), declared


def test_file_can_still_be_cloudpickled_by_value():
    """upload_policy is on the ban-lazy-imports-serialized list, so importing it
    must not make File (or the exported helpers) carry an unpicklable object."""
    for obj in (File, upload_bytes_with_policy, upload_path_with_policy):
        cloudpickle.dumps(obj)
