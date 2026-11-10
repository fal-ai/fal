"""The raw-ASGI middlewares fal.App installs (hints, request context, log labels)."""

from __future__ import annotations

import asyncio
import contextvars
from types import SimpleNamespace

import pytest

from fal import app as fal_app_module
from fal.app import (
    CDN_TOKEN_KEY,
    REQUEST_ENDPOINT_KEY,
    REQUEST_ID_KEY,
    _LogContextMiddleware,
    _ProvideHintsMiddleware,
    _RequestContextMiddleware,
)
from fal.toolkit.file._upload_policy import UPLOAD_POLICY_KEY


def _scope(headers=None, type_="http"):
    return {
        "type": type_,
        "method": "POST",
        "path": "/",
        "query_string": b"",
        "headers": [(k.encode(), v.encode()) for k, v in (headers or {}).items()],
    }


async def _noop_receive():
    return {"type": "http.disconnect"}


def _run(middleware, scope, downstream):
    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    async def go():
        await middleware(scope, _noop_receive, send)

    asyncio.run(go())
    return sent


def _responder(chunks=(b"a", b"b"), headers=None, hook=None):
    async def app(scope, receive, send):
        if hook:
            hook()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": list(headers or []),
            }
        )
        for i, chunk in enumerate(chunks):
            await send(
                {
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": i < len(chunks) - 1,
                }
            )

    return app


def _header(sent, name):
    return dict(sent[0]["headers"]).get(name)


class TestHints:
    def test_header_added_and_body_untouched(self):
        fal_app = SimpleNamespace(provide_hints=lambda: ["a", "b"])
        mw = _ProvideHintsMiddleware(_responder([b"x", b"y", b"z"]), fal_app)
        sent = _run(mw, _scope(), None)
        assert _header(sent, b"x-fal-runner-hints") == b"a,b"
        assert [m.get("body") for m in sent[1:]] == [b"x", b"y", b"z"]

    def test_existing_header_is_replaced_not_duplicated(self):
        fal_app = SimpleNamespace(provide_hints=lambda: ["new"])
        inner = _responder(headers=[(b"X-Fal-Runner-Hints", b"old")])
        sent = _run(_ProvideHintsMiddleware(inner, fal_app), _scope(), None)
        names = [k.lower() for k, _ in sent[0]["headers"]]
        assert names.count(b"x-fal-runner-hints") == 1
        assert _header(sent, b"x-fal-runner-hints") == b"new"

    def test_empty_hints_still_set_an_empty_header(self):
        fal_app = SimpleNamespace(provide_hints=lambda: [])
        sent = _run(_ProvideHintsMiddleware(_responder(), fal_app), _scope(), None)
        assert _header(sent, b"x-fal-runner-hints") == b""

    def test_unimplemented_hints_set_no_header(self):
        def boom():
            raise NotImplementedError

        sent = _run(
            _ProvideHintsMiddleware(_responder(), SimpleNamespace(provide_hints=boom)),
            _scope(),
            None,
        )
        assert _header(sent, b"x-fal-runner-hints") is None

    def test_failing_hints_do_not_fail_the_request(self):
        def boom():
            raise RuntimeError("nope")

        sent = _run(
            _ProvideHintsMiddleware(_responder(), SimpleNamespace(provide_hints=boom)),
            _scope(),
            None,
        )
        assert sent[0]["status"] == 200
        assert _header(sent, b"x-fal-runner-hints") is None

    def test_unencodable_hint_is_dropped(self):
        fal_app = SimpleNamespace(provide_hints=lambda: ["ok", "☃"])
        sent = _run(_ProvideHintsMiddleware(_responder(), fal_app), _scope(), None)
        assert _header(sent, b"x-fal-runner-hints") == b"ok"

    def test_non_http_scope_passes_through(self):
        seen = []

        async def app(scope, receive, send):
            seen.append(scope["type"])

        fal_app = SimpleNamespace(provide_hints=lambda: ["a"])
        _run(_ProvideHintsMiddleware(app, fal_app), _scope(type_="websocket"), None)
        assert seen == ["websocket"]


class TestRequestContext:
    def _fal_app(self):
        return SimpleNamespace(_current_request_context=contextvars.ContextVar("ctx"))

    def test_context_visible_to_the_handler_and_reset_afterwards(self):
        fal_app = self._fal_app()
        seen = {}

        def hook():
            ctx = fal_app._current_request_context.get()
            seen["request_id"] = ctx.request_id
            seen["endpoint"] = ctx.endpoint

        mw = _RequestContextMiddleware(_responder(hook=hook), fal_app)
        scope = _scope({REQUEST_ID_KEY: "r1", REQUEST_ENDPOINT_KEY: "e1"})
        _run(mw, scope, None)
        assert seen == {"request_id": "r1", "endpoint": "e1"}
        assert fal_app._current_request_context.get(None) is None

    def test_context_reset_when_the_handler_raises(self):
        fal_app = self._fal_app()

        async def app(scope, receive, send):
            raise RuntimeError("boom")

        mw = _RequestContextMiddleware(app, fal_app)
        with pytest.raises(RuntimeError):
            _run(mw, _scope(), None)
        assert fal_app._current_request_context.get(None) is None

    def test_reissued_cdn_token_is_forwarded_to_the_caller(self):
        fal_app = self._fal_app()

        def hook():
            fal_app._current_request_context.get().headers[CDN_TOKEN_KEY] = "new"

        mw = _RequestContextMiddleware(_responder(hook=hook), fal_app)
        sent = _run(mw, _scope({CDN_TOKEN_KEY: "old"}), None)
        assert _header(sent, CDN_TOKEN_KEY.encode()) == b"new"

    def test_unchanged_cdn_token_is_not_added(self):
        fal_app = self._fal_app()
        mw = _RequestContextMiddleware(_responder(), fal_app)
        sent = _run(mw, _scope({CDN_TOKEN_KEY: "same"}), None)
        assert _header(sent, CDN_TOKEN_KEY.encode()) is None

    def test_streamed_body_chunks_pass_through_unchanged(self):
        fal_app = self._fal_app()
        chunks = [bytes([i]) * 7 for i in range(50)]
        mw = _RequestContextMiddleware(_responder(chunks), fal_app)
        sent = _run(mw, _scope(), None)
        assert [m["body"] for m in sent[1:]] == chunks
        assert [m["more_body"] for m in sent[1:]] == [True] * 49 + [False]

    def test_missing_context_var_falls_through(self):
        fal_app = SimpleNamespace(_current_request_context=None)
        sent = _run(_RequestContextMiddleware(_responder(), fal_app), _scope(), None)
        assert sent[0]["status"] == 200

    def test_bad_upload_policy_is_a_422_that_bills_zero(self):
        fal_app = self._fal_app()
        called = []

        async def app(scope, receive, send):
            called.append(1)

        mw = _RequestContextMiddleware(app, fal_app)
        sent = _run(mw, _scope({UPLOAD_POLICY_KEY: "{not json"}), None)
        assert not called
        assert sent[0]["status"] == 422
        assert _header(sent, b"x-fal-billable-units") == b"0"

    def test_non_http_scope_passes_through(self):
        seen = []

        async def app(scope, receive, send):
            seen.append(scope["type"])

        fal_app = self._fal_app()
        _run(_RequestContextMiddleware(app, fal_app), _scope(type_="websocket"), None)
        assert seen == ["websocket"]


class TestLogContext:
    @pytest.fixture
    def labels(self, monkeypatch):
        calls: list[dict] = []

        async def fake_set(labels, channel=None):
            calls.append(dict(labels))

        monkeypatch.setattr(fal_app_module, "_set_logger_labels", fake_set)
        monkeypatch.setenv("IS_ISOLATE_AGENT", "1")
        monkeypatch.setenv("NOMAD_ALLOC_PORT_grpc", "1234")
        return calls

    def _fal_app(self):
        return SimpleNamespace(isolate_channel=object())

    def test_labels_set_then_cleared_after_the_response(self, labels):
        order = []

        async def app(scope, receive, send):
            order.append(list(labels))
            await _responder()(scope, receive, send)

        mw = _LogContextMiddleware(app, self._fal_app())
        _run(mw, _scope({REQUEST_ID_KEY: "r1", REQUEST_ENDPOINT_KEY: "e1"}), None)
        assert order == [[{"fal_request_id": "r1", "fal_endpoint": "e1"}]]
        assert labels == [{"fal_request_id": "r1", "fal_endpoint": "e1"}, {}]

    def test_labels_cleared_when_the_handler_raises(self, labels):
        async def app(scope, receive, send):
            raise RuntimeError("boom")

        mw = _LogContextMiddleware(app, self._fal_app())
        with pytest.raises(RuntimeError):
            _run(mw, _scope({REQUEST_ID_KEY: "r1"}), None)
        assert labels == [{"fal_request_id": "r1"}, {}]

    def test_requests_without_ids_are_untouched(self, labels):
        mw = _LogContextMiddleware(_responder(), self._fal_app())
        sent = _run(mw, _scope(), None)
        assert sent[0]["status"] == 200
        assert labels == []

    def test_outside_the_isolate_agent_nothing_is_set(self, labels, monkeypatch):
        monkeypatch.delenv("IS_ISOLATE_AGENT")
        mw = _LogContextMiddleware(_responder(), self._fal_app())
        _run(mw, _scope({REQUEST_ID_KEY: "r1"}), None)
        assert labels == []
