"""Tests for bytes-safe ``RequestValidationError`` handling.

The default 422 handler in ``BaseServable._build_app`` used to serialize
``exc.errors()`` with ``jsonable_encoder``, which decodes ``bytes`` as strict
UTF-8. A binary request body (e.g. a PNG or a multipart upload sent to a JSON
endpoint) therefore crashed the exception handler itself with
``UnicodeDecodeError`` and surfaced as a 500 instead of a 422.
"""

import json

import pytest
from pydantic import BaseModel

import fal
from fal.api.api import _sanitize_validation_errors

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + bytes(range(256))


@pytest.fixture
def isolate_agent_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IS_ISOLATE_AGENT", "1")


class DemoInput(BaseModel):
    prompt: str


class DemoOutput(BaseModel):
    echo: str


class ValidationDemoApp(fal.App):
    @fal.endpoint("/")
    def run(self, input: DemoInput) -> DemoOutput:
        return DemoOutput(echo=input.prompt)


class TestSanitizer:
    def test_bytes_become_placeholder(self):
        errors = [
            {
                "type": "model_attributes_type",
                "loc": ("body",),
                "msg": "Input should be a valid dictionary",
                "input": PNG_BYTES,
            }
        ]
        sanitized = _sanitize_validation_errors(errors)
        assert sanitized[0]["input"] == f"<binary {len(PNG_BYTES)} bytes>"
        assert sanitized[0]["loc"] == ["body"]
        json.dumps(sanitized)

    def test_bytearray_and_memoryview(self):
        assert _sanitize_validation_errors(bytearray(b"\xff\xd8")) == "<binary 2 bytes>"
        assert (
            _sanitize_validation_errors(memoryview(b"\xff\xd8\xff"))
            == "<binary 3 bytes>"
        )

    def test_lone_surrogates_made_utf8_encodable(self):
        sanitized = _sanitize_validation_errors([{"input": "hello \udc40 world"}])
        # JSONResponse.render dumps with ensure_ascii=False and UTF-8
        # encodes; a lone surrogate would raise UnicodeEncodeError there.
        json.dumps({"detail": sanitized}, ensure_ascii=False).encode("utf-8")
        assert sanitized[0]["input"] == "hello ? world"

    def test_valid_non_ascii_preserved(self):
        assert _sanitize_validation_errors("Café 😄 東京") == "Café 😄 東京"

    def test_non_finite_floats_stringified(self):
        sanitized = _sanitize_validation_errors(
            {"nan": float("nan"), "inf": float("inf"), "ok": 1.5}
        )
        assert sanitized["nan"] == "nan"
        assert sanitized["inf"] == "inf"
        assert sanitized["ok"] == 1.5
        json.dumps(sanitized, allow_nan=False)

    def test_ctx_exception_objects_stringified(self):
        sanitized = _sanitize_validation_errors(
            [{"ctx": {"error": ValueError("boom")}}]
        )
        assert sanitized[0]["ctx"]["error"] == "boom"
        json.dumps(sanitized)

    def test_json_primitives_and_containers_preserved(self):
        data = {
            "a": 1,
            "b": 2.5,
            "c": None,
            "d": True,
            "e": [False, 0],
            "f": ("x", "y"),
        }
        assert _sanitize_validation_errors(data) == {**data, "f": ["x", "y"]}


def test_binary_body_returns_422_not_500(isolate_agent_env):
    from fastapi.testclient import TestClient

    app = ValidationDemoApp()
    client = TestClient(app._build_app(), raise_server_exceptions=False)

    resp = client.post(
        "/",
        content=PNG_BYTES,
        headers={"Content-Type": "multipart/form-data; boundary=xyz"},
    )

    assert resp.status_code == 422, resp.text
    assert f"<binary {len(PNG_BYTES)} bytes>" in resp.text
    assert resp.headers["x-fal-billable-units"] == "0"


def test_valid_json_still_works(isolate_agent_env):
    from fastapi.testclient import TestClient

    app = ValidationDemoApp()
    client = TestClient(app._build_app(), raise_server_exceptions=False)

    resp = client.post("/", json={"prompt": "hello"})

    assert resp.status_code == 200
    assert resp.json() == {"echo": "hello"}


def test_utf8_text_validation_error_still_422(isolate_agent_env):
    from fastapi.testclient import TestClient

    app = ValidationDemoApp()
    client = TestClient(app._build_app(), raise_server_exceptions=False)

    resp = client.post(
        "/",
        content=b"not json",
        headers={"Content-Type": "text/plain"},
    )

    assert resp.status_code == 422
    assert resp.headers["x-fal-billable-units"] == "0"
