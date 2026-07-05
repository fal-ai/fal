"""Tests for bytes-safe ``RequestValidationError`` handling.

The default 422 handler in ``BaseServable._build_app`` used to serialize
``exc.errors()`` with ``jsonable_encoder``, which decodes ``bytes`` as strict
UTF-8. A binary request body (e.g. a PNG or a multipart upload sent to a JSON
endpoint) therefore crashed the exception handler itself with
``UnicodeDecodeError`` and surfaced as a 500 instead of a 422.
"""

import pydantic
import pytest
from pydantic import BaseModel

import fal

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + bytes(range(256))
PYDANTIC_V2 = pydantic.VERSION.startswith("2")


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


class ValidationEdgeCaseApp(ValidationDemoApp):
    def _add_extra_routes(self, app):
        from fastapi.exceptions import RequestValidationError

        def raise_validation_error():
            raise RequestValidationError(
                [
                    {"loc": ("body", "raw"), "input": bytearray(b"\xff\xd8")},
                    {"loc": ("body", "view"), "input": memoryview(b"\xff\xd8\xff")},
                    {"loc": ("body", "text"), "input": "hello \udc40 world"},
                    {
                        "loc": ("body", "number"),
                        "input": {"nan": float("nan"), "inf": float("inf"), "ok": 1.5},
                    },
                    {"loc": ("body", "ctx"), "ctx": {"error": ValueError("boom")}},
                    {"loc": ("body", "unicode"), "input": "Café 😄 東京"},
                    {"loc": ("body", "tuple"), "input": ("x", "y")},
                ]
            )

        app.add_api_route(
            "/validation-error",
            raise_validation_error,
            methods=["GET"],
        )


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
    assert resp.headers["x-fal-billable-units"] == "0"
    assert resp.json()["detail"]
    if PYDANTIC_V2:
        # Pydantic v2 echoes the raw body bytes in the error's ``input``
        # field, which is what used to crash the handler; the sanitizer
        # must render them as a placeholder. Pydantic v1 does not echo
        # the body, so its 422 never contained bytes to begin with.
        assert f"<binary {len(PNG_BYTES)} bytes>" in resp.text


def test_validation_error_payload_is_safe_for_json_response(isolate_agent_env):
    from fastapi.testclient import TestClient

    app = ValidationEdgeCaseApp()
    client = TestClient(app._build_app(), raise_server_exceptions=False)

    resp = client.get("/validation-error")

    assert resp.status_code == 422
    assert resp.headers["x-fal-billable-units"] == "0"
    detail = resp.json()["detail"]
    assert detail[0]["input"] == "<binary 2 bytes>"
    assert detail[1]["input"] == "<binary 3 bytes>"
    assert detail[2]["input"] == "hello ? world"
    assert detail[3]["input"] == {"nan": "nan", "inf": "inf", "ok": 1.5}
    assert detail[4]["ctx"]["error"] == "boom"
    assert detail[5]["input"] == "Café 😄 東京"
    assert detail[6]["input"] == ["x", "y"]


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
