import httpx
import pytest

import fal_client
from fal_client import _validation
from fal_client._validation import (
    fetch_required_arguments,
    missing_required,
    required_arguments,
)

APPLICATION = "fal-ai/flux/schnell"

INPUT_MODEL = {
    "type": "object",
    "required": ["prompt"],
    "properties": {
        "prompt": {"type": "string"},
        "num_images": {"type": "integer"},
    },
}


def spec(model=INPUT_MODEL, schema=None):
    """Shaped like the published queue document: paths keyed by endpoint id."""
    if schema is None:
        schema = {"$ref": "#/components/schemas/FluxSchnellInput"}
    return {
        "openapi": "3.0.4",
        "paths": {
            f"/{APPLICATION}": {
                "post": {
                    "requestBody": {"content": {"application/json": {"schema": schema}}}
                }
            },
            f"/{APPLICATION}/requests/{{request_id}}/status": {"get": {}},
        },
        "components": {"schemas": {"FluxSchnellInput": model}},
    }


@pytest.fixture(autouse=True)
def _clear_cache():
    _validation.clear_schema_cache()
    yield
    _validation.clear_schema_cache()


class TestRequiredArguments:
    def test_follows_a_local_ref(self):
        assert required_arguments(spec(), APPLICATION) == ["prompt"]

    def test_tolerates_surrounding_slashes(self):
        assert required_arguments(spec(), f"/{APPLICATION}/") == ["prompt"]

    def test_reads_an_inline_schema(self):
        document = spec(schema={"type": "object", "required": ["image_url"]})
        assert required_arguments(document, APPLICATION) == ["image_url"]

    def test_unknown_endpoint_is_none(self):
        assert required_arguments(spec(), "fal-ai/nope") is None

    def test_all_optional_model_is_empty_not_none(self):
        """pydantic omits ``required`` when every field has a default.

        That is an all-optional model, not a missing schema, and conflating the
        two would report a problem for arguments that are in fact complete.
        """
        assert required_arguments(spec(model={"type": "object"}), APPLICATION) == []

    def test_malformed_required_is_none(self):
        document = spec(model={"type": "object", "required": "prompt"})
        assert required_arguments(document, APPLICATION) is None

    def test_refuses_a_ref_it_cannot_resolve_locally(self):
        document = spec(schema={"$ref": "https://example.test/s.json#/Input"})
        assert required_arguments(document, APPLICATION) is None

    def test_non_json_request_body_is_none(self):
        document = spec()
        document["paths"][f"/{APPLICATION}"]["post"]["requestBody"]["content"] = {
            "multipart/form-data": {"schema": {"required": ["file"]}}
        }
        assert required_arguments(document, APPLICATION) is None

    def test_route_without_a_post_is_none(self):
        assert (
            required_arguments(spec(), f"{APPLICATION}/requests/{{request_id}}/status")
            is None
        )


class TestMissingRequired:
    def test_reports_an_absent_key(self):
        assert missing_required({"num_images": 2}, ["prompt"]) == ["prompt"]

    def test_none_is_not_absence(self):
        """A required field typed ``Optional[T]`` appears in ``required`` and
        accepts ``None``, so a null value must not be reported as missing."""
        assert missing_required({"prompt": None}, ["prompt"]) == []

    def test_preserves_declaration_order(self):
        assert missing_required({"b": 1}, ["a", "b", "c"]) == ["a", "c"]


class TestFetch:
    def _stub(self, monkeypatch, handler):
        def get(url, **kwargs):
            return handler(url, kwargs)

        monkeypatch.setattr(_validation.httpx, "get", get)

    def test_returns_required_names(self, monkeypatch):
        self._stub(
            monkeypatch,
            lambda url, kw: httpx.Response(200, json=spec()),
        )
        assert fetch_required_arguments(APPLICATION) == ["prompt"]

    def test_caches_by_endpoint(self, monkeypatch):
        calls = []

        def handler(url, kw):
            calls.append(kw["params"]["endpoint_id"])
            return httpx.Response(200, json=spec())

        self._stub(monkeypatch, handler)
        fetch_required_arguments(APPLICATION)
        fetch_required_arguments(APPLICATION)
        assert calls == [APPLICATION]

    @pytest.mark.parametrize(
        "handler",
        [
            lambda url, kw: httpx.Response(404),
            lambda url, kw: httpx.Response(500),
            lambda url, kw: httpx.Response(200, text="not json"),
        ],
        ids=["not-found", "server-error", "not-json"],
    )
    def test_fails_open_on_a_bad_response(self, monkeypatch, handler):
        self._stub(monkeypatch, handler)
        assert fetch_required_arguments(APPLICATION) is None

    def test_fails_open_when_the_request_raises(self, monkeypatch):
        def handler(url, kw):
            raise httpx.ConnectError("no route to host")

        self._stub(monkeypatch, handler)
        assert fetch_required_arguments(APPLICATION) is None


class TestClientSurface:
    def test_check_arguments_reports_what_is_missing(self, monkeypatch):
        monkeypatch.setattr(
            _validation, "fetch_required_arguments", lambda app: ["prompt"]
        )
        client = fal_client.SyncClient(key="test")
        assert client.check_arguments(APPLICATION, {}) == ["prompt"]
        assert client.check_arguments(APPLICATION, {"prompt": "a cat"}) == []

    def test_check_arguments_is_silent_when_the_schema_is_unknown(self, monkeypatch):
        """An unreadable schema must not look like a problem with arguments."""
        monkeypatch.setattr(_validation, "fetch_required_arguments", lambda app: None)
        client = fal_client.SyncClient(key="test")
        assert client.check_arguments(APPLICATION, {}) == []

    def test_validate_arguments_raises_with_the_names(self, monkeypatch):
        monkeypatch.setattr(
            _validation, "fetch_required_arguments", lambda app: ["prompt", "image_url"]
        )
        client = fal_client.SyncClient(key="test")
        with pytest.raises(fal_client.MissingRequiredArguments) as excinfo:
            client.validate_arguments(APPLICATION, {})
        assert excinfo.value.missing == ["prompt", "image_url"]
        assert "prompt, image_url" in str(excinfo.value)

    def test_validate_arguments_passes_a_complete_set(self, monkeypatch):
        monkeypatch.setattr(
            _validation, "fetch_required_arguments", lambda app: ["prompt"]
        )
        client = fal_client.SyncClient(key="test")
        client.validate_arguments(APPLICATION, {"prompt": "a cat"})

    def test_error_is_a_fal_client_error(self):
        """Callers already catching FalClientError must keep working."""
        assert issubclass(
            fal_client.MissingRequiredArguments, fal_client.FalClientError
        )

    @pytest.mark.asyncio
    async def test_async_validate_arguments_raises(self, monkeypatch):
        async def fetch(app):
            return ["prompt"]

        monkeypatch.setattr(_validation, "fetch_required_arguments_async", fetch)
        client = fal_client.AsyncClient(key="test")
        with pytest.raises(fal_client.MissingRequiredArguments):
            await client.validate_arguments(APPLICATION, {})

    @pytest.mark.asyncio
    async def test_async_check_arguments_is_silent_when_unknown(self, monkeypatch):
        async def fetch(app):
            return None

        monkeypatch.setattr(_validation, "fetch_required_arguments_async", fetch)
        client = fal_client.AsyncClient(key="test")
        assert await client.check_arguments(APPLICATION, {}) == []
