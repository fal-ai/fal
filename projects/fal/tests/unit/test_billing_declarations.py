from __future__ import annotations

import dataclasses
from decimal import Decimal
from typing import Any

import pytest
from pydantic import BaseModel

import fal
from fal.wma import App as WmaApp


@pytest.fixture(autouse=True)
def isolate_agent_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IS_ISOLATE_AGENT", "1")


APP = {"type": "object", "properties": {}, "required": []}


class Declaration:
    def __init__(self, metric: str) -> None:
        self.metric = metric

    def declaration(self) -> fal.BillingComponents:
        return {"version": 1, "schema_id": f"id-{self.metric}", "app": APP}


class EditInput(BaseModel):
    prompt: str


class EditOutput(BaseModel):
    images: int


class BilledApp(fal.App):
    @fal.endpoint("/edit", billing=Declaration("images"))
    def edit(self, input: EditInput) -> EditOutput:
        return EditOutput(images=1)

    @fal.endpoint("/upscale", billing=Declaration("megapixels"))
    def upscale(self, input: EditInput) -> EditOutput:
        return EditOutput(images=1)

    @fal.endpoint("/plain")
    def plain(self, input: EditInput) -> EditOutput:
        return EditOutput(images=1)


class PlainApp(fal.App):
    @fal.endpoint("/edit")
    def edit(self, input: EditInput) -> EditOutput:
        return EditOutput(images=1)


def test_billing_components_are_published_outside_openapi() -> None:
    metadata = BilledApp.build_metadata()
    assert metadata["billing_components"] == {
        "/edit": {"version": 1, "schema_id": "id-images", "app": APP},
        "/upscale": {"version": 1, "schema_id": "id-megapixels", "app": APP},
    }
    assert "billing" not in str(metadata["openapi"]).lower()


def test_billing_does_not_change_the_openapi_spec() -> None:
    billed = BilledApp().openapi()["paths"]["/edit"]
    plain = PlainApp().openapi()["paths"]["/edit"]
    assert billed["post"]["requestBody"] == plain["post"]["requestBody"]
    assert billed["post"]["responses"] == plain["post"]["responses"]


def test_apps_without_billing_publish_only_openapi() -> None:
    assert set(PlainApp.build_metadata()) == {"openapi"}


def test_billing_must_be_a_declaration() -> None:
    with pytest.raises(TypeError, match="declaration"):
        fal.endpoint("/edit", billing=object())  # type: ignore[arg-type]
    # The class itself, not an instance: declaration() would need self.
    with pytest.raises(TypeError, match="declaration"):
        fal.endpoint("/edit", billing=Declaration)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Websocket"):
        fal.endpoint("/ws", is_websocket=True, billing=Declaration("images"))


@dataclasses.dataclass
class EqualityDeclaration:
    """A plain dataclass: equality without hashing."""

    metric: str

    def declaration(self) -> fal.BillingComponents:
        return {"version": 1, "schema_id": f"id-{self.metric}", "app": APP}


def test_a_declaration_need_not_be_hashable() -> None:
    class DataclassApp(fal.App):
        @fal.endpoint("/edit", billing=EqualityDeclaration("images"))
        def edit(self, input: EditInput) -> EditOutput:
            return EditOutput(images=1)

    assert set(DataclassApp.build_metadata()["billing_components"]) == {"/edit"}
    assert DataclassApp().collect_routes()


def test_declaration_must_be_a_method() -> None:
    class Property:
        declaration = {"version": 1, "schema_id": "x", "app": APP}

    with pytest.raises(TypeError, match="declaration"):
        fal.endpoint("/edit", billing=Property())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "declaration, reason",
    [
        ([], "must be a dict"),
        ({"version": 1, "schema_id": "x"}, "exactly version, schema_id and app"),
        ({"version": 2, "schema_id": "x", "app": APP}, "version 1"),
        ({"version": True, "schema_id": "x", "app": APP}, "version 1"),
        ({"version": 1, "schema_id": "", "app": APP}, "schema_id"),
        (
            {"version": 1, "schema_id": "x", "app": {"type": "array"}},
            "object JSON Schema",
        ),
        *(
            ({"version": 1, "schema_id": "x", "app": {**APP, "x": bad}}, reason)
            for bad, reason in [
                (Decimal("0.5"), "JSON-serializable"),
                (float("nan"), "without NaN"),
                (float("inf"), "without NaN or infinity"),
                ({"a", "b"}, "JSON-serializable"),
                (("1K", "2K"), "round trip"),
                ({1: "one"}, "round trip"),
            ]
        ),
    ],
)
def test_a_malformed_declaration_is_refused_before_publishing(
    declaration: Any, reason: str
) -> None:
    class Malformed:
        def declaration(self) -> Any:
            return declaration

    class MalformedApp(fal.App):
        @fal.endpoint("/edit", billing=Malformed())
        def edit(self, input: EditInput) -> EditOutput:
            return EditOutput(images=1)

    with pytest.raises(ValueError, match=reason):
        MalformedApp.build_metadata()


def test_wma_apps_publish_billing_components() -> None:
    class BilledWmaApp(WmaApp):
        @fal.endpoint("/edit", billing=Declaration("images"))
        def edit(self, input: EditInput) -> EditOutput:
            return EditOutput(images=1)

    metadata = BilledWmaApp.build_metadata()
    assert set(metadata["billing_components"]) == {"/edit"}
    assert "/edit" in metadata["openapi"]["paths"]
