from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

import fal


@pytest.fixture(autouse=True)
def isolate_agent_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IS_ISOLATE_AGENT", "1")


class Declaration:
    def __init__(self, metric: str) -> None:
        self.metric = metric

    def declaration(self) -> dict[str, Any]:
        return {"version": 1, "schema_id": f"id-{self.metric}", "app": {}}


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
        "/edit": {"version": 1, "schema_id": "id-images", "app": {}},
        "/upscale": {"version": 1, "schema_id": "id-megapixels", "app": {}},
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
    with pytest.raises(ValueError, match="Websocket"):
        fal.endpoint("/ws", is_websocket=True, billing=Declaration("images"))
