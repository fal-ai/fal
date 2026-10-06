from __future__ import annotations

from typing import Any, Callable

import pytest
from fastapi import Header, Response
from pydantic import BaseModel

import fal
from fal._estimate import ESTIMATE_HEADER


@pytest.fixture(autouse=True)
def isolate_agent_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IS_ISOLATE_AGENT", "1")


class Declaration:
    def __init__(self, metric: str) -> None:
        self.metric = metric

    def declaration(self) -> dict[str, Any]:
        return {"version": 1, "schema_id": f"id-{self.metric}", "app": {}}


class Report:
    def __init__(self, images: int) -> None:
        self.images = images

    def to_json(self) -> str:
        return f'{{"components":[{{"metric":"images","quantity":{self.images}}}]}}'


class EditInput(BaseModel):
    prompt: str
    num_images: int = 1


class EditOutput(BaseModel):
    images: int


CALLS: list[str] = []


def estimate_edit(input: EditInput) -> Report:
    return Report(input.num_images)


async def estimate_sync(input: EditInput) -> dict[str, int]:
    return {"images": input.num_images}


class EstimateApp(fal.App):
    @fal.endpoint("/edit", billing=Declaration("images"), estimate=estimate_edit)
    async def edit(
        self,
        input: EditInput,
        response: Response,
        caller: str | None = Header(None, alias="x-caller"),
    ) -> EditOutput:
        CALLS.append(input.prompt)
        response.headers["x-fal-billable-units"] = str(input.num_images)
        return EditOutput(images=input.num_images)

    @fal.endpoint("/sync", billing=Declaration("seconds"), estimate=estimate_sync)
    def sync_run(self, input: EditInput) -> EditOutput:
        CALLS.append(input.prompt)
        return EditOutput(images=input.num_images)

    @fal.endpoint("/plain")
    def plain(self, input: EditInput) -> EditOutput:
        return EditOutput(images=input.num_images)


class PlainApp(fal.App):
    @fal.endpoint("/edit")
    async def edit(self, input: EditInput, response: Response) -> EditOutput:
        return EditOutput(images=input.num_images)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    CALLS.clear()
    return TestClient(EstimateApp()._build_app())


def test_estimate_header_runs_estimate_instead_of_handler(client) -> None:
    response = client.post(
        "/edit",
        json={"prompt": "hi", "num_images": 3},
        headers={ESTIMATE_HEADER: "1"},
    )
    assert response.status_code == 200
    assert response.json() == {"components": [{"metric": "images", "quantity": 3}]}
    assert response.headers["x-fal-billable-units"] == "0"
    assert CALLS == []


def test_requests_without_the_header_run_the_handler(client) -> None:
    response = client.post("/edit", json={"prompt": "hi", "num_images": 2})
    assert response.status_code == 200
    assert response.json() == {"images": 2}
    assert response.headers["x-fal-billable-units"] == "2"
    assert CALLS == ["hi"]
    response = client.post(
        "/edit", json={"prompt": "again"}, headers={ESTIMATE_HEADER: "0"}
    )
    assert response.json() == {"images": 1}
    assert CALLS == ["hi", "again"]


def test_sync_handlers_and_async_estimates_are_supported(client) -> None:
    response = client.post(
        "/sync",
        json={"prompt": "hi", "num_images": 4},
        headers={ESTIMATE_HEADER: "true"},
    )
    assert response.json() == {"images": 4}
    assert response.headers["x-fal-billable-units"] == "0"
    assert CALLS == []
    assert client.post("/sync", json={"prompt": "run"}).json() == {"images": 1}
    assert CALLS == ["run"]


def test_estimates_reuse_request_validation(client) -> None:
    response = client.post(
        "/edit", json={"num_images": 3}, headers={ESTIMATE_HEADER: "1"}
    )
    assert response.status_code == 422
    assert response.headers["x-fal-billable-units"] == "0"


def test_estimate_mode_does_not_change_the_openapi_spec() -> None:
    with_estimate = EstimateApp().openapi()["paths"]["/edit"]["post"]
    without = PlainApp().openapi()["paths"]["/edit"]["post"]
    assert with_estimate["requestBody"] == without["requestBody"]
    assert with_estimate["responses"] == without["responses"]
    assert [p["name"] for p in with_estimate["parameters"]] == ["x-caller"]
    assert "billing" not in str(EstimateApp().openapi()).lower()


def test_billing_components_are_published_outside_openapi() -> None:
    metadata = EstimateApp.build_metadata()
    assert metadata["billing_components"] == {
        "/edit": {"version": 1, "schema_id": "id-images", "app": {}, "estimate": True},
        "/sync": {
            "version": 1,
            "schema_id": "id-seconds",
            "app": {},
            "estimate": True,
        },
    }
    assert "billing_components" not in PlainApp.build_metadata()


class OtherInput(BaseModel):
    prompt: str


def estimate_other(input: OtherInput) -> Report:
    return Report(1)


def estimate_two(input: EditInput, extra: int) -> Report:
    return Report(extra)


def test_estimates_must_take_only_the_endpoint_input_model() -> None:
    estimates: tuple[Callable[..., Any], ...] = (estimate_other, estimate_two)
    for estimate in estimates:

        class WrongApp(fal.App):
            @fal.endpoint("/edit", billing=Declaration("images"), estimate=estimate)
            def edit(self, input: EditInput) -> EditOutput:
                return EditOutput(images=1)

        with pytest.raises(ValueError, match="plain function"):
            WrongApp()._build_app()


def test_estimates_require_billing_and_http_endpoints() -> None:
    with pytest.raises(ValueError, match="needs billing"):
        fal.endpoint("/edit", estimate=estimate_edit)
    with pytest.raises(TypeError, match="declaration"):
        fal.endpoint("/edit", billing=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Websocket"):
        fal.endpoint("/ws", is_websocket=True, billing=Declaration("images"))
