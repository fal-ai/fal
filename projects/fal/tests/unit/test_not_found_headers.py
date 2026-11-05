from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import fal


@pytest.fixture(autouse=True)
def isolate_agent_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IS_ISOLATE_AGENT", "1")


class NotFoundApp(fal.App):
    @fal.endpoint("/lookup")
    def lookup(self) -> dict:
        raise HTTPException(404, "No such item", headers={"x-fal-billable-units": "0"})


def client() -> TestClient:
    return TestClient(NotFoundApp(_allow_init=True)._build_app())


def test_an_app_404_keeps_its_message_and_headers() -> None:
    response = client().post("/lookup")

    assert response.status_code == 404
    assert response.json() == {"detail": "No such item"}
    assert response.headers["x-fal-billable-units"] == "0"


def test_an_unknown_path_is_still_free() -> None:
    response = client().post("/missing")

    assert response.status_code == 404
    assert response.json() == {"detail": "Path /missing not found"}
    assert response.headers["x-fal-billable-units"] == "0"
