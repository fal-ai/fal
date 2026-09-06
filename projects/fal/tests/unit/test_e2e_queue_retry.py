import json
from types import SimpleNamespace

import httpx
import pytest

from fal import apps
from tests.e2e import test_apps as e2e_apps
from tests.e2e.test_apps import retry_queue_alias_submission  # noqa: F401


@pytest.fixture
def queue_client(monkeypatch):
    responses = []
    requests = []

    def handle(request):
        requests.append(request)
        return responses.pop(0)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(apps, "_get_http_client", lambda: client)
        monkeypatch.setattr(
            apps, "get_credentials", lambda: SimpleNamespace(to_headers=lambda: {})
        )
        now = [0.0]

        def sleep(seconds):
            now[0] += seconds

        monkeypatch.setattr(
            e2e_apps, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=sleep)
        )
        yield responses, requests, now


@pytest.mark.parametrize(
    "app_id, path",
    [
        ("owner/model", "/increment"),
        ("owner/model/increment", ""),
        ("owner-model", "increment"),
    ],
)
def test_queue_run_retries_submission_only(queue_client, app_id, path):
    responses, requests, now = queue_client
    responses.extend(
        [
            httpx.Response(404, json={"detail": "Application 'model' not found"}),
            httpx.Response(404, json={"detail": 'Application "model" not found'}),
            httpx.Response(200, json={"request_id": "accepted"}),
            httpx.Response(200, json={"logs": []}),
            httpx.Response(200, json={"result": 42}),
        ]
    )

    assert apps.run(app_id, {"value": 1}, path=path) == {"result": 42}
    assert [request.method for request in requests] == ["POST"] * 3 + ["GET"] * 2
    assert len({str(request.url) for request in requests[:3]}) == 1
    assert requests[0].url.path == "/owner/model/increment"
    assert all(json.loads(request.content) == {"value": 1} for request in requests[:3])
    assert now[0] == 1


@pytest.mark.parametrize(
    "status, body",
    [
        (404, {"detail": "Application 'different' not found"}),
        (404, {"detail": "Not Found"}),
        (404, {"detail": []}),
        (404, []),
        (404, "not json"),
        (401, {"detail": "Application 'model' not found"}),
        (429, {"error": "Queue is too long"}),
        (500, {"detail": "Application 'model' not found"}),
    ],
)
def test_queue_submission_preserves_other_errors(queue_client, status, body):
    responses, requests, now = queue_client
    response = (
        httpx.Response(status, text=body)
        if isinstance(body, str)
        else httpx.Response(status, json=body)
    )
    responses.append(response)
    with pytest.raises(httpx.HTTPStatusError) as exc:
        apps.submit("owner/model", {})
    assert exc.value.response is response
    assert len(requests) == 1
    assert now[0] == 0


def test_queue_submission_retry_has_a_deadline(queue_client):
    responses, requests, now = queue_client
    responses.extend(
        httpx.Response(404, json={"detail": "Application 'model' not found"})
        for _ in range(120)
    )
    with pytest.raises(httpx.HTTPStatusError):
        apps.submit("owner/model", {})
    assert now[0] == 60
    assert len(requests) == 120


def test_queue_run_does_not_resubmit_after_acceptance(queue_client):
    responses, requests, now = queue_client
    responses.extend(
        [
            httpx.Response(200, json={"request_id": "accepted"}),
            httpx.Response(200, json={"logs": []}),
            httpx.Response(404, json={"detail": "Application 'model' not found"}),
        ]
    )
    with pytest.raises(httpx.HTTPStatusError):
        apps.run("owner/model", {})
    assert [request.method for request in requests] == ["POST", "GET", "GET"]
    assert now[0] == 0
