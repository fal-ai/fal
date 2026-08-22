from unittest.mock import Mock

import pytest

from fal import apps
from fal.sdk import Credentials


@pytest.fixture
def queue_client(monkeypatch):
    submit_response = Mock()
    submit_response.json.return_value = {"request_id": "request-id"}

    status_response = Mock(status_code=202)
    status_response.json.return_value = {"status": "IN_PROGRESS", "logs": []}

    client = Mock()
    client.post.return_value = submit_response
    client.get.return_value = status_response
    monkeypatch.setattr(apps, "_get_http_client", lambda: client)
    monkeypatch.setattr(apps, "get_credentials", lambda: Credentials())
    return client


def test_submit_keeps_endpoint_in_url_and_base_app_id_in_handle(queue_client):
    handle = apps.submit("owner/app/reset", arguments={})

    queue_client.post.assert_called_once_with(
        apps._QUEUE_URL_FORMAT.format(app_id="owner/app/reset"),
        json={},
        headers={},
    )
    assert handle.app_id == "owner/app"

    assert isinstance(handle.status(), apps.InProgress)
    queue_client.get.assert_called_once_with(
        apps._QUEUE_URL_FORMAT.format(app_id="owner/app")
        + "/requests/request-id/status/",
        headers={},
        params={"logs": 0},
    )


@pytest.mark.parametrize("path", ["increment", "/increment"])
def test_submit_path_does_not_require_leading_slash(queue_client, path):
    apps.submit("owner/app", arguments={}, path=path)

    queue_client.post.assert_called_once_with(
        apps._QUEUE_URL_FORMAT.format(app_id="owner/app") + "/increment",
        json={},
        headers={},
    )
