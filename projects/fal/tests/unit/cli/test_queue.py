from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fal.cli.main import parse_args
from fal.cli.queue import _queue_flush


def test_flush_with_older_than():
    args = parse_args(["queue", "flush", "my-app", "--older-than", "1h"])

    assert args.func == _queue_flush
    assert args.app_name == "my-app"
    assert args.older_than == "1h"


def test_flush_forwards_filters():
    args = parse_args(
        [
            "queue",
            "flush",
            "my-app",
            "--caller-user-id",
            "user-id",
            "--older-than",
            "1h",
        ]
    )
    rest_client = MagicMock(
        base_url="https://rest.example.com",
        get_headers=MagicMock(return_value={"Authorization": "Bearer token"}),
    )
    serverless_client = MagicMock()
    serverless_client._create_rest_client.return_value = rest_client
    http_client = MagicMock()
    http_client.delete.return_value = MagicMock(status_code=HTTPStatus.OK)
    http_client_context = MagicMock()
    http_client_context.__enter__.return_value = http_client
    with patch(
        "fal.api.client.SyncServerlessClient", return_value=serverless_client
    ), patch(
        "fal.api.deploy._get_user",
        return_value=SimpleNamespace(username="app-owner"),
    ), patch(
        "fal.cli.queue.httpx.Client", return_value=http_client_context
    ):
        _queue_flush(args)

    http_client.delete.assert_called_once_with(
        "https://rest.example.com/applications/app-owner/my-app/queue",
        params={"caller_user_id": "user-id", "older_than": "1h"},
    )
