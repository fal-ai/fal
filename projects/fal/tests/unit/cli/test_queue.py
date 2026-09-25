from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fal.cli.main import parse_args
from fal.cli.parser import FalParserExit
from fal.cli.queue import _queue_flush


def test_queue_flush_with_limit():
    args = parse_args(["queue", "flush", "my-app", "--limit", "2"])

    assert args.func == _queue_flush
    assert args.app_name == "my-app"
    assert args.limit == 2
    assert args.caller_user_id is None


def test_queue_flush_sends_limit():
    args = parse_args(["queue", "flush", "my-app", "--limit", "2"])
    args.console = MagicMock()
    rest_client = SimpleNamespace(
        base_url="https://api.example.com",
        get_headers=lambda: {"Authorization": "Key test"},
    )
    response = SimpleNamespace(status_code=200)
    http_client = MagicMock()
    http_client.__enter__.return_value.delete.return_value = response

    with (
        patch("fal.api.client.SyncServerlessClient") as client_type,
        patch(
            "fal.api.deploy._get_user",
            return_value=SimpleNamespace(username="owner"),
        ),
        patch("fal.cli.queue.httpx.Client", return_value=http_client),
    ):
        client_type.return_value._create_rest_client.return_value = rest_client
        args.func(args)

    http_client.__enter__.return_value.delete.assert_called_once_with(
        "https://api.example.com/applications/owner/my-app/queue",
        params={"limit": 2},
    )


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_queue_flush_rejects_non_positive_limit(limit: str):
    with pytest.raises(FalParserExit):
        parse_args(["queue", "flush", "my-app", "--limit", limit])


def test_queue_flush_rejects_limit_with_caller_user_id():
    with pytest.raises(FalParserExit):
        parse_args(
            [
                "queue",
                "flush",
                "my-app",
                "--limit",
                "2",
                "--caller-user-id",
                "caller",
            ]
        )
