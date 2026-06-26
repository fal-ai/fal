from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fal.cli._utils import AppData
from fal.cli.main import parse_args
from fal.cli.queue import _queue_flush, _queue_size


def test_queue_size_parser():
    args = parse_args(["queue", "size", "team-app"])

    assert args.func == _queue_size
    assert args.app_name == "team-app"


def test_queue_flush_parser():
    args = parse_args(["queue", "flush", "team-app"])

    assert args.func == _queue_flush
    assert args.app_name == "team-app"


def _mock_rest_client():
    rest_client = MagicMock()
    rest_client.base_url = "https://api.example"
    rest_client.get_headers.return_value = {"authorization": "Bearer token"}
    return rest_client


@patch("fal.cli.queue.httpx.Client")
@patch("fal.api.deploy._get_user", return_value=SimpleNamespace(username="owner"))
@patch("fal.cli._utils.get_app_data_from_toml", return_value=AppData(team="internal"))
@patch("fal.api.client.SyncServerlessClient")
def test_queue_size_with_team_from_toml(
    mock_client_cls,
    mock_get_app_data_from_toml,
    mock_get_user,
    mock_httpx_client,
):
    rest_client = _mock_rest_client()
    mock_client_cls.return_value._create_rest_client.return_value = rest_client
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"size": 3}
    mock_httpx_client.return_value.__enter__.return_value.get.return_value = (
        mock_response
    )

    args = parse_args(["queue", "size", "team-app"])
    args.host = "my-host"
    args.console = MagicMock()
    _queue_size(args)

    mock_client_cls.assert_called_once_with(host="my-host", team="internal")


@patch("fal.cli.queue.httpx.Client")
@patch("fal.api.deploy._get_user", return_value=SimpleNamespace(username="owner"))
@patch("fal.cli._utils.get_app_data_from_toml", return_value=AppData(team="internal"))
@patch("fal.api.client.SyncServerlessClient")
def test_queue_flush_with_team_from_toml(
    mock_client_cls,
    mock_get_app_data_from_toml,
    mock_get_user,
    mock_httpx_client,
):
    rest_client = _mock_rest_client()
    mock_client_cls.return_value._create_rest_client.return_value = rest_client
    mock_response = MagicMock(status_code=200)
    mock_httpx_client.return_value.__enter__.return_value.delete.return_value = (
        mock_response
    )

    args = parse_args(["queue", "flush", "team-app"])
    args.host = "my-host"
    args.console = MagicMock()
    _queue_flush(args)

    mock_client_cls.assert_called_once_with(host="my-host", team="internal")
