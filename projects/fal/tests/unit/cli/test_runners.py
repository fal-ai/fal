from unittest.mock import MagicMock, patch

from fal.cli.main import parse_args
from fal.cli.runners import _list


def test_list():
    args = parse_args(["runners", "list"])
    assert args.func == _list
    assert args.skip_leases is False


@patch("fal.cli.runners.SyncServerlessClient")
def test_list_includes_leases_by_default(mock_client_cls):
    mock_client = MagicMock()
    mock_client.runners.list.return_value = []
    mock_client_cls.return_value = mock_client

    args = parse_args(["runners", "list", "--output", "json"])
    args.console = MagicMock()
    _list(args)

    mock_client.runners.list.assert_called_once_with(since=None, include_leases=True)


@patch("fal.cli.runners.SyncServerlessClient")
def test_list_skips_leases(mock_client_cls):
    mock_client = MagicMock()
    mock_client.runners.list.return_value = []
    mock_client_cls.return_value = mock_client

    args = parse_args(["runners", "list", "--skip-leases", "--output", "json"])
    args.console = MagicMock()
    _list(args)

    mock_client.runners.list.assert_called_once_with(since=None, include_leases=False)
