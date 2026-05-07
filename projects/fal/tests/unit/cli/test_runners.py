import json
from unittest.mock import MagicMock, patch

import pytest

from fal.cli.main import parse_args
from fal.cli.runners import _gpus

_GPUS_PAYLOAD = {
    "gpus": {"H100": 394, "B200": 353, "H200": 334},
    "total": 1120,
}


def test_gpus_parser_registered():
    args = parse_args(["runners", "gpus"])
    assert args.func == _gpus


def _mock_client(payload):
    client = MagicMock()
    client.runners.gpus.return_value = payload
    return client


@patch("fal.cli.runners.SyncServerlessClient")
def test_gpus_json(mock_client_cls):
    mock_client_cls.return_value = _mock_client(_GPUS_PAYLOAD)

    args = parse_args(["runners", "gpus", "--json"])
    args.console = MagicMock()
    args.func(args)

    output = args.console.print.call_args[0][0]
    result = json.loads(output)

    assert result["total"] == 1120
    # Sorted by gpus desc by render_gpus
    assert list(result["gpus"].items()) == [
        ("H100", 394),
        ("B200", 353),
        ("H200", 334),
    ]


@patch("fal.cli.runners.SyncServerlessClient")
def test_gpus_pretty_runs(mock_client_cls):
    mock_client_cls.return_value = _mock_client(_GPUS_PAYLOAD)

    args = parse_args(["runners", "gpus"])
    args.console = MagicMock()
    args.func(args)

    printed = " ".join(str(call.args[0]) for call in args.console.print.call_args_list)
    assert "Total: 1120" in printed


@patch("fal.cli.runners.SyncServerlessClient")
def test_gpus_empty(mock_client_cls):
    mock_client_cls.return_value = _mock_client({"gpus": {}, "total": 0})

    args = parse_args(["runners", "gpus", "--json"])
    args.console = MagicMock()
    args.func(args)

    output = args.console.print.call_args[0][0]
    assert json.loads(output) == {"gpus": {}, "total": 0}


@patch("fal.cli.runners.SyncServerlessClient")
def test_gpus_propagates_api_error(mock_client_cls):
    client = MagicMock()
    client.runners.gpus.side_effect = RuntimeError("Failed to fetch metrics: 500 boom")
    mock_client_cls.return_value = client

    args = parse_args(["runners", "gpus"])
    args.console = MagicMock()

    with pytest.raises(RuntimeError, match="Failed to fetch metrics"):
        args.func(args)
