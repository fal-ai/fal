from unittest.mock import MagicMock, patch

import pytest

from fal.cli.main import parse_args
from fal.cli.parser import FalParserExit
from fal.cli.secrets import _list, _set, _unset


def test_set():
    args = parse_args(["secrets", "set", "secret1=value1", "secret2=value2"])
    assert args.func == _set
    assert args.secrets == {"secret1": "value1", "secret2": "value2"}
    assert args.default_exposed is None


def test_set_not_exposed_by_default():
    args = parse_args(["secrets", "set", "secret1=value1", "--not-exposed-by-default"])
    assert args.func == _set
    assert args.secrets == {"secret1": "value1"}
    assert args.default_exposed is False


def test_set_exposed_by_default():
    args = parse_args(["secrets", "set", "secret1=value1", "--exposed-by-default"])
    assert args.func == _set
    assert args.secrets == {"secret1": "value1"}
    assert args.default_exposed is True


def test_set_exposure_flags_are_mutually_exclusive():
    with pytest.raises(FalParserExit):
        parse_args(
            [
                "secrets",
                "set",
                "secret1=value1",
                "--not-exposed-by-default",
                "--exposed-by-default",
            ]
        )


@patch("fal.cli.secrets.SyncServerlessClient")
def test_set_forwards_no_exposure_opinion(mock_client_cls):
    client = MagicMock()
    mock_client_cls.return_value = client
    args = parse_args(["secrets", "set", "secret1=value1"])
    args.func(args)

    # A plain rotation must leave the exposure alone, so the API sees None rather
    # than an explicit True that would reset a per-secret opt-out.
    assert client.secrets.set.call_args.kwargs["default_exposed"] is None


@patch("fal.cli.secrets.SyncServerlessClient")
def test_set_forwards_not_exposed_by_default(mock_client_cls):
    client = MagicMock()
    mock_client_cls.return_value = client
    args = parse_args(["secrets", "set", "secret1=value1", "--not-exposed-by-default"])
    args.func(args)

    assert client.secrets.set.call_args.kwargs["default_exposed"] is False


@patch("fal.cli.secrets.SyncServerlessClient")
def test_set_forwards_exposed_by_default(mock_client_cls):
    client = MagicMock()
    mock_client_cls.return_value = client
    args = parse_args(["secrets", "set", "secret1=value1", "--exposed-by-default"])
    args.func(args)

    assert client.secrets.set.call_args.kwargs["default_exposed"] is True


@patch("fal.cli.secrets.SyncServerlessClient")
def test_set_forwards_exposure_for_every_pair(mock_client_cls):
    client = MagicMock()
    mock_client_cls.return_value = client
    args = parse_args(["secrets", "set", "secret1=value1", "secret2=value2"])
    args.func(args)

    assert client.secrets.set.call_count == 2
    assert [call.args[0] for call in client.secrets.set.call_args_list] == [
        "secret1",
        "secret2",
    ]
    for call in client.secrets.set.call_args_list:
        assert call.kwargs["default_exposed"] is None


def test_set_with_env():
    args = parse_args(["secrets", "set", "secret1=value1", "--env", "dev"])
    assert args.func == _set
    assert args.secrets == {"secret1": "value1"}
    assert args.env == "dev"


def test_list():
    args = parse_args(["secrets", "list"])
    assert args.func == _list


def test_list_with_env():
    args = parse_args(["secrets", "list", "--env", "prod"])
    assert args.func == _list
    assert args.env == "prod"


def test_unset():
    args = parse_args(["secrets", "unset", "secret"])
    assert args.func == _unset
    assert args.secret == "secret"


def test_unset_with_env():
    args = parse_args(["secrets", "unset", "secret", "--env", "staging"])
    assert args.func == _unset
    assert args.secret == "secret"
    assert args.env == "staging"


@patch.dict("os.environ", {"FAL_ENV": "from-env-var"})
def test_set_uses_fal_env_variable():
    args = parse_args(["secrets", "set", "secret1=value1"])
    assert args.env == "from-env-var"


@patch.dict("os.environ", {"FAL_ENV": "from-env-var"})
def test_set_cli_env_overrides_fal_env_variable():
    args = parse_args(["secrets", "set", "secret1=value1", "--env", "cli-env"])
    assert args.env == "cli-env"


@patch.dict("os.environ", {"FAL_ENV": "from-env-var"})
def test_list_uses_fal_env_variable():
    args = parse_args(["secrets", "list"])
    assert args.env == "from-env-var"


@patch.dict("os.environ", {"FAL_ENV": "from-env-var"})
def test_unset_uses_fal_env_variable():
    args = parse_args(["secrets", "unset", "secret"])
    assert args.env == "from-env-var"
