import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fal.cli._utils import AppData, resolve_team_from_app_name
from fal.cli.apps import (
    _delete,
    _delete_rev,
    _gpus,
    _list,
    _list_rev,
    _rollout,
    _runners,
    _scale,
    _set_rev,
)
from fal.cli.main import parse_args


def test_list():
    args = parse_args(["apps", "list"])
    assert args.func == _list


def test_list_with_env():
    args = parse_args(["apps", "list", "--env", "dev"])
    assert args.func == _list
    assert args.env == "dev"


@patch.dict("os.environ", {"FAL_ENV": "from-env-var"})
def test_list_uses_fal_env_variable():
    args = parse_args(["apps", "list"])
    assert args.env == "from-env-var"


@patch.dict("os.environ", {"FAL_ENV": "from-env-var"})
def test_list_cli_env_overrides_fal_env_variable():
    args = parse_args(["apps", "list", "--env", "cli-env"])
    assert args.env == "cli-env"


def test_list_with_regions():
    args = parse_args(["apps", "list", "--regions", "us-east", "eu-west"])
    assert args.func == _list
    assert args.regions == ["us-east", "eu-west"]


@patch("fal.cli.apps._apps_table")
@patch("fal.cli.apps.SyncServerlessClient")
def test_list_filters_apps_by_regions(mock_client_cls, mock_apps_table):
    app_us = SimpleNamespace(
        alias="app-us", active_runners=1, valid_regions=["us-east"]
    )
    app_eu = SimpleNamespace(
        alias="app-eu", active_runners=2, valid_regions=["eu-west"]
    )
    app_multi = SimpleNamespace(
        alias="app-multi", active_runners=3, valid_regions=["us-east", "eu-west"]
    )

    mock_client = MagicMock()
    mock_client.apps.list.return_value = [app_us, app_eu, app_multi]
    mock_client_cls.return_value = mock_client
    mock_apps_table.return_value = "table-output"

    args = parse_args(["apps", "list", "--regions", "us-east"])
    args.console = MagicMock()
    _list(args)

    table_apps = mock_apps_table.call_args.args[0]
    assert table_apps == [app_multi, app_us]


def test_list_rev():
    args = parse_args(["apps", "list-rev"])
    assert args.func == _list_rev


def test_list_rev_with_env():
    args = parse_args(["apps", "list-rev", "myapp", "--env", "staging"])
    assert args.func == _list_rev
    assert args.app_name == "myapp"
    assert args.env == "staging"


def test_set_rev():
    args = parse_args(["apps", "set-rev", "myapp", "myrev", "--auth", "public"])
    assert args.func == _set_rev
    assert args.app_name == "myapp"
    assert args.app_rev == "myrev"
    assert args.auth == "public"


def test_set_rev_with_env():
    args = parse_args(
        ["apps", "set-rev", "myapp", "myrev", "--auth", "public", "--env", "prod"]
    )
    assert args.func == _set_rev
    assert args.app_name == "myapp"
    assert args.app_rev == "myrev"
    assert args.auth == "public"
    assert args.env == "prod"


def test_scale():
    args = parse_args(
        [
            "apps",
            "scale",
            "myapp",
            "--keep-alive",
            "123",
            "--max-multiplexing",
            "321",
            "--min-concurrency",
            "7",
            "--max-concurrency",
            "10",
            "--concurrency-buffer",
            "3",
            "--concurrency-buffer-perc",
            "27",
            "--scaling-delay",
            "44",
        ]
    )
    assert args.func == _scale
    assert args.app_name == "myapp"
    assert args.keep_alive == 123
    assert args.max_multiplexing == 321
    assert args.min_concurrency == 7
    assert args.max_concurrency == 10
    assert args.concurrency_buffer == 3
    assert args.concurrency_buffer_perc == 27
    assert args.scaling_delay == 44


def test_scale_with_env():
    args = parse_args(
        ["apps", "scale", "myapp", "--min-concurrency", "1", "--env", "prod"]
    )
    assert args.func == _scale
    assert args.app_name == "myapp"
    assert args.min_concurrency == 1
    assert args.env == "prod"


@patch("fal.cli.apps._apps_table")
@patch("fal.cli._utils.get_app_data_from_toml")
@patch("fal.cli.apps.SyncServerlessClient")
def test_scale_with_team_from_toml(
    mock_client_cls,
    mock_get_app_data_from_toml,
    mock_apps_table,
):
    mock_get_app_data_from_toml.return_value = AppData(team="internal")
    mock_client = MagicMock()
    mock_client.apps.scale.return_value = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_apps_table.return_value = "table-output"

    args = parse_args(["apps", "scale", "team-app", "--min-concurrency", "1"])
    args.host = "my-host"
    args.console = MagicMock()
    _scale(args)

    mock_get_app_data_from_toml.assert_called_once_with(
        "team-app", emit_deprecation_warnings=False
    )
    mock_client_cls.assert_called_once_with(host="my-host", team="internal")
    mock_client.apps.scale.assert_called_once()


@patch("fal.cli.apps._apps_table")
@patch("fal.cli._utils.get_app_data_from_toml")
@patch("fal.cli.apps.SyncServerlessClient")
def test_scale_cli_team_overrides_team_from_toml(
    mock_client_cls,
    mock_get_app_data_from_toml,
    mock_apps_table,
):
    mock_get_app_data_from_toml.return_value = AppData(team="internal")
    mock_client = MagicMock()
    mock_client.apps.scale.return_value = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_apps_table.return_value = "table-output"

    args = parse_args(
        [
            "apps",
            "scale",
            "team-app",
            "--min-concurrency",
            "1",
            "--team",
            "cli-team",
        ]
    )
    args.host = "my-host"
    args.console = MagicMock()
    _scale(args)

    mock_client_cls.assert_called_once_with(host="my-host", team="cli-team")


@patch("fal.cli._utils.get_app_data_from_toml")
def test_resolve_team_from_app_name_uses_toml(mock_get_app_data_from_toml):
    mock_get_app_data_from_toml.return_value = AppData(team="internal")

    team = resolve_team_from_app_name("team-app", None)

    assert team == "internal"
    mock_get_app_data_from_toml.assert_called_once_with(
        "team-app", emit_deprecation_warnings=False
    )


@patch("fal.cli._utils.get_app_data_from_toml")
def test_resolve_team_from_app_name_keeps_cli_override(mock_get_app_data_from_toml):
    mock_get_app_data_from_toml.return_value = AppData(team="internal")

    assert resolve_team_from_app_name("team-app", "cli-team") == "cli-team"


@patch("fal.cli._utils.get_app_data_from_toml")
def test_resolve_team_from_app_name_ignores_missing_toml(mock_get_app_data_from_toml):
    mock_get_app_data_from_toml.side_effect = ValueError(
        "App team-app not found in pyproject.toml"
    )

    assert resolve_team_from_app_name("team-app", None) is None


@patch("fal.cli._utils.get_app_data_from_toml")
def test_resolve_team_from_app_name_ignores_python_files(mock_get_app_data_from_toml):
    assert resolve_team_from_app_name("app.py", None) is None

    mock_get_app_data_from_toml.assert_not_called()


@patch("fal.cli._utils.get_app_data_from_toml", return_value=AppData(team="internal"))
@patch("fal.cli.apps.SyncServerlessClient")
def test_rollout_with_team_from_toml(mock_client_cls, mock_get_app_data_from_toml):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    args = parse_args(["apps", "rollout", "team-app"])
    args.host = "my-host"
    args.console = MagicMock()
    _rollout(args)

    mock_client_cls.assert_called_once_with(host="my-host", team="internal")
    mock_client.apps.rollout.assert_called_once_with(
        "team-app", force=False, environment_name=None
    )


@patch(
    "fal.cli.apps.runners.runners_table",
    return_value=SimpleNamespace(columns=[object()]),
)
@patch(
    "fal.cli.apps.runners.runners_requests_table",
    return_value=SimpleNamespace(rows=[]),
)
@patch("fal.cli._utils.get_app_data_from_toml", return_value=AppData(team="internal"))
@patch("fal.cli.apps.SyncServerlessClient")
def test_runners_with_team_from_toml(
    mock_client_cls,
    mock_get_app_data_from_toml,
    mock_requests_table,
    mock_runners_table,
):
    mock_client = MagicMock()
    mock_client.apps.runners.return_value = []
    mock_client_cls.return_value = mock_client

    args = parse_args(["apps", "runners", "team-app"])
    args.host = "my-host"
    args.console = MagicMock()
    _runners(args)

    mock_client_cls.assert_called_once_with(host="my-host", team="internal")
    mock_client.apps.runners.assert_called_once_with(
        "team-app", since=None, state=None, environment_name=None
    )


@patch("fal.cli._utils.get_app_data_from_toml", return_value=AppData(team="internal"))
@patch("fal.cli.apps.SyncServerlessClient")
def test_gpus_with_team_from_toml(mock_client_cls, mock_get_app_data_from_toml):
    mock_client_cls.return_value = _mock_gpus_client(_GPUS_PAYLOAD)

    args = parse_args(["apps", "gpus", "team-app"])
    args.host = "my-host"
    args.console = MagicMock()
    _gpus(args)

    mock_client_cls.assert_called_once_with(host="my-host", team="internal")


@patch("fal.cli.apps._apps_table", return_value="table-output")
@patch("fal.cli._utils.get_app_data_from_toml", return_value=AppData(team="internal"))
@patch("fal.cli.apps.get_client")
def test_list_rev_with_team_from_toml(
    mock_get_client,
    mock_get_app_data_from_toml,
    mock_apps_table,
):
    mock_connection = MagicMock()
    mock_connection.list_applications.return_value = []
    mock_get_client.return_value.connect.return_value.__enter__.return_value = (
        mock_connection
    )

    args = parse_args(["apps", "list-rev", "team-app"])
    args.host = "my-host"
    args.console = MagicMock()
    _list_rev(args)

    mock_get_client.assert_called_once_with("my-host", "internal")


@patch("fal.cli.apps._apps_table", return_value="table-output")
@patch("fal.cli._utils.get_app_data_from_toml", return_value=AppData(team="internal"))
@patch("fal.cli.apps.get_client")
def test_set_rev_with_team_from_toml(
    mock_get_client,
    mock_get_app_data_from_toml,
    mock_apps_table,
):
    mock_connection = MagicMock()
    mock_connection.create_alias.return_value = MagicMock()
    mock_get_client.return_value.connect.return_value.__enter__.return_value = (
        mock_connection
    )

    args = parse_args(["apps", "set-rev", "team-app", "app-rev"])
    args.host = "my-host"
    args.console = MagicMock()
    _set_rev(args)

    mock_get_client.assert_called_once_with("my-host", "internal")


@patch("fal.cli._utils.get_app_data_from_toml", return_value=AppData(team="internal"))
@patch("fal.cli.apps.get_client")
def test_delete_with_team_from_toml(mock_get_client, mock_get_app_data_from_toml):
    mock_connection = MagicMock()
    mock_connection.delete_alias.return_value = "deleted"
    mock_get_client.return_value.connect.return_value.__enter__.return_value = (
        mock_connection
    )

    args = parse_args(["apps", "delete", "team-app"])
    args.host = "my-host"
    args.console = MagicMock()
    _delete(args)

    mock_get_client.assert_called_once_with("my-host", "internal")


def test_rollout():
    args = parse_args(["apps", "rollout", "myapp"])
    assert args.func == _rollout
    assert args.app_name == "myapp"


def test_rollout_with_env():
    args = parse_args(["apps", "rollout", "myapp", "--force", "--env", "staging"])
    assert args.func == _rollout
    assert args.app_name == "myapp"
    assert args.force is True
    assert args.env == "staging"


def test_runners():
    args = parse_args(["apps", "runners", "myapp"])
    assert args.func == _runners


def test_runners_with_env():
    args = parse_args(["apps", "runners", "myapp", "--env", "dev"])
    assert args.func == _runners
    assert args.app_name == "myapp"
    assert args.env == "dev"


def test_delete():
    args = parse_args(["apps", "delete", "myapp"])
    assert args.func == _delete
    assert args.app_name == "myapp"


def test_delete_with_env():
    args = parse_args(["apps", "delete", "myapp", "--env", "dev"])
    assert args.func == _delete
    assert args.app_name == "myapp"
    assert args.env == "dev"


def test_delete_rev():
    args = parse_args(["apps", "delete-rev", "myrev"])
    assert args.func == _delete_rev
    assert args.app_rev == "myrev"


_GPUS_PAYLOAD = {"gpus": {"H100": 8, "B200": 3, "L40": 5}, "total": 16}


def test_apps_gpus_parser():
    args = parse_args(["apps", "gpus", "myapp"])
    assert args.func == _gpus
    assert args.app_name == "myapp"


def _mock_gpus_client(payload):
    client = MagicMock()
    client.apps.gpus.return_value = payload
    return client


@patch("fal.cli.apps.SyncServerlessClient")
def test_apps_gpus_json(mock_client_cls):
    mock_client_cls.return_value = _mock_gpus_client(_GPUS_PAYLOAD)
    args = parse_args(["apps", "gpus", "fal-ai/flux", "--json"])
    args.console = MagicMock()
    args.func(args)

    output = args.console.print.call_args[0][0]
    result = json.loads(output)
    assert result["total"] == 16
    # Sorted by count desc by render_gpus
    assert list(result["gpus"].items()) == [("H100", 8), ("L40", 5), ("B200", 3)]


@patch("fal.cli.apps.SyncServerlessClient")
def test_apps_gpus_pretty_runs(mock_client_cls):
    mock_client_cls.return_value = _mock_gpus_client(_GPUS_PAYLOAD)
    args = parse_args(["apps", "gpus", "fal-ai/flux"])
    args.console = MagicMock()
    args.func(args)

    printed = " ".join(str(call.args[0]) for call in args.console.print.call_args_list)
    assert "Total: 16" in printed


@patch("fal.cli.apps.SyncServerlessClient")
def test_apps_gpus_app_not_found(mock_client_cls):
    client = MagicMock()
    client.apps.gpus.side_effect = RuntimeError(
        "Application 'fal-ai/missing' not found in metrics."
    )
    mock_client_cls.return_value = client
    args = parse_args(["apps", "gpus", "fal-ai/missing"])
    args.console = MagicMock()

    with pytest.raises(RuntimeError, match="not found in metrics"):
        args.func(args)
