from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fal.cli.apps import (
    _delete,
    _delete_rev,
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
