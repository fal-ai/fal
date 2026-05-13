from unittest.mock import MagicMock, patch

from fal.cli.auth import _login, _logout, _whoami
from fal.cli.main import parse_args


def test_login():
    args = parse_args(["auth", "login"])
    assert args.func == _login


def test_login_with_connection():
    args = parse_args(["auth", "login", "--connection", "github"])
    assert args.func == _login
    assert args.connection == "github"


def test_login_connection_default_is_none():
    args = parse_args(["auth", "login"])
    assert args.connection is None


def test_logout():
    args = parse_args(["auth", "logout"])
    assert args.func == _logout


def test_whoami():
    args = parse_args(["auth", "whoami"])
    assert args.func == _whoami


def test_login_with_connection_runs_auth_flow():
    args = parse_args(["auth", "login", "--connection", "github", "--no-browser"])
    args.console = MagicMock()

    with patch("fal.auth.login") as mock_login, patch(
        "fal.config.Config"
    ) as mock_config_cls, patch(
        "fal.cli.auth._prompt_connection"
    ) as mock_prompt_connection, patch(
        "fal.cli.auth._save_last_connection"
    ) as mock_save_last_connection, patch(
        "fal.cli.auth._set_account"
    ) as mock_set_account:
        mock_config_cls.return_value.get_internal.return_value = None

        assert args.func(args) is None

    mock_prompt_connection.assert_not_called()
    mock_save_last_connection.assert_called_once_with("github")
    mock_login.assert_called_once_with(
        args.console,
        connection="github",
        no_browser=True,
    )
    mock_config_cls.return_value.get_internal.assert_called_once_with("team")
    mock_set_account.assert_called_once_with(args)


def test_logout_runs_auth_flow():
    args = parse_args(["auth", "logout", "--no-browser"])
    args.console = MagicMock()

    with patch("fal.auth.logout") as mock_logout, patch(
        "fal.cli.auth._unset_account"
    ) as mock_unset_account:
        assert args.func(args) is None

    mock_logout.assert_called_once_with(args.console, no_browser=True)
    mock_unset_account.assert_called_once_with(args)


def test_whoami_fetches_current_user_from_credentials():
    args = parse_args(["auth", "whoami"])
    args.console = MagicMock()
    credentials = MagicMock()
    credentials.to_headers.return_value = {"Authorization": "Bearer access-token"}
    user_info = {
        "full_name": "Jane Doe",
        "nickname": "jane",
        "user_id": "user-123",
    }

    with patch("fal.cli.auth.get_credentials", return_value=credentials), patch(
        "fal.cli.auth.current_user_info", return_value=user_info
    ) as mock_current_user_info:
        assert args.func(args) is None

    credentials.to_headers.assert_called_once_with()
    mock_current_user_info.assert_called_once_with(
        {"Authorization": "Bearer access-token"}
    )
    args.console.print.assert_called_once_with("Hello, Jane Doe: 'jane' - 'user-123'")
