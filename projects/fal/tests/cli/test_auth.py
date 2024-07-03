from fal.cli.auth import _login, _logout, _whoami
from fal.cli.main import parse_args


def test_login():
    args = parse_args(["auth", "login"])
    assert args.func == _login


def test_logout():
    args = parse_args(["auth", "logout"])
    assert args.func == _logout


def test_whoami():
    args = parse_args(["auth", "whoami"])
    assert args.func == _whoami
