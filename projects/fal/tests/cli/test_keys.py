from fal.cli.keys import _create, _list, _revoke
from fal.cli.main import parse_args


def test_create():
    args = parse_args(
        [
            "keys",
            "create",
            "--scope",
            "API",
            "--desc",
            "My test key",
        ]
    )
    assert args.func == _create
    assert args.scope == "API"
    assert args.desc == "My test key"


def test_list():
    args = parse_args(["keys", "list"])
    assert args.func == _list


def test_revoke():
    args = parse_args(["keys", "revoke", "my-key"])
    assert args.func == _revoke
    assert args.key_id == "my-key"
