from fal.cli.main import parse_args
from fal.cli.secrets import _list, _set, _unset


def test_set():
    args = parse_args(["secrets", "set", "secret1=value1", "secret2=value2"])
    assert args.func == _set
    assert args.secrets == {"secret1": "value1", "secret2": "value2"}


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
