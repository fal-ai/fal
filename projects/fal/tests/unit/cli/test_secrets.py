from fal.cli.main import parse_args
from fal.cli.secrets import _list, _set, _unset


def test_set():
    args = parse_args(["secrets", "set", "secret1=value1", "secret2=value2"])
    assert args.func == _set
    assert args.secrets == {"secret1": "value1", "secret2": "value2"}


def test_list():
    args = parse_args(["secrets", "list"])
    assert args.func == _list


def test_unset():
    args = parse_args(["secrets", "unset", "secret"])
    assert args.func == _unset
    assert args.secret == "secret"
