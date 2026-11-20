from fal.cli.main import parse_args
from fal.cli.volumes import _create, _delete, _list


def test_create():
    args = parse_args(["volumes", "create", "my-volume"])
    assert args.func == _create
    assert args.name == "my-volume"


def test_list():
    args = parse_args(["volumes", "list", "--json"])
    assert args.func == _list
    assert args.output == "json"


def test_delete():
    args = parse_args(["volumes", "delete", "my-volume"])
    assert args.func == _delete
    assert args.name == "my-volume"
