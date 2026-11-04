from fal.cli.main import parse_args
from fal.cli.volumes import _create


def test_create():
    args = parse_args(["volumes", "create", "my-volume"])
    assert args.func == _create
    assert args.name == "my-volume"
