from fal.cli.apps import (
    _delete,
    _delete_rev,
    _list,
    _list_rev,
    _runners,
    _scale,
    _set_rev,
)
from fal.cli.main import parse_args


def test_list():
    args = parse_args(["apps", "list"])
    assert args.func == _list


def test_list_rev():
    args = parse_args(["apps", "list-rev"])
    assert args.func == _list_rev


def test_set_rev():
    args = parse_args(["apps", "set-rev", "myapp", "myrev", "--auth", "public"])
    assert args.func == _set_rev
    assert args.app_name == "myapp"
    assert args.app_rev == "myrev"
    assert args.auth == "public"


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
        ]
    )
    assert args.func == _scale
    assert args.app_name == "myapp"
    assert args.keep_alive == 123
    assert args.max_multiplexing == 321
    assert args.min_concurrency == 7
    assert args.max_concurrency == 10


def test_runners():
    args = parse_args(["apps", "runners", "myapp"])
    assert args.func == _runners


def test_delete():
    args = parse_args(["apps", "delete", "myapp"])
    assert args.func == _delete
    assert args.app_name == "myapp"


def test_delete_rev():
    args = parse_args(["apps", "delete-rev", "myrev"])
    assert args.func == _delete_rev
    assert args.app_rev == "myrev"
