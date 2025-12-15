from fal.cli.environments import _create, _delete, _list
from fal.cli.main import parse_args


def test_list():
    args = parse_args(["environments", "list"])
    assert args.func == _list


def test_list_alias():
    args = parse_args(["env", "list"])
    assert args.func == _list


def test_list_json_output():
    args = parse_args(["environments", "list", "--output", "json"])
    assert args.func == _list
    assert args.output == "json"


def test_create():
    args = parse_args(["environments", "create", "dev"])
    assert args.func == _create
    assert args.name == "dev"


def test_create_with_description():
    args = parse_args(
        ["environments", "create", "dev", "--description", "Development environment"]
    )
    assert args.func == _create
    assert args.name == "dev"
    assert args.description == "Development environment"


def test_create_alias():
    args = parse_args(["env", "create", "staging"])
    assert args.func == _create
    assert args.name == "staging"


def test_delete():
    args = parse_args(["environments", "delete", "dev"])
    assert args.func == _delete
    assert args.name == "dev"


def test_delete_alias():
    args = parse_args(["env", "delete", "staging"])
    assert args.func == _delete
    assert args.name == "staging"


def test_delete_has_yes_flag():
    """Test that the delete command accepts --yes flag"""
    args = parse_args(["environments", "delete", "dev", "--yes"])
    assert args.func == _delete
    assert args.name == "dev"
    assert args.yes is True


def test_delete_without_yes_flag():
    """Test that the delete command works without --yes flag (requires confirmation)"""
    args = parse_args(["environments", "delete", "dev"])
    assert args.func == _delete
    assert args.name == "dev"
    assert args.yes is False


def test_delete_alias_with_yes():
    """Test that the delete alias accepts --yes flag"""
    args = parse_args(["env", "delete", "staging", "--yes"])
    assert args.func == _delete
    assert args.name == "staging"
    assert args.yes is True
