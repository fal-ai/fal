import pytest

from fal.cli.keys import PRESET_DESCRIPTIONS, _create, _list, _revoke
from fal.cli.main import parse_args
from fal.cli.parser import FalParserExit
from fal.sdk import KeyPreset, KeyScope


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


def test_create_with_preset():
    args = parse_args(
        [
            "keys",
            "create",
            "--preset",
            "API",
            "--alias",
            "ci-deploy",
        ]
    )
    assert args.func == _create
    assert args.preset == "API"
    assert args.desc == "ci-deploy"


def test_create_rejects_preset_and_scope():
    with pytest.raises(FalParserExit):
        parse_args(["keys", "create", "--preset", "API", "--scope", "ADMIN"])


def test_preset_scope_mapping():
    assert KeyPreset.from_scope(KeyScope.ADMIN) == KeyPreset.FULL
    assert KeyPreset.from_scope(KeyScope.API) == KeyPreset.API


def test_create_accepts_v2_only_presets():
    for preset in ("DEPLOY", "READONLY"):
        args = parse_args(["keys", "create", "--preset", preset])
        assert args.preset == preset


def test_every_preset_has_a_description():
    # The interactive picker indexes this by preset, so a new preset without a
    # description would raise a KeyError mid-prompt.
    assert set(PRESET_DESCRIPTIONS) == set(KeyPreset)


def test_list():
    args = parse_args(["keys", "list"])
    assert args.func == _list


def test_revoke():
    args = parse_args(["keys", "revoke", "my-key"])
    assert args.func == _revoke
    assert args.key_id == "my-key"


def test_unknown_scope_lists_as_scopeless():
    # A key minted from DEPLOY/READONLY has no v1 scope; listing must not fail.
    assert KeyScope.from_proto(99) is None  # type: ignore[arg-type]
