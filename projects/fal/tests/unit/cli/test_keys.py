from types import SimpleNamespace

import pytest

from fal.cli.keys import _create, _list, _revoke
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


def test_create_sends_the_preset_and_no_scope():
    # An old server reads an unset scope as admin, so a preset must never fall
    # back to the deprecated field -- it goes out as policy_preset or not at all.
    from fal.sdk import FalServerlessConnection

    captured = {}

    class _Stub:
        def CreateUserKey(self, request):
            captured["request"] = request
            return SimpleNamespace(key_id="id", key_secret="secret")

    conn = object.__new__(FalServerlessConnection)
    conn._stub = _Stub()
    conn.create_user_key(KeyPreset.FULL, "ci")

    assert captured["request"].policy_preset == "FULL"
    assert captured["request"].HasField("scope") is False


def test_list():
    args = parse_args(["keys", "list"])
    assert args.func == _list


def test_revoke():
    args = parse_args(["keys", "revoke", "my-key"])
    assert args.func == _revoke
    assert args.key_id == "my-key"
