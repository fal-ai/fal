from types import SimpleNamespace

import pytest
from dev.app import fakes
from dev.app.fakes import make_tree_session

from fal.cli.main import parse_args
from fal.dev.account import Account
from fal.dev.app import client, deploy
from fal.dev.app import session as app_session
from fal.dev.errors import DevError

repo = fakes.repo

pytestmark = pytest.mark.usefixtures("dev_account")


def test_start_ignores_the_pyproject_team(repo, monkeypatch):
    from fal.dev.cli import app as dev_app

    calls = {}

    def deploy_box(file_path, class_name, **kwargs):
        calls.update(kwargs, file_path=file_path)
        return deploy.DeployedBox(
            class_name="Demo",
            run_url="https://fal.run/me/dev-demo",
            layout=deploy.Layout(
                repo, [".", "pkg/models"], "pkg/models", ".", [r"\.bin$"]
            ),
        )

    monkeypatch.setattr(deploy, "deploy_box", deploy_box)
    monkeypatch.setattr(deploy, "box_exists", lambda *a, **k: False)
    monkeypatch.setattr(app_session, "SESSIONS_DIR", repo / ".sessions")
    monkeypatch.setattr(client, "boot", lambda session: {"app": {}})
    args = parse_args(["dev", "app", "start", "demo", "--team", "mine"])
    dev_app.cmd_start(args)

    assert calls["team"] == "mine"
    assert calls["root"] == repo
    assert calls["explicit_root"] is False
    session = app_session.find_session(None)
    assert session.name.startswith("dev-demo-")
    assert session.account == Account("mine", False)
    assert session.module == "pkg.models.app"
    assert session.paths == [".", "pkg/models"]
    assert session.workdir == "pkg/models"
    assert session.ignore == [r"\.bin$"]


def test_an_index_that_refuses_the_box_is_explained(monkeypatch):
    from fal.dev.cli import app as dev_app

    session = SimpleNamespace(account=Account("mine", False))
    monkeypatch.setattr(dev_app, "_booted", lambda args: session)
    errors = iter(
        [
            DevError(
                "deps failed (400): HTTP status client error (401 Unauthorized) for url"
            ),
            DevError("deps failed (400): no matching distribution for nope"),
            DevError("deps failed (403): Forbidden"),
        ]
    )

    def deps_call(session, payload, **kwargs):
        raise next(errors)

    monkeypatch.setattr(client, "deps_call", deps_call)
    args = parse_args(["dev", "app", "deps", "install", "private-pkg"])
    with pytest.raises(DevError, match="fal_wheels_access") as info:
        dev_app.cmd_deps_install(args)
    assert "--team" in str(info.value)
    assert "runs as team mine" in str(info.value)
    assert "401 Unauthorized" in str(info.value)
    # Other failures, and a refused CLI key, are passed through as they are.
    for _ in range(2):
        with pytest.raises(DevError) as info:
            dev_app.cmd_deps_install(args)
        assert "fal_wheels_access" not in str(info.value)


@pytest.mark.parametrize(
    "message",
    [
        "401 Client Error: Unauthorized for url: https://huggingface.co/api/x",
        "Downloaded 401 packages from https://pypi.example/simple",
        "deploy failed (403): Forbidden",
        "start failed (401): Unauthorized",
    ],
)
def test_other_401s_get_no_index_hint(message):
    assert client.index_hint(message, Account(None, True)) == message


def test_start_runs_as_the_remembered_team_else_your_personal_account(
    repo, monkeypatch, capsys
):
    from fal.dev.cli import app as dev_app

    seen = []
    monkeypatch.setattr(
        deploy,
        "deploy_box",
        lambda *a, **k: deploy.DeployedBox(
            "Demo", "u", deploy.Layout(repo, ["."], ".", ".", [])
        ),
    )
    monkeypatch.setattr(deploy, "box_exists", lambda *a, **k: False)
    monkeypatch.setattr(app_session, "SESSIONS_DIR", repo / ".sessions")
    monkeypatch.setattr(client, "boot", lambda session: {"app": {}})
    monkeypatch.setattr(
        dev_app, "confirm_deploy", lambda *a, **k: seen.append((a[1], k["yes"]))
    )
    start = ["dev", "app", "start", "demo", "--name", "box"]

    dev_app.cmd_start(parse_args(start))
    assert app_session.find_session("box").account == Account("me", True)

    dev_app.cmd_start(parse_args([*start, "--team", "mine", "--replace", "--yes"]))
    dev_app.cmd_start(parse_args([*start, "--replace"]))
    assert app_session.find_session("box").account == Account("mine", False)
    assert seen == [
        (Account("me", True), False),
        (Account("mine", False), True),
        (Account("mine", False), False),
    ]
    assert capsys.readouterr().err.splitlines() == [
        "Team: personal (me)",
        "Team: mine",
        "Team: mine",
    ]


def test_start_explains_an_index_that_refuses_the_environment_build(repo, monkeypatch):
    from fal.dev.cli import app as dev_app

    def refuse(*args, **kwargs):
        raise RuntimeError(
            "Failed to build the environment: error: HTTP status client error "
            "(401 Unauthorized) for url (https://wheels.example/simple/pkg/)"
        )

    monkeypatch.setattr(deploy, "start_box", refuse)
    args = parse_args(["dev", "app", "start", "demo", "--team", "mine"])
    with pytest.raises(DevError, match="(?s)fal_wheels_access.*runs as team mine"):
        dev_app.cmd_start(args)
    args = parse_args(["dev", "app", "start", "demo", "--name", "other"])
    with pytest.raises(DevError, match="runs as your personal account"):
        dev_app.cmd_start(args)


def test_start_saves_session_env(repo, monkeypatch):
    from fal.dev.cli import app as dev_app

    monkeypatch.setattr(
        deploy,
        "deploy_box",
        lambda *a, **k: deploy.DeployedBox(
            "Demo", "u", deploy.Layout(repo, ["."], ".", ".", [])
        ),
    )
    monkeypatch.setattr(deploy, "box_exists", lambda *a, **k: False)
    monkeypatch.setattr(app_session, "SESSIONS_DIR", repo / ".sessions")
    monkeypatch.setattr(client, "boot", lambda session: {"app": {}})
    args = parse_args(
        ["dev", "app", "start", "demo", "--team", "t", "--session-env", "MY_NS"]
    )
    dev_app.cmd_start(args)
    assert app_session.find_session(None).session_env == ["MY_NS"]

    # A name the box would refuse stops the start before anything deploys.
    deployed = []
    monkeypatch.setattr(deploy, "deploy_box", lambda *a, **k: deployed.append(1))
    args = parse_args(
        ["dev", "app", "start", "demo", "--team", "t", "--session-env", "LD_PRELOAD"]
    )
    with pytest.raises(DevError, match="cannot set"):
        dev_app.cmd_start(args)
    assert not deployed


@pytest.mark.parametrize(
    "account, key, team, acts, outcome",
    [
        (Account("me", True), None, "red", True, "--team red is ignored"),
        (Account("me", True), None, "ME", True, ""),
        (
            Account("acme", False, via_key=True),
            Account("ACME", False, True),
            None,
            True,
            "",
        ),
        (Account("acme", False, via_key=True), None, None, True, "set FAL_KEY"),
        (
            Account("acme", False, via_key=True),
            Account("other", False, True),
            None,
            True,
            "belongs to 'other'",
        ),
        (
            Account(None, False, via_key=True),
            Account("any", False, True),
            None,
            True,
            "",
        ),
        # Looking at or stopping a box is never refused.
        (Account("acme", False, via_key=True), None, None, False, ""),
        (
            Account("acme", False, via_key=True),
            Account("other", False, True),
            None,
            False,
            "",
        ),
    ],
)
def test_a_started_box_keeps_its_account(
    tmp_path, monkeypatch, capsys, account, key, team, acts, outcome
):
    from fal.dev.cli import app as dev_app

    session = make_tree_session(tmp_path, "a.py")
    session.account = account
    monkeypatch.setattr(dev_app, "find_session", lambda name: session)
    monkeypatch.setattr(dev_app, "key_account", lambda: key)
    args = SimpleNamespace(name=None, team=team)

    if outcome in ("set FAL_KEY", "belongs to 'other'"):
        with pytest.raises(DevError, match=outcome):
            dev_app._session(args, acts_as_account=acts)
    else:
        assert dev_app._session(args, acts_as_account=acts) is session
        assert outcome in capsys.readouterr().err
