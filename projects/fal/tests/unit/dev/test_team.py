import argparse
import io
import json
import re
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.error import HTTPError, URLError

import pytest

from fal.cli.main import parse_args
from fal.cli.parser import FalParserExit
from fal.dev import state, team
from fal.dev.account import Account
from fal.dev.app import session as app_session
from fal.dev.cli import box as box_commands
from fal.dev.cli.common import get_box
from fal.dev.errors import DevError
from fal.exceptions import FalServerlessException

key_account = team.key_account
PERSONAL = Account("me", True)
RED = Account("red", False)
ACCOUNTS = [
    {"nickname": "me", "is_personal": True},
    {"nickname": "red", "is_personal": False},
    {"nickname": "blue", "is_personal": False},
]


@pytest.fixture(autouse=True)
def dev_state(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(team, "key_account", lambda: None)
    monkeypatch.setattr(team, "accounts", lambda: ACCOUNTS)


def _args(name="demo", team=None, host="api.example"):
    return argparse.Namespace(name=name, host=host, team=team)


def _app(alias):
    return SimpleNamespace(alias=alias)


def _stdin(monkeypatch, tty):
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: tty))


def _production(monkeypatch, apps):
    asked = []
    monkeypatch.setattr(
        team, "production_apps", lambda client: asked.append(client) or apps
    )
    return asked


def _confirm(client, account=RED, app="devbox-demo", yes=False, name="demo"):
    team.confirm_deploy(client, account, app, "api.example", yes=yes, name=name)


def test_a_box_runs_as_the_personal_account_even_with_a_default_team(monkeypatch):
    monkeypatch.setattr(
        "fal.config.Config.get_internal", lambda self, key: "default-team"
    )

    box = get_box(_args())

    assert box.account == PERSONAL
    assert box.client.team == "me"


def test_a_box_without_a_personal_account_is_an_error(monkeypatch):
    monkeypatch.setattr(team, "accounts", lambda: ACCOUNTS[1:])

    with pytest.raises(DevError, match="no personal account"):
        get_box(_args())


def test_an_api_key_picks_the_account(monkeypatch):
    key = Account("acme", False, via_key=True)
    monkeypatch.setattr(team, "key_account", lambda: key)

    box = get_box(_args(team="red"))

    assert box.account == key
    assert state.load_box("devbox-demo", "api.example") == {}


def test_key_account_is_named_by_its_owner(monkeypatch):
    monkeypatch.setattr("fal.auth.key_credentials", lambda: ("id", "secret"))
    monkeypatch.setattr(
        "fal.auth.current_user_info", lambda headers: {"nickname": "acme"}
    )

    assert key_account() == Account("acme", False, via_key=True)


def _key_lookup(monkeypatch, error):
    def lookup(headers):
        raise error

    monkeypatch.setattr("fal.auth.key_credentials", lambda: ("id", "secret"))
    monkeypatch.setattr("fal.auth.current_user_info", lookup)


def _failed_lookup(status, body=b""):
    failure = FalServerlessException("Failed to fetch user info")
    failure.__cause__ = HTTPError("u", status, "x", {}, io.BytesIO(body))
    return failure


def test_a_key_that_cannot_read_its_account_is_unnamed(monkeypatch):
    _key_lookup(monkeypatch, _failed_lookup(403, b"Insufficient permissions"))

    assert key_account() == Account(None, False, via_key=True)


@pytest.mark.parametrize(
    "error", [_failed_lookup(500), _failed_lookup(403, b"nope"), URLError("offline")]
)
def test_other_failures_to_read_the_key_account_are_errors(monkeypatch, error):
    _key_lookup(monkeypatch, error)

    with pytest.raises(DevError, match="could not read the API key's account"):
        key_account()


@pytest.mark.parametrize("body", [{}, [], "acme"])
def test_a_malformed_key_account_reply_is_an_error(monkeypatch, body):
    monkeypatch.setattr("fal.auth.key_credentials", lambda: ("id", "secret"))
    monkeypatch.setattr("fal.auth.current_user_info", lambda headers: body)

    with pytest.raises(DevError, match="could not read the API key's account"):
        key_account()


def test_fal_is_given_no_team_under_a_key(monkeypatch):
    key = Account("acme", False, via_key=True)
    monkeypatch.setattr(team, "key_account", lambda: key)

    assert key.fal_team is None
    assert get_box(_args()).client.team is None
    assert RED.fal_team == "red"


def test_a_team_given_under_a_key_is_said_to_be_ignored(monkeypatch, capsys):
    monkeypatch.setattr(team, "key_account", lambda: Account("acme", False, True))

    get_box(_args(team="ACME"), quiet=True)
    assert capsys.readouterr().err == ""
    get_box(_args(team="red"), quiet=True)
    assert "--team red is ignored" in capsys.readouterr().err


def test_no_key_means_no_key_account(monkeypatch):
    monkeypatch.setattr("fal.auth.key_credentials", lambda: None)

    assert key_account() is None


def test_an_explicit_team_is_remembered_for_that_box_only():
    assert get_box(_args(name="a", team="red")).account == RED

    assert get_box(_args(name="a")).account == RED
    assert get_box(_args(name="b")).account == PERSONAL
    assert get_box(_args(name="a", host="other.example")).account == PERSONAL


def test_a_newer_team_replaces_the_remembered_one():
    get_box(_args(team="red"))
    get_box(_args(team="blue"))

    assert get_box(_args()).account.team == "blue"


def test_a_team_that_is_not_yours_is_refused_and_not_remembered():
    with pytest.raises(DevError, match="no team named 'rde'.*me, red, blue"):
        get_box(_args(team="rde"))

    assert get_box(_args()).account == PERSONAL


def test_a_remembered_team_that_is_gone_says_how_to_recover():
    get_box(_args(team="red"))
    state.update_box("devbox-demo", "api.example", team="gone")

    with pytest.raises(DevError, match="remembered for devbox-demo.*--team me"):
        get_box(_args())
    assert get_box(_args(team="me")).account == PERSONAL


def test_naming_your_personal_account_runs_as_personal():
    assert get_box(_args(team="ME")).account == PERSONAL


def test_the_team_line_goes_to_stderr(capsys):
    get_box(_args())
    get_box(_args(name="b", team="red"))

    out, err = capsys.readouterr()
    assert out == ""
    assert err.splitlines() == ["Team: personal (me)", "Team: red"]


def test_a_quiet_box_prints_no_team_line(capsys):
    get_box(_args(), quiet=True)

    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "account, text, runs_as",
    [
        (PERSONAL, "personal (me)", "your personal account"),
        (RED, "red", "team red"),
        (
            Account("acme", False, True),
            "acme (API key)",
            "the API key's account (acme)",
        ),
        (
            Account(None, False, True),
            "unnamed account (API key)",
            "the API key's account (unnamed)",
        ),
    ],
)
def test_account_description(account, text, runs_as):
    assert str(account) == text
    assert account.runs_as == runs_as


def test_production_apps_are_the_ones_that_are_not_dev_boxes():
    client = MagicMock()
    client.apps.list.return_value = [
        _app("devbox-demo"),
        _app("dev-flux-jane"),
        _app("flux"),
        _app("api"),
    ]

    assert team.production_apps(client) == ["api", "flux"]


def test_a_team_with_production_apps_must_be_confirmed(monkeypatch, capsys):
    _production(monkeypatch, ["flux", "api"])
    _stdin(monkeypatch, True)
    asks = []
    monkeypatch.setattr(team.Confirm, "ask", lambda *a, **k: asks.append(a) or True)

    _confirm(MagicMock())

    assert len(asks) == 1
    err = capsys.readouterr().err
    assert "team red has production apps (flux, api)" in err
    assert "demo (devbox-demo) would be deployed next to them" in err


def test_declining_the_confirmation_deploys_nothing(monkeypatch):
    _production(monkeypatch, ["flux"])
    _stdin(monkeypatch, True)
    monkeypatch.setattr(team.Confirm, "ask", lambda *a, **k: False)

    with pytest.raises(DevError, match="cancelled"):
        _confirm(MagicMock())
    assert state.load_box("devbox-demo", "api.example") == {}


def test_without_a_terminal_the_confirmation_needs_yes(monkeypatch):
    _production(monkeypatch, ["flux"])
    _stdin(monkeypatch, False)

    with pytest.raises(DevError) as info:
        _confirm(MagicMock())
    assert str(info.value) == (
        "team red has production apps (flux); "
        "pass --yes to deploy demo (devbox-demo) anyway"
    )


def test_a_failing_app_list_is_an_error_naming_the_account(monkeypatch):
    def fail(client):
        raise RuntimeError("boom")

    monkeypatch.setattr(team, "production_apps", fail)
    _stdin(monkeypatch, False)

    with pytest.raises(DevError, match="could not list the apps of team red: boom"):
        _confirm(MagicMock())


def test_an_unnamed_key_account_is_confirmed_once(monkeypatch):
    asked = _production(monkeypatch, ["flux"])
    _stdin(monkeypatch, True)
    monkeypatch.setattr(team.Confirm, "ask", lambda *a, **k: True)
    unnamed = Account(None, False, True)
    _confirm(MagicMock(), account=unnamed)
    asked.clear()

    _confirm(MagicMock(), account=unnamed)

    assert not asked


def test_an_api_key_account_is_checked_too(monkeypatch):
    asked = _production(monkeypatch, ["flux"])
    _stdin(monkeypatch, False)

    with pytest.raises(DevError, match="the API key's account"):
        _confirm(MagicMock(), account=Account(None, False, True))
    assert asked


def test_yes_skips_the_check(monkeypatch):
    asked = _production(monkeypatch, ["flux"])
    _stdin(monkeypatch, False)

    _confirm(MagicMock(), yes=True)

    assert not asked


def test_the_personal_account_is_not_checked(monkeypatch):
    asked = _production(monkeypatch, ["flux"])

    _confirm(MagicMock(), account=PERSONAL)

    assert not asked


def test_a_team_without_production_apps_needs_no_confirmation(monkeypatch):
    _production(monkeypatch, [])
    _stdin(monkeypatch, False)

    _confirm(MagicMock())


def test_a_confirmation_is_remembered_for_the_box_and_team(monkeypatch):
    asked = _production(monkeypatch, ["flux"])
    _stdin(monkeypatch, True)
    monkeypatch.setattr(team.Confirm, "ask", lambda *a, **k: True)
    _confirm(MagicMock())
    asked.clear()

    _confirm(MagicMock())
    assert not asked

    _confirm(MagicMock(), account=Account("blue", False))
    _confirm(MagicMock(), app="devbox-other")
    assert len(asked) == 2


def _fake_box(deployed):
    box = MagicMock(app="devbox-demo", host="api.example", account=RED)
    box.name = "demo"
    box.app_info.return_value = object() if deployed else None
    return box


@pytest.fixture
def events(monkeypatch):
    events = []
    monkeypatch.setattr(
        box_commands, "confirm_deploy", lambda *a, **k: events.append(("confirm", k))
    )
    monkeypatch.setattr(box_commands, "_cleanup_on_hangup", lambda: None)
    return events


def test_deploy_confirms_before_deploying(monkeypatch, events):
    box = _fake_box(deployed=True)
    box.deploy.side_effect = lambda **k: events.append("deploy") or "ok"
    monkeypatch.setattr(box_commands, "get_box", lambda args, **kw: box)

    box_commands.cmd_deploy(parse_args(["dev", "deploy", "--yes"]))

    assert events == [("confirm", {"yes": True, "name": "demo"}), "deploy"]


def test_up_confirms_only_when_it_has_to_deploy(monkeypatch, events):
    args = parse_args(["dev", "up"])

    monkeypatch.setattr(box_commands, "get_box", lambda args, **kw: _fake_box(True))
    box_commands.cmd_up(args)
    assert not events

    monkeypatch.setattr(box_commands, "get_box", lambda args, **kw: _fake_box(False))
    box_commands.cmd_up(args)
    assert events == [("confirm", {"yes": False, "name": "demo"})]


@pytest.mark.parametrize("command", ["deploy", "up", "app start x.py"])
def test_yes_is_a_flag_of_the_commands_that_deploy(command):
    assert parse_args(["dev", *command.split(), "--yes"]).yes
    assert not parse_args(["dev", *command.split()]).yes


def test_yes_is_not_a_flag_of_other_commands():
    with pytest.raises(FalParserExit):
        parse_args(["dev", "status", "--yes"])


def test_team_help_says_it_is_remembered_and_defaults_to_personal(capsys):
    with pytest.raises(FalParserExit):
        parse_args(["dev", "up", "--help"])

    out = " ".join(capsys.readouterr().out.split())
    assert re.search(
        r"--team TEAM .*Changes the team remembered for the box.*personal account", out
    )


def test_ssh_proxy_does_not_announce_the_account(monkeypatch):
    seen = {}
    box = MagicMock()
    box.ssh_proxy.return_value = 0
    monkeypatch.setattr(box_commands, "_cleanup_on_hangup", lambda: None)
    monkeypatch.setattr(
        box_commands, "get_box", lambda args, **kwargs: seen.update(kwargs) or box
    )

    box_commands.cmd_ssh_proxy(parse_args(["dev", "ssh-proxy"]))

    assert seen == {"quiet": True, "remember": False}


def test_app_commands_after_start_do_not_offer_team(capsys):
    for command, offered in (("app start x.py", True), ("app sync", False)):
        with pytest.raises(FalParserExit):
            parse_args(["dev", *command.split(), "--help"])
        assert ("--team" in capsys.readouterr().out) is offered


def test_a_session_remembers_its_account(monkeypatch, tmp_path):
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path)
    app_session.Session(
        name="dev-x",
        file_path="/a.py",
        root="/",
        module="a",
        class_name="A",
        run_url="u",
        token="t",
        account=RED,
    ).save()

    assert app_session.find_session("dev-x").account == RED


def test_a_session_from_before_accounts_is_ignored(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path)
    (tmp_path / "dev-x.json").write_text(
        json.dumps(
            {
                "name": "dev-x",
                "file_path": "/a.py",
                "root": "/",
                "module": "a",
                "class_name": "A",
                "run_url": "u",
                "token": "t",
                "team": None,
            }
        )
    )

    assert app_session.load_sessions() == []
    assert "written by an older `fal dev`" in capsys.readouterr().err
