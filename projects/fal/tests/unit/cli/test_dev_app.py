import pytest

from fal.cli.main import parse_args
from fal.dev.app import deploy
from fal.dev.errors import DevError

pytestmark = pytest.mark.usefixtures("dev_account")


def test_parser_registers_commands():
    args = parse_args(
        ["dev", "app", "start", "app.py::Demo", "--machine-type", "GPU-H100"]
    )
    assert args.app_ref == ("app.py", "Demo")
    assert args.machine_type == "GPU-H100"
    args = parse_args(["dev", "app", "sync", "--watch", "--request", "/"])
    assert args.watch and args.request == "/"


def test_start_refuses_to_take_over_an_unknown_app(monkeypatch, tmp_path):
    from fal.dev.cli import app as dev_app

    (tmp_path / "app.py").write_text("import fal\n")
    monkeypatch.setattr(deploy, "box_exists", lambda *a, **k: True)
    monkeypatch.setattr(
        deploy,
        "find_session",
        lambda name: (_ for _ in ()).throw(DevError("none")),
    )
    deployed = []
    monkeypatch.setattr(deploy, "deploy_box", lambda *a, **k: deployed.append(1))
    args = parse_args(
        ["dev", "app", "start", f"{tmp_path / 'app.py'}::App", "--team", "t"]
    )
    with pytest.raises(DevError, match="--replace"):
        dev_app.cmd_start(args)
    assert not deployed


def test_start_help_says_synced_source_is_shared_with_the_team(capsys):
    from fal.cli.parser import FalParserExit

    with pytest.raises(FalParserExit):
        parse_args(["dev", "app", "start", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "shared /data" in out
    assert "Everyone on the team can read" in out


def test_dash_wait_defaults_to_the_terminal_and_can_be_forced():
    assert parse_args(["dev", "app", "dash"]).wait is None
    assert parse_args(["dev", "app", "dash", "--wait"]).wait is True
    assert parse_args(["dev", "app", "dash", "--no-wait"]).wait is False
    assert parse_args(["dev", "app", "split"]).name is None


def test_dash_without_wait_still_errors_when_no_box(monkeypatch, tmp_path):
    from fal.dev.app import session as app_session
    from fal.dev.cli import app as dev_app

    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path)
    with pytest.raises(DevError, match="No dev app session"):
        dev_app.cmd_dash(parse_args(["dev", "app", "dash", "--no-wait"]))


def test_dash_waits_then_attaches_and_waits_again_when_the_box_is_gone(
    monkeypatch,
):
    from dev.app.fakes import make_session

    from fal.dev.cli import app as dev_app
    from fal.dev.dash import tui, waiting

    sessions = iter([make_session(), make_session(name="dev-second")])
    waits, attached = [], []
    monkeypatch.setattr(
        waiting, "wait_for_session", lambda name, console: next(sessions)
    )
    monkeypatch.setattr(
        tui,
        "run_dashboard",
        lambda box, **k: attached.append(box.name) or len(attached) == 1,
    )
    # First box is stopped from elsewhere (gone), the second ends with q.
    dev_app.cmd_dash(parse_args(["dev", "app", "dash", "--wait"]))
    assert attached == ["dev-demo-me", "dev-second"] and not waits


def test_sync_records_what_the_dash_shows(monkeypatch, tmp_path):
    from dev.app.fakes import make_session

    from fal.dev.app import client, live
    from fal.dev.app import session as app_session
    from fal.dev.cli import app as dev_app

    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path)
    session = make_session()
    result = client.SyncResult(
        changed=["a.py", "b.py"],
        reload={"action": "reload", "ms": 80, "code_version": 3},
    )
    monkeypatch.setattr(client, "sync_files", lambda s, full=False: result)
    args = parse_args(["dev", "app", "sync"])
    dev_app._sync_once(session, args)
    last = live.read_sync(session.name)["last"]
    assert (last["files"], last["action"], last["ms"]) == (2, "reload", 80)
    # No changes: nothing new recorded.
    live.record_sync(session.name, files=9, action="x", ms=None, seconds=0)
    monkeypatch.setattr(client, "sync_files", lambda s, full=False: client.SyncResult())
    dev_app._sync_once(session, args)
    assert live.read_sync(session.name)["last"]["files"] == 9


def test_start_sets_the_deploying_phase_and_clears_it_on_a_failed_deploy(
    monkeypatch, tmp_path
):
    from fal.dev.app import live
    from fal.dev.app import session as app_session
    from fal.dev.cli import app as dev_app

    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path / "s")
    (tmp_path / "app.py").write_text("import fal\n")
    seen = []

    def failing(*a, **k):
        seen.append([b["phase"] for b in live.starting()])
        raise DevError("deploy failed")

    monkeypatch.setattr(deploy, "start_box", failing)
    monkeypatch.setattr(deploy, "box_exists", lambda *a, **k: False)
    monkeypatch.setattr(dev_app, "confirm_deploy", lambda *a, **k: None)
    args = parse_args(
        ["dev", "app", "start", f"{tmp_path / 'app.py'}::App", "--team", "t"]
    )
    with pytest.raises(DevError, match="deploy failed"):
        dev_app.cmd_start(args)
    assert seen == [["deploying"]] and live.starting() == []


def test_sync_watch_without_a_box_fails_fast_off_a_terminal(monkeypatch, tmp_path):
    from fal.dev.app import session as app_session
    from fal.dev.cli import app as dev_app

    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False, raising=False)
    with pytest.raises(DevError, match="No dev app session"):
        dev_app.cmd_sync(parse_args(["dev", "app", "sync", "--watch"]))
