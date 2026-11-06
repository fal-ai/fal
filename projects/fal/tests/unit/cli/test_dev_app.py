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
