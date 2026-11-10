import json

import pytest
from dev.app.fakes import FakeDepsBox, make_cli_session

from fal.cli.main import parse_args

pytestmark = pytest.mark.usefixtures("dev_account")


def test_status_json_hides_replay(tmp_path, monkeypatch, capsys):
    box = FakeDepsBox()
    make_cli_session(tmp_path, monkeypatch, box)
    box.state = {"packages": ["six==1.16.0"], "index_url": "https://u:s@x/simple"}
    args = parse_args(["dev", "app", "deps", "status", "--json"])
    args.func(args)
    out = capsys.readouterr().out
    assert "replay" not in json.loads(out)
    assert "u:s@" not in out


def test_deps_lines_are_plain_data(tmp_path, monkeypatch, capsys):
    box = FakeDepsBox()
    make_cli_session(tmp_path, monkeypatch, box)
    box.state = {"packages": ["six==1.16.0"]}
    args = parse_args(["dev", "app", "deps", "status"])
    args.func(args)
    assert "  six==1 (new)\n" in capsys.readouterr().out
