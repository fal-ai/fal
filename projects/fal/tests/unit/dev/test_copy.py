from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fal.cli.main import parse_args
from fal.cli.parser import FalParserExit
from fal.dev import copy
from fal.dev.cli import box as box_commands
from fal.dev.errors import DevError


def test_plan_download_and_upload():
    down = copy.plan(":/work/out.mp4", ".")
    assert (down.upload, down.local, down.remote) == (False, ".", "/work/out.mp4")
    up = copy.plan("./data", ":/data/x")
    assert (up.upload, up.local, up.remote) == (True, "./data", "/data/x")


@pytest.mark.parametrize("src,dst", [("a", "b"), (":a", ":b"), (":", "."), (".", ":")])
def test_plan_rejects_bad_sides(src, dst):
    with pytest.raises(DevError):
        copy.plan(src, dst)


def test_scp_command_never_uses_the_users_known_hosts():
    argv = copy.scp_command(
        copy.plan("./d", ":/x"), "dev-box", "/tmp/cfg", True, "/usr/bin/scp"
    )

    assert argv[:3] == ["/usr/bin/scp", "-F", "/tmp/cfg"]
    assert "UserKnownHostsFile=/dev/null" in argv
    assert "StrictHostKeyChecking=no" in argv
    assert "ControlPath=none" in argv
    assert "-r" in argv
    assert argv[-3:] == ["--", "./d", "dev-box:/x"]


def test_scp_command_download_puts_the_box_first():
    argv = copy.scp_command(copy.plan(":/a", "."), "dev-box", "/tmp/cfg", False, "scp")

    assert "-r" not in argv
    assert argv[-3:] == ["--", "dev-box:/a", "."]


def test_check_local_wants_r_for_folders(tmp_path):
    with pytest.raises(DevError):
        copy.check_local(copy.plan(str(tmp_path), ":/x"), False)
    copy.check_local(copy.plan(str(tmp_path), ":/x"), True)
    with pytest.raises(DevError):
        copy.check_local(copy.plan(str(tmp_path / "missing"), ":/x"), True)


def test_tree_size_and_human_size(tmp_path):
    (tmp_path / "a").write_bytes(b"x" * 10)
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "b").write_bytes(b"x" * 5)

    assert copy.tree_size(str(tmp_path)) == 15
    assert copy.tree_size(str(tmp_path / "a")) == 10
    assert copy.human_size(15) == "15 B"
    assert copy.human_size(3 * 1024 * 1024) == "3.0 MB"


def test_downloaded_path(tmp_path):
    assert copy.downloaded_path(copy.plan(":/w/o.mp4", str(tmp_path))) == str(
        tmp_path / "o.mp4"
    )
    assert copy.downloaded_path(copy.plan(":/w/o.mp4", "x.mp4")) == "x.mp4"


def test_downloaded_path_into_a_new_folder_is_the_folder_itself(tmp_path):
    new = str(tmp_path / "new")
    planned = copy.plan(":/data/x", new)
    (tmp_path / "new").mkdir()  # scp -r creates it

    assert copy.downloaded_path(planned) == new


def test_downloaded_path_unknown_for_globs_and_roots(tmp_path):
    for remote in (":/w/*.mp4", ":/", ":."):
        assert copy.downloaded_path(copy.plan(remote, str(tmp_path))) is None


def test_a_local_colon_is_not_read_by_scp_as_a_host():
    assert copy.plan("notes:v2.txt", ":/x").local == "./notes:v2.txt"
    assert copy.plan(":/a", "out:1").local == "./out:1"
    assert copy.plan("/tmp/a:b", ":/x").local == "/tmp/a:b"
    assert copy.plan("./a:b", ":/x").local == "./a:b"


def test_cmd_cp_runs_scp_with_the_ssh_config(monkeypatch, tmp_path):
    seen = {}

    def fake_call(argv):
        seen["argv"] = argv
        seen["config"] = open(argv[2]).read()
        return 0

    monkeypatch.setattr(copy, "find_scp", lambda: "scp")
    monkeypatch.setattr(box_commands, "_run_scp", fake_call)
    monkeypatch.setattr(box_commands, "_fal_command", lambda: "/bin/fal")
    (tmp_path / "f").write_text("hi")
    args = parse_args(["dev", "cp", str(tmp_path / "f"), ":/data/f", "--name", "b"])
    args.console = MagicMock()

    box_commands.cmd_cp(args)

    assert seen["argv"][-1].endswith(":/data/f")
    assert "ProxyCommand /bin/fal dev ssh-proxy --name b" in seen["config"]


def test_cmd_cp_reports_scp_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(copy, "find_scp", lambda: "scp")
    monkeypatch.setattr(box_commands, "_run_scp", lambda argv: 1)
    args = parse_args(["dev", "cp", ":/a", str(tmp_path), "--name", "b"])
    args.console = MagicMock()

    with pytest.raises(DevError, match="exit 1"):
        box_commands.cmd_cp(args)


def test_run_scp_terminates_the_child_when_interrupted(monkeypatch):
    proc = MagicMock()
    proc.wait.side_effect = [KeyboardInterrupt, 0]
    monkeypatch.setattr(box_commands.subprocess, "Popen", lambda argv: proc)

    with pytest.raises(KeyboardInterrupt):
        box_commands._run_scp(["scp"])

    proc.terminate.assert_called_once()


def test_cp_help_explains_the_colon(capsys):
    with pytest.raises(FalParserExit):
        parse_args(["dev", "cp", "--help"])

    out = capsys.readouterr().out
    assert "fal dev cp :/work/out.mp4 ." in out
    assert "known_hosts" in out
