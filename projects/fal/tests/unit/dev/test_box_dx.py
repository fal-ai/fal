"""Developer-experience behavior of the `fal dev` box commands."""

from __future__ import annotations

import io
import json
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from dev.fakes import make_box, make_runner
from dev.test_dash import NOW, FakeDashBox, _probe, _snapshot
from fal.cli.main import parse_args
from fal.dev import copy, execute, state
from fal.dev.box import install_ssh_config, ssh_config
from fal.dev.cli import box as cli
from fal.dev.cli import summary
from fal.dev.errors import DevError
from fal.dev.inventory import BoxRow
from fal.dev.steps import Steps
from fal.sdk import RunnerState


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "dev-state")


def _console():
    out = io.StringIO()
    return Console(file=out, width=200, force_terminal=False), out


# --- logs -------------------------------------------------------------------


def test_long_log_lines_collapse_unless_full():
    long = "x" * 900
    short = cli._collapse(long, False)
    assert short.startswith("x" * 200 + "...")
    assert "(900 chars, --full to show)" in short
    assert cli._collapse(long, True) == long
    assert cli._collapse("y" * 500, False) == "y" * 500


def test_latest_runner_prefers_a_live_one_over_a_pending_one(monkeypatch):
    pending = make_runner("pend", RunnerState.PENDING, uptime=1)
    live = make_runner("live", RunnerState.RUNNING, uptime=500)
    box = make_box()
    started = datetime(2026, 11, 12, 9, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(
        cli, "get_runner_info", lambda rest, rid: SimpleNamespace(started_at=started)
    )
    runner, at = cli._latest_runner(box, [pending, live])
    assert runner.runner_id == "live" and at == started


def test_latest_runner_falls_back_to_the_latest_start(monkeypatch):
    old = make_runner("old", RunnerState.DEAD)
    new = make_runner("new", RunnerState.DEAD)
    times = {
        "old": datetime(2026, 11, 12, 8, tzinfo=timezone.utc),
        "new": datetime(2026, 11, 12, 9, tzinfo=timezone.utc),
    }
    monkeypatch.setattr(
        cli,
        "get_runner_info",
        lambda rest, rid: SimpleNamespace(started_at=times[rid]),
    )
    runner, at = cli._latest_runner(make_box(), [old, new])
    assert runner.runner_id == "new" and at == times["new"]


def _logs_args(console, **fields):
    base = dict(all=True, tail=None, no_follow=False, full=False, console=console)
    return SimpleNamespace(**{**base, **fields})


def test_logs_all_names_the_runner_and_says_when_it_is_empty(monkeypatch):
    runner = make_runner("run-1", RunnerState.RUNNING)
    box = make_box([runner])
    box.runners = lambda since=None: [runner]
    monkeypatch.setattr(cli, "get_box", lambda args: box)
    monkeypatch.setattr(
        cli,
        "get_runner_info",
        lambda rest, rid: SimpleNamespace(
            started_at=datetime(2026, 11, 12, 9, 30, tzinfo=timezone.utc),
            ended_at=None,
        ),
    )
    monkeypatch.setattr(cli, "_iter_logs", lambda *a, **k: iter(()))
    console, out = _console()
    cli.cmd_logs(_logs_args(console))
    text = out.getvalue()
    assert "full log of run-1 (running, started " in text
    assert "no log lines from run-1 yet" in text


def test_logs_tail_no_follow_prints_history_and_does_not_stream(monkeypatch):
    runner = make_runner("run-1")
    box = make_box([runner])
    seen = {}
    box.logs = lambda rid, count, since: (
        seen.setdefault("count", count)
        and [
            {"timestamp": "2026-11-12T10:00:00Z", "message": "m" * 700, "level": "INFO"}
        ]
    )
    monkeypatch.setattr(cli, "get_box", lambda args: box)
    monkeypatch.setattr(
        cli, "_stream_logs", lambda *a, **k: pytest.fail("must not follow")
    )
    console, out = _console()
    cli.cmd_logs(_logs_args(console, all=False, tail=5, no_follow=True))
    assert seen["count"] == 5
    assert "700 chars, --full to show" in out.getvalue()
    assert "ctrl-c to stop" not in out.getvalue()


def test_logs_follow_header_says_how_to_stop(monkeypatch):
    runner = make_runner("run-1")
    box = make_box([runner])
    box.logs = lambda rid, count, since: []
    monkeypatch.setattr(cli, "get_box", lambda args: box)
    monkeypatch.setattr(cli, "_stream_logs", lambda *a, **k: iter(()))
    console, out = _console()
    cli.cmd_logs(_logs_args(console, all=False))
    assert "then following (ctrl-c to stop)" in out.getvalue()


def test_logs_flags_parse():
    args = parse_args(["dev", "logs", "-n", "50", "--no-follow", "--full", "--all"])
    assert (args.tail, args.no_follow, args.full, args.all) == (50, True, True, True)
    assert parse_args(["dev", "logs"]).tail is None


# --- hold -------------------------------------------------------------------


def _hold_args(console, **fields):
    return SimpleNamespace(
        off=False, hours=None, console=console, parser=MagicMock(), **fields
    )


def test_hold_success_says_until_when_and_how_to_clear(monkeypatch):
    monkeypatch.setattr(
        cli, "get_box", lambda args: SimpleNamespace(hold=lambda h: True)
    )
    console, out = _console()
    args = _hold_args(console)
    args.hours = 4.0
    cli.cmd_hold(args)
    text = out.getvalue()
    assert "idle shutdown held until " in text and "(4h)" in text
    assert "fal dev hold --off to clear" in text
    assert "next stops" not in text


@pytest.mark.parametrize(
    "text, hours", [("2", 2.0), ("0.5", 0.5), ("30m", 0.5), ("2h", 2.0), ("90M", 1.5)]
)
def test_hold_takes_hours_or_a_duration(text, hours):
    assert parse_args(["dev", "hold", text]).hours == pytest.approx(hours)


def test_hold_rejects_a_bad_duration_with_the_accepted_forms(capsys):
    with pytest.raises(BaseException):
        parse_args(["dev", "hold", "soon"])
    assert "use hours (2, 0.5) or 30m, 2h" in capsys.readouterr().err


def test_hold_over_the_cap_names_the_cap(monkeypatch):
    monkeypatch.setattr(cli, "get_box", lambda args: pytest.fail("no box needed"))
    console, _ = _console()
    args = _hold_args(console)
    args.hours = 500.0
    with pytest.raises(DevError, match="at most 168 hours"):
        cli.cmd_hold(args)


def test_a_negative_hold_is_rejected(monkeypatch):
    monkeypatch.setattr(cli, "get_box", lambda args: pytest.fail("no box needed"))
    console, _ = _console()
    args = _hold_args(console)
    args.hours = -1.0
    with pytest.raises(DevError, match="more than 0"):
        cli.cmd_hold(args)


def test_hold_help_has_examples_the_cap_and_what_a_stop_does():
    from fal.cli.parser import FalParserExit

    with pytest.raises(FalParserExit):
        parse_args(["dev", "hold", "--help"])


# --- down -------------------------------------------------------------------


class _DownBox:
    name = "demo"
    app = "devbox-demo"

    def __init__(self, saved):
        self.saved = saved

    def down(self, no_save):
        return None if no_save else self.saved


def _down(monkeypatch, saved, **fields):
    box = _DownBox(saved)
    monkeypatch.setattr(cli, "get_box", lambda args: box)
    monkeypatch.setattr(cli, "_confirm_down", lambda box, args: None)
    console, out = _console()
    args = SimpleNamespace(
        no_save=False, yes=True, name=None, console=console, host=None, **fields
    )
    return cli.cmd_down(args), out.getvalue()


def test_down_names_the_box_and_what_was_saved(monkeypatch):
    code, text = _down(monkeypatch, "devbox-save: saved /work (4.0K) in 1s")
    assert code is None
    assert "stopped demo" in text
    assert "/work saved (4.0K)" in text
    assert "start again: fal dev up" in text


def test_down_says_when_work_was_not_saved(monkeypatch):
    box = _DownBox(None)
    monkeypatch.setattr(cli, "get_box", lambda args: box)
    monkeypatch.setattr(cli, "_confirm_down", lambda box, args: None)
    console, out = _console()
    cli.cmd_down(
        SimpleNamespace(no_save=True, yes=True, name="demo", console=console, host=None)
    )
    assert "/work NOT saved (--no-save)" in out.getvalue()
    assert "start again: fal dev up --name demo" in out.getvalue()


def test_down_interrupted_exits_130(monkeypatch):
    class Box(_DownBox):
        def down(self, no_save):
            raise KeyboardInterrupt

    monkeypatch.setattr(cli, "get_box", lambda args: Box(None))
    monkeypatch.setattr(cli, "_confirm_down", lambda box, args: None)
    console, out = _console()
    args = SimpleNamespace(
        no_save=False, yes=True, name=None, console=console, host=None
    )
    assert cli.cmd_down(args) == 130
    assert "fal dev status" in out.getvalue()


# --- destroy ----------------------------------------------------------------


def test_destroy_confirmation_lists_delete_keep_and_recreate():
    box = SimpleNamespace(
        name="demo",
        app="devbox-demo",
        account=SimpleNamespace(runs_as="troy"),
    )
    lines = "\n".join(cli._destroy_summary(box, 1))
    assert "will delete: the app devbox-demo and 1 runner(s)" in lines
    assert "will keep:" in lines and "devbox-demo" in lines
    assert "snapshot" in lines and "cannot be deleted yet" in lines
    assert "recreate with: fal dev up --name demo" in lines


# --- up ---------------------------------------------------------------------


def test_ready_card_reports_total_and_runner_time():
    snap = _snapshot(_probe())

    def text(**kw):
        console = Console(record=True, width=110, force_terminal=False)
        console.print(summary.ready_card(snap, FakeDashBox(), now=NOW, **kw))
        return console.export_text()

    assert "ready in 2m21s (runner 1m37s)" in text(waited=97, total=141)
    assert "already running" in text(waited=2, total=2, already_running=True)
    assert "ready in 2s" not in text(waited=2, total=2, already_running=True)


def test_up_remembers_and_shows_the_last_start(monkeypatch):
    box = SimpleNamespace(app="devbox-demo")
    args = SimpleNamespace(host=None)
    assert cli._last_start_seconds(box, args) is None
    cli._record_use(box, args, 97.4)
    assert cli._last_start_seconds(box, args) == 97
    assert state.load_box("devbox-demo", None)["last_used_at"] <= time.time()


def test_a_step_note_replaces_the_typical_time():
    out = io.StringIO()
    steps = Steps(
        [("runner", "start runner", 100)],
        title="demo",
        team="t",
        machine="M",
        stream=out,
        animate=False,
    )
    steps.set_note("runner", "last start took 1m37s")
    steps.begin("runner")
    steps.close()
    assert "(last start took 1m37s)" in out.getvalue()
    assert "typically" not in out.getvalue()


def test_up_warns_about_flags_ignored_by_an_existing_box(monkeypatch):
    from fal.dev import guard

    box = make_box([make_runner("r1")])
    said = []
    box._say = said.append
    box.check_admin_key = lambda: None
    box.start = lambda *a, **k: make_runner("r1")
    monkeypatch.setattr(guard, "check_limits", lambda machine_type: None)
    box.up(60, None, secrets=["A"], set_env={"B": "1"})
    assert any(
        "--secret, --set-env ignored" in m and "fal dev deploy" in m for m in said
    )


def test_up_help_has_the_exit_codes_and_the_default_machine():
    from fal.cli.parser import FalParserExit

    for command in ("up", "down"):
        with pytest.raises(FalParserExit):
            parse_args(["dev", command, "--help"])


# --- status -----------------------------------------------------------------


def test_status_json_nulls_the_8_eib_sentinel():
    data = {"probe": {"disks": [{"total_bytes": 2**63, "used_bytes": 5}]}}
    out = cli._null_sentinels(data)
    assert out["probe"]["disks"][0] == {"total_bytes": None, "used_bytes": 5}
    assert data["probe"]["disks"][0]["total_bytes"] == 2**63


def test_uptime_comes_from_the_runner_start_time(monkeypatch):
    snap = _snapshot(_probe())
    snap.state.runner.state = RunnerState.RUNNING
    snap.state.runner.uptime = timedelta(seconds=27)
    started = datetime.fromtimestamp(NOW - 125, tz=timezone.utc)
    import fal.cli.runners as runners

    monkeypatch.setattr(
        runners,
        "_get_runner_info",
        lambda rest, rid: SimpleNamespace(started_at=started),
    )
    box = SimpleNamespace(client=MagicMock())
    summary._correct_uptime(box, snap)
    assert round(snap.state.runner.uptime.total_seconds()) == 125


def test_uptime_is_left_alone_when_the_start_time_cannot_be_read(monkeypatch):
    snap = _snapshot(_probe())
    snap.state.runner.state = RunnerState.RUNNING
    snap.state.runner.uptime = timedelta(seconds=27)
    summary._correct_uptime(SimpleNamespace(client=None), snap)
    assert snap.state.runner.uptime.total_seconds() == 27


def test_status_line_hides_your_own_owner_and_shows_someone_elses():
    class Box(FakeDashBox):
        def __init__(self, owner, me):
            super().__init__()
            self._owner, self._me = owner, me

        def owner(self):
            return self._owner

        def owner_identity(self):
            return self._me

    snap = _snapshot(_probe())
    assert "owner" not in summary.status_line(snap, Box("troy", "troy"), now=NOW).plain
    assert "owner ann" in summary.status_line(snap, Box("ann", "troy"), now=NOW).plain


# --- list -------------------------------------------------------------------


def _row(name, state_, **kw):
    return BoxRow(
        name=name,
        app=f"devbox-{name}",
        state=state_,
        machine_type="M",
        uptime_seconds=None,
        hold_until=None,
        hold_known=True,
        volume=f"devbox-{name}",
        **kw,
    )


def _list(monkeypatch, rows, mine=None, **fields):
    monkeypatch.setattr(cli.inventory, "list_boxes", lambda *a, **k: rows)
    monkeypatch.setattr(
        cli,
        "resolve_account",
        lambda *a, **k: SimpleNamespace(runs_as="troy", fal_team=None),
    )
    monkeypatch.setattr(cli, "SyncServerlessClient", MagicMock())
    monkeypatch.setattr(cli, "announce", lambda account, *a, **k: None)
    monkeypatch.setattr(cli, "remembered_name", lambda args: mine)
    console, out = _console()
    args = SimpleNamespace(
        team=None, host=None, json=False, all=False, console=console, **fields
    )
    cli.cmd_list(args)
    return out.getvalue()


def test_list_orders_running_first_then_recent_and_summarizes(monkeypatch):
    rows = [_row("a", "STOPPED"), _row("b", "STOPPED"), _row("c", "RUNNING")]
    state.update_box("devbox-b", None, last_used_at=time.time())
    state.update_box("devbox-a", None, last_used_at=time.time() - 1000)
    ordered = cli._ordered(rows, None)
    assert [r.name for r in ordered] == ["c", "b", "a"]
    text = _list(monkeypatch, rows)
    assert "1 running, 2 stopped" in text


def test_list_caps_stopped_boxes_but_keeps_this_folders(monkeypatch):
    rows = [_row("run", "RUNNING")] + [_row(f"s{i:02d}", "STOPPED") for i in range(15)]
    text = _list(monkeypatch, rows, mine="s14")
    assert "+4 stopped (fal dev list --all)" in text
    assert "s14  <- this folder" in text
    assert "1 running, 15 stopped" in text
    full = _list(monkeypatch, rows, mine="s14", **{})
    assert "s14" in full


def test_list_all_shows_every_box(monkeypatch):
    rows = [_row(f"s{i:02d}", "STOPPED") for i in range(15)]
    monkeypatch.setattr(cli.inventory, "list_boxes", lambda *a, **k: rows)
    monkeypatch.setattr(
        cli,
        "resolve_account",
        lambda *a, **k: SimpleNamespace(runs_as="t", fal_team=None),
    )
    monkeypatch.setattr(cli, "SyncServerlessClient", MagicMock())
    monkeypatch.setattr(cli, "announce", lambda account, *a, **k: None)
    monkeypatch.setattr(cli, "remembered_name", lambda args: None)
    console, out = _console()
    cli.cmd_list(
        SimpleNamespace(team=None, host=None, json=False, all=True, console=console)
    )
    assert "s14" in out.getvalue() and "fal dev list --all" not in out.getvalue()


def test_list_team_help_does_not_claim_it_is_remembered():
    from fal.cli.parser import FalParserExit

    assert parse_args(["dev", "list", "--all"]).all is True
    with pytest.raises(FalParserExit):
        parse_args(["dev", "list", "--help"])


# --- cp ---------------------------------------------------------------------


def _cp(monkeypatch, argv, tmp_path):
    monkeypatch.setattr(copy, "find_scp", lambda: "scp")
    monkeypatch.setattr(cli, "_run_scp", lambda argv: 0)
    monkeypatch.setattr(cli, "_proxy_fal", lambda: ("fal", None))
    console, out = _console()
    args = parse_args(["dev", "cp", *argv, "--name", "b"])
    args.console = console
    cli.cmd_cp(args)
    return out.getvalue()


def test_cp_says_it_is_connecting_and_ends_the_same_way_both_ways(
    monkeypatch, tmp_path
):
    (tmp_path / "f").write_text("x" * 21)
    up = _cp(monkeypatch, [str(tmp_path / "f"), ":/work/x"], tmp_path)
    assert up.index("connecting to the box") < up.index("uploaded")
    assert "uploaded 21 B to :/work/x (" in up
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "x").write_text("y" * 21)
    down = _cp(monkeypatch, [":/work/x", str(dest / "x")], tmp_path)
    assert f"downloaded 21 B to {dest / 'x'} (" in down


def test_cp_recursive_ends_like_a_single_file(monkeypatch, tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "a").write_text("1234")
    text = _cp(monkeypatch, ["-r", str(folder), ":/data/x"], tmp_path)
    assert "uploaded 4 B to :/data/x (" in text


def test_uploaded_path_follows_a_trailing_slash():
    plan = copy.plan("./a.txt", ":/work/dir/")
    assert copy.uploaded_path(plan) == "/work/dir/a.txt"
    assert copy.uploaded_path(copy.plan("./a.txt", ":/work/x")) == "/work/x"


# --- exec / shell -----------------------------------------------------------


def test_exit_127_note_depends_on_the_machine():
    assert "this is a CPU box" in execute.not_found_note("M")
    assert execute.not_found_note("GPU-H100") == "(command not found on the box)"
    assert execute.not_found_note(None) == "(command not found on the box)"


def test_exec_adds_the_note_on_127(monkeypatch, capsys):
    box = SimpleNamespace(
        client=None,
        require_runner=lambda: "r1",
        state=lambda: SimpleNamespace(machine_type="M"),
    )
    monkeypatch.setattr(cli, "get_box", lambda args: box)
    monkeypatch.setattr(cli, "warn_if_foreign", lambda box, args: None)
    monkeypatch.setattr(execute, "run", lambda *a, **k: 127)
    args = SimpleNamespace(
        command=["--", "nvidia-smi"], parser=MagicMock(), console=None
    )
    assert cli.cmd_exec(args) == 127
    assert "command not found on the box; this is a CPU box" in capsys.readouterr().err


def test_exec_stdin_flag_reaches_the_session(monkeypatch):
    seen = {}

    def session(client, runner_id, command, **kwargs):
        seen.update(kwargs)
        seen["command"] = command
        if kwargs["interactive"]:
            seen["piped"] = cli.sys.stdin.read()
        return 0

    from fal.cli import runners

    monkeypatch.setattr(runners, "shell_session", session)
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"hi\n")))
    execute.run(
        None,
        "r1",
        ["true"],
        MagicMock(),
        stdout=io.BytesIO(),
        stderr=io.BytesIO(),
        stdin=True,
    )
    assert seen["interactive"] is True
    # The box reads exactly the piped bytes, since the stream never signals EOF.
    assert 'head -c 3 | "$@"' in seen["command"][2]
    assert seen["piped"] == "hi\n"
    execute.run(
        None, "r1", ["true"], MagicMock(), stdout=io.BytesIO(), stderr=io.BytesIO()
    )
    assert seen["interactive"] is False


def test_exec_stdin_over_the_cap_points_at_cp(monkeypatch):
    monkeypatch.setattr(execute, "MAX_STDIN_BYTES", 4)
    monkeypatch.setattr(
        cli.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"too long"))
    )
    with pytest.raises(DevError) as caught:
        execute.run(None, "r1", ["cat"], MagicMock(), stdin=True)
    assert caught.value.fixes[0][0].startswith("fal dev cp")


def test_exec_stdin_needs_piped_input(monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)
    args = SimpleNamespace(command=["cat"], stdin=True, parser=MagicMock())
    with pytest.raises(DevError, match="nothing is piped in") as caught:
        cli.cmd_exec(args)
    assert ("fal dev shell", "an interactive session") in caught.value.fixes
    assert caught.value.code == 2


def test_a_stopped_box_says_so_for_exec_and_shell():
    with pytest.raises(DevError) as caught:
        make_box([]).require_runner()
    assert str(caught.value).endswith(" is stopped")
    assert caught.value.fixes[0][0].startswith("fal dev up --name ")


# --- ssh-config -------------------------------------------------------------


def test_ssh_config_is_compact_and_starts_with_the_host_line():
    entry = ssh_config("demo", "fal")
    assert entry.startswith("Host devbox-demo")
    assert sum(line.strip().startswith("#") for line in entry.splitlines()) == 2
    assert "—" not in entry
    noted = ssh_config("demo", "/abs/fal", note="absolute path: why")
    assert "# absolute path: why" in noted
    assert sum(line.strip().startswith("#") for line in noted.splitlines()) == 3


def test_proxy_command_uses_bare_fal_only_for_the_same_install(monkeypatch, tmp_path):
    real = tmp_path / "venv" / "bin" / "fal"
    real.parent.mkdir(parents=True)
    real.write_text("")
    link = tmp_path / "bin" / "fal"
    link.parent.mkdir()
    link.symlink_to(real)
    monkeypatch.setattr(cli, "_fal_command", lambda: str(real))
    monkeypatch.setattr(cli.shutil, "which", lambda name: str(link))
    assert cli._proxy_fal() == ("fal", None)
    other = tmp_path / "other" / "fal"
    other.parent.mkdir()
    other.write_text("")
    monkeypatch.setattr(cli.shutil, "which", lambda name: str(other))
    command, note = cli._proxy_fal()
    assert command == str(real) and "different install" in note
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert "not on PATH" in cli._proxy_fal()[1]


def test_install_refuses_a_half_block_and_leaves_the_file_alone(tmp_path):
    path = tmp_path / "config"
    original = (
        "# >>> fal dev demo >>>\nHost devbox-demo\n\nHost important\n    User me\n"
    )
    path.write_text(original)

    with pytest.raises(DevError, match="only one of the fal dev markers"):
        install_ssh_config(path, "demo", ssh_config("demo", "fal"))
    assert path.read_text() == original


def test_install_adds_replaces_and_is_idempotent(tmp_path):
    path = tmp_path / ".ssh" / "config"
    first = ssh_config("demo", "fal")
    assert install_ssh_config(path, "demo", first) == ("added", None)
    assert path.read_text().startswith("# >>> fal dev demo >>>\nHost devbox-demo")
    assert install_ssh_config(path, "demo", first) == ("unchanged", None)

    path.write_text("Host other\n    User me\n\n" + path.read_text())
    second = ssh_config("demo", "/x/fal")
    action, backup = install_ssh_config(path, "demo", second)
    assert action == "updated" and backup is not None
    text = path.read_text()
    assert text.count("# >>> fal dev demo >>>") == 1
    assert text.startswith("Host other\n    User me\n")
    assert "/x/fal dev ssh-proxy" in text and "ProxyCommand fal dev" not in text
    assert "ProxyCommand fal dev" in backup.read_text()


def test_install_appends_to_a_file_without_a_trailing_newline(tmp_path):
    path = tmp_path / "config"
    path.write_text("Host other\n    User me")
    action, backup = install_ssh_config(path, "demo", ssh_config("demo", "fal"))
    assert action == "added" and backup.read_text() == "Host other\n    User me"
    assert path.read_text().startswith("Host other\n    User me\n\n# >>> fal dev demo")


def test_install_flag_parses_and_cmd_installs(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(cli, "_proxy_fal", lambda: ("fal", None))
    console, out = _console()
    args = parse_args(["dev", "ssh-config", "--install", "--name", "demo"])
    args.console = console
    cli.cmd_ssh_config(args)
    assert "# >>> fal dev demo >>>" in (tmp_path / ".ssh" / "config").read_text()
    assert "connect with: ssh devbox-demo" in out.getvalue()
    json.dumps({})
