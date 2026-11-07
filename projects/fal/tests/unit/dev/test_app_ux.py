"""`fal dev app` UX: the waiting dash, start phases, the sync row, split."""

import os
import subprocess
import sys
import time

import pytest
from rich.console import Console

from dev.app.fakes import make_session
from fal.dev.app import live, split
from fal.dev.app import session as app_session
from fal.dev.dash import app as app_dash
from fal.dev.dash import waiting
from fal.dev.dash.model import Snapshot
from fal.dev.errors import DevError

from .test_dash import NOW, _state
from .test_dash_app import _probe

pytestmark = pytest.mark.usefixtures("dev_account")


@pytest.fixture(autouse=True)
def sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path / "sessions")
    return tmp_path / "sessions"


def _text(renderable) -> str:
    console = Console(record=True, width=100, force_terminal=False, color_system=None)
    console.print(renderable)
    return console.export_text()


# --- waiting frame ---------------------------------------------------------


def test_waiting_frame_is_one_frame_with_the_start_hint():
    text = _text(waiting.frame([], NOW))
    assert "waiting for a dev box (start one with `fal dev app start`)" in text
    assert "error" not in text
    assert text.count("fal dev app") == 2  # title and the hint


def test_waiting_frame_shows_the_start_phase():
    starting = [{"name": "dev-llm", "phase": live.PROVISIONING, "since": NOW - 65}]
    text = _text(waiting.frame(starting, NOW))
    assert "starting dev-llm: provisioning" in text
    assert "waiting for a dev box" not in text


def test_error_is_a_line_in_the_same_frame():
    text = _text(waiting.frame([], NOW, error="Several sessions (a, b)"))
    assert "waiting for a dev box" in text and "Several sessions (a, b)" in text


def test_find_treats_no_session_as_waiting_not_an_error():
    assert waiting.find(None) == (None, None)
    session = make_session()
    session.save()
    found, error = waiting.find(None)
    assert found.name == session.name and error is None
    make_session(name="dev-other").save()
    found, error = waiting.find(None)
    assert found is None and "Several sessions" in error


def test_wait_attaches_when_a_box_appears_and_prints_once_off_a_tty():
    session = make_session()
    answers = iter([(None, None), (None, None), (None, None), (session, None)])
    console = Console(record=True, width=100, force_terminal=False)
    got = waiting.wait_for_session(
        None,
        console,
        find_session_fn=lambda name: next(answers),
        sleep=lambda s: None,
    )
    assert got is session
    # Three polls, one line: no stacking.
    assert console.export_text().count("waiting for a dev box") == 1


def test_wait_returns_at_once_when_a_box_exists():
    session = make_session()
    got = waiting.wait_for_session(
        None, Console(), find_session_fn=lambda n: (session, None)
    )
    assert got is session


# --- phases ----------------------------------------------------------------


def test_phase_goes_provisioning_then_setup_and_clears():
    live.set_phase("dev-a", live.DEPLOYING)
    (box,) = live.starting()
    assert (box["name"], box["phase"]) == ("dev-a", live.DEPLOYING)
    since = box["since"]
    live.set_phase("dev-a", live.PROVISIONING)
    live.set_phase("dev-a", live.PROVISIONING)
    assert live.starting()[0]["phase"] == live.PROVISIONING
    live.set_phase("dev-a", live.SETUP)
    assert live.starting()[0]["since"] >= since
    live.clear_phase("dev-a")
    assert live.starting() == []


def test_phase_of_a_dead_or_old_start_is_ignored(sessions_dir):
    live.set_phase("dev-a", live.SETUP)
    path = sessions_dir / "dev-a.phase"
    path.write_text(
        f'{{"phase": "setup", "since": {time.time()}, "pid": {2**22 + 12345}}}'
    )
    assert live.starting() == []
    path.write_text(
        f'{{"phase": "setup", "since": {time.time() - live.PHASE_MAX_AGE - 1}, '
        f'"pid": {os.getpid()}}}'
    )
    assert live.starting() == []
    path.write_text("not json")
    assert live.starting() == []


def test_clearing_on_error_drops_the_phase_only_on_failure():
    with live.clearing_on_error("dev-a"):
        live.set_phase("dev-a", live.DEPLOYING)
    assert live.starting()
    with pytest.raises(DevError), live.clearing_on_error("dev-a"):
        raise DevError("deploy failed")
    assert live.starting() == []


def test_phase_and_sync_files_are_not_sessions_and_delete_with_the_session():
    session = make_session()
    session.save()
    live.set_phase(session.name, live.SETUP)
    live.record_sync(session.name, files=1, action="reload", ms=5, seconds=0.1)
    assert [s.name for s in app_session.load_sessions()] == [session.name]
    session.delete()
    assert not list(app_session.SESSIONS_DIR.glob("*"))


def test_dash_shows_the_phase_of_a_starting_box():
    live.set_phase("dev-demo-me", live.SETUP)
    box = app_dash.AppBox(make_session())
    snapshot = Snapshot(state=_state("starting"), state_at=NOW)
    text = _text(app_dash.render(snapshot, box, time.time()))
    assert "starting: running setup() on the box" in text


# --- hot reload row --------------------------------------------------------


def _frame(sync=None) -> str:
    if sync:
        live.record_sync("dev-demo-me", **sync)
    box = app_dash.AppBox(make_session())
    snapshot = Snapshot(
        state=_state("running"), state_at=NOW, probe=_probe(), probe_at=NOW
    )
    return _text(app_dash.render(snapshot, box, NOW))


def test_sync_row_before_any_sync_says_not_watching():
    text = _frame()
    assert "hot reload" in text and "not watching: `fal dev app sync --watch`" in text
    assert "no sync yet" in text


def test_sync_row_shows_count_action_duration_and_watching():
    live.set_watching("dev-demo-me", True)
    text = _frame(dict(files=3, action="reload", ms=640, seconds=1.234))
    assert "watching" in text and "not watching" not in text
    assert "3 files, reloaded in place, 640 ms, synced in 1.2s" in text
    # The local record replaces the box's own "last sync" row.
    assert "reload: app.py" not in text


def test_sync_row_restart_and_failure():
    text = _frame(dict(files=1, action="restart", ms=None, seconds=2.0))
    assert "1 file, process restart, synced in 2.0s" in text
    text = _frame(dict(files=2, action="error", ms=None, seconds=0.5))
    assert "not reloaded (error)" in text


def test_dead_watcher_pid_is_not_watching(sessions_dir):
    live.set_watching("dev-demo-me", True)
    data = (sessions_dir / "dev-demo-me.sync").read_text()
    (sessions_dir / "dev-demo-me.sync").write_text(
        data.replace(str(os.getpid()), str(2**22 + 12345))
    )
    assert live.read_sync("dev-demo-me")["watching"] is False


def test_only_the_watcher_that_set_the_pid_clears_it(sessions_dir):
    live.set_watching("dev-demo-me", True)
    path = sessions_dir / "dev-demo-me.sync"
    path.write_text(path.read_text().replace(str(os.getpid()), "1"))
    live.set_watching("dev-demo-me", False)
    assert '"watch_pid": 1' in path.read_text()


# --- dash ends when the box is gone ---------------------------------------


def test_controls_report_a_box_that_was_stopped():
    session = make_session()
    session.save()
    controls = app_dash.controls(app_dash.AppBox(session))
    assert controls.gone() is False
    session.delete()
    assert controls.gone() is True


# --- split -----------------------------------------------------------------


def test_split_commands_carry_the_name():
    dash, sync = split.commands("dev-x")
    assert dash[1:] == ["-m", "fal", "dev", "app", "dash", "--name", "dev-x"]
    assert dash[0] == sys.executable
    assert sync[-3:] == ["--watch", "--name", "dev-x"]
    assert split.commands(None)[0][-3:] == ["dev", "app", "dash"]


def test_split_in_tmux_targets_the_new_dash_pane():
    calls = []

    def runner(cmd, check, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="%7\n")

    how = split.run(
        None, "/my repo", {"TMUX": "/tmp/tmux-1/default,1,0"}, runner=runner
    )
    assert how == "tmux" and len(calls) == 2
    assert calls[0][:3] == ["tmux", "split-window", "-h"]
    assert "-d" in calls[0] and "dev app dash" in calls[0][-1]
    assert "'/my repo'" in calls[0][-1]
    assert calls[1][calls[1].index("-t") + 1] == "%7"
    assert "sync --watch" in calls[1][-1]


def test_split_in_iterm_runs_applescript():
    calls = []
    how = split.run(
        "dev-x",
        "/r",
        {"TERM_PROGRAM": "iTerm.app"},
        runner=lambda cmd, check, **kw: calls.append(cmd),
    )
    assert how == "iterm"
    assert calls[0][:2] == ["osascript", "-e"]
    script = calls[0][2]
    assert (
        "split vertically" in script and "dev app sync --watch --name dev-x" in script
    )


def test_split_elsewhere_prints():
    assert split.run(None, "/r", {}, runner=pytest.fail) == "print"
    assert split.run(None, "/r", {"TERM_PROGRAM": "Apple_Terminal"}) == "print"


def test_split_failure_is_a_dev_error():
    def boom(cmd, check, **kw):
        raise subprocess.CalledProcessError(1, cmd)

    with pytest.raises(DevError, match="Could not open the panes"):
        split.run(None, "/r", {"TMUX": "x"}, runner=boom)


def test_a_refused_start_does_not_erase_another_starts_phase(sessions_dir):
    live.set_phase("dev-a", live.PROVISIONING)
    path = sessions_dir / "dev-a.phase"
    path.write_text(path.read_text().replace(str(os.getpid()), str(os.getppid())))
    live.clear_phase("dev-a")
    assert path.exists()


def test_a_watcher_waits_while_start_is_booting_the_box():
    session = make_session()
    session.save()
    live.set_phase(session.name, live.SETUP)
    assert waiting.find(None, settled=True) == (None, None)
    assert waiting.find(None)[0].name == session.name
    live.clear_phase(session.name)
    assert waiting.find(None, settled=True)[0].name == session.name


def test_missing_session_is_a_typed_waiting_state():
    from fal.dev.app.session import NoSession, find_session

    with pytest.raises(NoSession):
        find_session(None)
    assert waiting.find(None) == (None, None)


def test_a_second_start_cannot_take_over_a_live_phase_note(sessions_dir):
    live.set_phase("dev-a", live.PROVISIONING)
    path = sessions_dir / "dev-a.phase"
    # Another live process (the parent) owns the note.
    path.write_text(path.read_text().replace(str(os.getpid()), str(os.getppid())))
    live.set_phase("dev-a", live.DEPLOYING)
    with pytest.raises(DevError), live.clearing_on_error("dev-a"):
        raise DevError("refused")
    assert f'"pid": {os.getppid()}' in path.read_text()
    assert live.starting()[0]["phase"] == live.PROVISIONING


def test_a_named_box_that_does_not_exist_yet_is_waiting_not_an_error():
    assert waiting.find("dev-x") == (None, None)
    make_session(name="dev-other").save()
    assert waiting.find("dev-x") == (None, None)


def test_an_old_session_file_is_announced_once_not_every_poll(sessions_dir, capsys):
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "dev-old.json").write_text('{"unknown_field": 1}')
    app_session._WARNED.clear()
    for _ in range(3):
        app_session.load_sessions()
    assert capsys.readouterr().err.count("Ignoring") == 1


def test_clear_sync_forgets_the_previous_box():
    live.record_sync("dev-a", files=1, action="reload", ms=1, seconds=0.1)
    live.clear_sync("dev-a")
    assert live.read_sync("dev-a")["last"] is None


def test_probe_without_the_hot_section_still_draws_and_is_stored():
    from fal.dev.dash.model import waits

    snapshot = Snapshot(
        state=_state("running"), state_at=NOW, probe=_probe(), probe_at=NOW
    )
    assert "hot" not in snapshot.probe
    assert waits(snapshot) == set()


def test_a_missing_probe_field_blanks_one_panel_not_all():
    probe = _probe()
    del probe["gpus"][0]["util_pct"]
    box = app_dash.AppBox(make_session())
    snapshot = Snapshot(
        state=_state("running"), state_at=NOW, probe=probe, probe_at=NOW
    )
    text = _text(app_dash.render(snapshot, box, NOW))
    assert "probe has no" in text
    assert "scale to 0" in text and "code v2, pid 12" in text
    assert "waiting for the box" not in text


def test_session_lookup_is_per_user_not_per_folder(tmp_path, monkeypatch):
    make_session(root="/elsewhere").save()
    monkeypatch.chdir(tmp_path)
    assert app_session.find_session(None).name == "dev-demo-me"


def test_unknown_session_fields_are_dropped_and_unreadable_files_are_named(
    sessions_dir, capsys
):
    make_session().save()
    path = sessions_dir / "dev-demo-me.json"
    import json

    data = json.loads(path.read_text())
    path.write_text(json.dumps({**data, "future_field": 1}))
    assert app_session.find_session(None).name == "dev-demo-me"
    path.write_text(json.dumps({"name": "dev-demo-me"}))
    with pytest.raises(app_session.NoSession) as caught:
        app_session.find_session(None)
    message = str(caught.value)
    assert str(sessions_dir) in message and "dev-demo-me.json" in message


def test_app_dash_has_no_idle_watchdog_warning():
    from fal.dev.dash.model import IdleWarning
    from fal.dev.dash.tui import box_controls

    controls = app_dash.controls(app_dash.AppBox(make_session()))
    assert controls.idle_warning is False
    assert box_controls(object()).idle_warning is True
    # An app probe has no idle section, so the watchdog check cannot run on it.
    snapshot = Snapshot(
        state=_state("running"), state_at=NOW, probe=_probe(), probe_at=NOW
    )
    with pytest.raises(KeyError):
        IdleWarning().check(snapshot, NOW)
