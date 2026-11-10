import base64
import io
import tarfile
from types import SimpleNamespace

import pytest

from fal.dev.errors import DevError
from fal.dev.remote import process
from fal.dev.remote.process import ChildProcess, LogBuffer

pytestmark = pytest.mark.usefixtures("dev_account")


def test_child_process_reload_states(tmp_path):
    child = ChildProcess(tmp_path, LogBuffer())
    assert child.start() is False
    assert child.reload([])["action"] == "not_running"
    child.phase = "setup"
    assert child.reload([])["action"] == "pending"
    assert child.pending_reload


def test_child_process_ignores_stale_restart(tmp_path):
    child = ChildProcess(tmp_path, LogBuffer())
    child.module, child.class_name = "app", "App"
    stale = child.generation
    child.stop()
    assert child.start(generation=stale) is False


def test_log_buffer_returns_what_came_after_a_line_number(capsys):
    buffer = LogBuffer()
    for number in range(5):
        buffer.emit(f"line {number}")
    assert buffer.since(None, 2) == {"lines": ["line 3", "line 4"], "total": 5}
    assert buffer.since(None, 0)["lines"] == []
    assert buffer.since(3, 10)["lines"] == ["line 3", "line 4"]
    # A client ahead of the box (a new runner) starts from the last lines.
    assert buffer.since(99, 1)["lines"] == ["line 4"]


def test_child_layout_is_passed_to_the_app_process(tmp_path, monkeypatch):
    child = ChildProcess(tmp_path, LogBuffer())
    # No background launch or log pump threads.
    monkeypatch.setattr(
        process.threading, "Thread", lambda **kwargs: SimpleNamespace(start=dict)
    )
    with pytest.raises(ValueError, match="refusing"):
        child.start("m", "C", layout={"paths": ["../outside"], "workdir": "."})

    launched = []
    monkeypatch.setattr(child, "_install_sdk_deps", lambda generation: None)
    monkeypatch.setattr(child, "_wait_ready", lambda proc: None)
    monkeypatch.setattr(
        process.subprocess,
        "Popen",
        lambda command, **kwargs: launched.append((command, kwargs)),
    )
    child.module, child.class_name = "pkg.models.app", "Demo"
    child.paths, child.workdir = [".", "pkg/models"], "pkg/models"
    child._spawn(child.generation)
    command, _ = launched[0]
    assert "--cwd=pkg/models" in command
    assert [arg for arg in command if arg.startswith("--path=")] == [
        "--path=.",
        "--path=pkg/models",
    ]


def test_app_process_gets_the_session_name(tmp_path, monkeypatch):
    child = ChildProcess(tmp_path / "dev-tiny-image-me", LogBuffer())
    assert child._env()["FAL_DEV_SESSION"] == "dev-tiny-image-me"

    # Apps that read another name for it get that one too.
    monkeypatch.setattr(
        process.threading, "Thread", lambda **kwargs: SimpleNamespace(start=dict)
    )
    child.start("m", "C", session_env=["MY_CACHE_NAMESPACE", "FAL_MY_CACHE_NS"])
    assert child._env()["MY_CACHE_NAMESPACE"] == "dev-tiny-image-me"
    assert child._env()["FAL_MY_CACHE_NS"] == "dev-tiny-image-me"
    for bad in (
        "PYTHONPATH",
        "LD_PRELOAD",
        "FAL_DEV_APP",
        "FAL_DEV_SESSION",
        "FAL_KEY",
        "FAL_HOST",
        "FAL_RUN_HOST",
        "PATH",
        "home",
        "1BAD",
        "A-B",
    ):
        with pytest.raises(DevError, match="cannot set"):
            child.start("m", "C", session_env=[bad])
    assert child.session_env == ["MY_CACHE_NAMESPACE", "FAL_MY_CACHE_NS"]


def test_a_bad_sdk_archive_leaves_the_old_sdk_in_place(tmp_path):
    child = ChildProcess(tmp_path, LogBuffer())
    child.sdk.mkdir(parents=True)
    (child.sdk / "marker").write_text("old")
    with pytest.raises(Exception):
        child.replace_sdk(base64.b64encode(b"not a tarball").decode())
    assert (child.sdk / "marker").read_text() == "old"
    assert not (tmp_path / "sdk.new").exists()


def test_layout_paths_must_be_a_list_of_strings(tmp_path):
    child = ChildProcess(tmp_path, LogBuffer())
    with pytest.raises(ValueError, match="list of strings"):
        child.check_start({"paths": "pkg"}, None)


def test_replace_sdk_uses_a_unique_temp_dir_and_cleans_up(tmp_path, monkeypatch):
    child = ChildProcess(tmp_path, LogBuffer())
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("fal/x.py")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    blob = base64.b64encode(buffer.getvalue()).decode()
    seen = []
    real = process.tempfile.mkdtemp

    def mkdtemp(**kwargs):
        seen.append(real(**kwargs))
        return seen[-1]

    monkeypatch.setattr(process.tempfile, "mkdtemp", mkdtemp)
    child.replace_sdk(blob)
    child.replace_sdk(blob)
    assert len(set(seen)) == 2
    assert (child.sdk / "fal" / "x.py").exists()
    assert [p.name for p in tmp_path.glob("sdk.new-*")] == []


def test_failed_reload_during_setup_is_not_reported_ready(tmp_path, monkeypatch):
    child = ChildProcess(tmp_path, LogBuffer())
    proc = SimpleNamespace(poll=lambda: None)
    child.proc = proc
    child.pending_reload = True

    def fake_call(action, payload, timeout):
        if action == "status":
            return 200, {}
        return 200, {"action": "error", "reason": "SyntaxError in app.py"}

    monkeypatch.setattr(process, "child_call", fake_call)
    child._wait_ready(proc)
    assert child.phase == "error"
    assert "SyntaxError" in child.detail
    assert child.ready_at is None

    # A later successful reload clears it.
    monkeypatch.setattr(
        process, "child_call", lambda *a, **k: (200, {"action": "reload"})
    )
    assert child.reload(["app.py"])["action"] == "reload"
    assert child.phase == "ready"
    assert child.ready_at is not None


def test_fix_synced_during_a_failing_deferred_reload_is_not_dropped(
    tmp_path, monkeypatch
):
    child = ChildProcess(tmp_path, LogBuffer())
    proc = SimpleNamespace(poll=lambda: None)
    child.proc = proc
    child.pending_reload = True
    child.phase = "setup"
    reloads = []

    def fake_call(action, payload, timeout):
        if action == "status":
            return 200, {}
        reloads.append(1)
        if len(reloads) == 1:
            child.reload(["app.py"])  # the user syncs a fix mid-reload
            return 200, {"action": "error", "reason": "SyntaxError"}
        return 200, {"action": "reload"}

    monkeypatch.setattr(process, "child_call", fake_call)
    child._wait_ready(proc)
    assert len(reloads) == 2
    assert child.phase == "ready"
    assert child.ready_at is not None
