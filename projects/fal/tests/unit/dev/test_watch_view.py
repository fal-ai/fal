import json
from pathlib import Path

import pytest
from rich.console import Console

from fal.dev import hot
from fal.dev.cli import error_panel
from fal.dev.cli.watch_view import LoopView, exception_line, failing_line
from fal.dev.errors import DevError
from fal.dev.hot import RunResult

TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/work/klein/hot_server.py", line 183, in run\n'
    "    result = self.infer(self.state, **params)\n"
    '  File "/work/klein/infer.py", line 14, in infer\n'
    "    image = sepia(image)\n"
    "NameError: name 'sepia' is not defined\n"
)
IMAGE = {
    "content-type": "image/png",
    "x-hot-code-version": "4",
    "x-hot-infer-ms": "571",
}


def _console() -> Console:
    return Console(record=True, width=100, force_terminal=True, color_system=None)


def _view(console, **kwargs) -> LoopView:
    status = {"code_version": 3, "load_seconds": 23.4, "ready": True}
    return LoopView(console, status, **kwargs)


def _failure(status=422) -> RunResult:
    body = json.dumps({"error": "infer() raised", "traceback": TRACEBACK}).encode()
    return RunResult(status, {}, body, 0.2)


def test_failing_line_points_at_your_code():
    assert failing_line(TRACEBACK) == ("infer.py:14", "image = sepia(image)")
    assert failing_line(None) is None
    assert exception_line(TRACEBACK) == "NameError: name 'sepia' is not defined"


def test_header_card_says_what_is_watched():
    console = _console()
    _view(console).watching(Path("klein"), run=True)
    text = console.export_text()
    assert "watching klein" in text and "code v3" in text
    assert "model loaded in 23.4s" in text
    assert "sync it and run a request" in text and "the box keeps running" in text


def test_ok_run_prints_one_line_and_draws_the_image():
    console = _console()
    seen = []
    view = _view(console, on_image=lambda path, headers: seen.append(path))
    result = RunResult(200, {**IMAGE, "x-hot-reload-ms": "12"}, b"", 1.0)
    result.path = Path("hot-out/1.png")
    view.ran(result, edited_at=view.clock() - 1.5)
    text = console.export_text()
    assert "ok" in text and "v4" in text and "infer 571 ms" in text
    assert "reload 12 ms" in text and "hot-out/1.png" in text
    assert seen == [Path("hot-out/1.png")]


def test_error_panel_then_fixed_on_the_next_good_save():
    console = _console()
    view = _view(console)
    view.ran(_failure(), edited_at=view.clock())
    text = console.export_text()
    assert "NameError: name 'sepia' is not defined" in text
    assert "infer.py:14  image = sepia(image)" in text
    assert "the model is still loaded" in text
    ok = RunResult(200, {**IMAGE, "content-type": "application/json"}, b"{}", 1.0)
    view.ran(ok, edited_at=view.clock())
    assert "fixed" in console.export_text()
    view.ran(ok, edited_at=view.clock())
    assert "fixed" not in console.export_text()


def test_a_crashed_server_does_not_claim_the_model_is_loaded():
    console = _console()
    _view(console).ran(_failure(status=0), edited_at=0)
    assert "still loaded" not in console.export_text()


def test_restart_line_names_the_reason():
    console = _console()
    _view(console).restarting("load.py changed")
    assert "FULL RESTART (load.py changed)" in console.export_text()


def test_error_panel_on_a_terminal_and_plain_otherwise():
    exc = DevError(
        "the box restarted since this command began",
        title="the box restarted",
        fixes=[("fal dev up", "if it is not running")],
    )
    console = _console()
    error_panel.show(exc, console)
    text = console.export_text()
    assert "error  the box restarted" in text.splitlines()[0]
    assert "try  fal dev up  if it is not running" in text
    plain = Console(record=True, width=100, force_terminal=False)
    error_panel.show(exc, plain)
    assert plain.export_text().splitlines() == [
        "Error: the box restarted since this command began",
        "  try: fal dev up  (if it is not running)",
    ]


def _hot_box(monkeypatch, answers):
    box = hot.HotBox.__new__(hot.HotBox)
    calls = iter(answers)
    monkeypatch.setattr(box, "post", lambda *a, **k: next(calls))
    return box


def test_run_rides_out_a_restart_and_saves_the_image(tmp_path, monkeypatch):
    box = _hot_box(
        monkeypatch,
        [(409, {}, b'{"restarting": "load.py changed"}'), (200, IMAGE, b"PNG")],
    )
    monkeypatch.setattr(box, "wait_ready", lambda **k: {"load_seconds": 3})
    reasons = []
    result = box.run({}, str(tmp_path / "o.png"), on_restart=reasons.append)
    assert result.ok and result.restarted == "load.py changed"
    assert reasons == ["load.py changed"]
    assert result.path.read_bytes() == b"PNG"


def test_watch_pushes_a_save_runs_it_and_reports(tmp_path, monkeypatch):
    (tmp_path / "infer.py").write_text("v1")
    box = hot.HotBox.__new__(hot.HotBox)
    events = []

    class View:
        def watching(self, root, run):
            events.append(("watching", run))

        def synced(self, files, after_save):
            events.append(("synced", files))

        def ran(self, result, edited_at):
            events.append(("ran", result.ok))
            raise KeyboardInterrupt

        def sync_failed(self, files, error):  # pragma: no cover
            events.append(("sync_failed", error))

        def restarting(self, reason):  # pragma: no cover
            events.append(("restarting", reason))

        def lost(self, error):  # pragma: no cover
            events.append(("lost", error))

    sleeps = iter(range(10))

    def sleep(_):
        if next(sleeps) == 0:
            (tmp_path / "infer.py").write_text("v2")
            import os

            os.utime(tmp_path / "infer.py", (1e9, 1e9))

    monkeypatch.setattr(hot.time, "sleep", sleep)
    monkeypatch.setattr(box, "push", lambda root, remote, rels: ["infer.py"])
    monkeypatch.setattr(box, "run", lambda *a, **k: RunResult(200, {}, b"{}", 0.1))
    with pytest.raises(KeyboardInterrupt):
        box.watch(tmp_path, "/work/x", {}, None, View())
    assert events == [("watching", True), ("synced", ["infer.py"]), ("ran", True)]


def test_header_card_says_when_the_model_is_not_loaded():
    console = _console()
    LoopView(console, {}).watching(Path("klein"), run=True)
    assert "not loaded: start it with fal dev hot start" in console.export_text()


def test_lines_never_wrap():
    console = Console(record=True, width=50, force_terminal=True, color_system=None)
    _view(console).synced([f"module_{i}.py" for i in range(10)], 0.3)
    assert len(console.export_text().splitlines()) == 1


def test_watch_survives_a_failing_view(tmp_path, monkeypatch):
    (tmp_path / "infer.py").write_text("v1")
    box = hot.HotBox.__new__(hot.HotBox)
    lost = []

    class View:
        def watching(self, root, run):
            pass

        def synced(self, files, after_save):
            pass

        def ran(self, result, edited_at):
            raise OSError("preview broke")

        def restarting(self, reason):  # pragma: no cover
            pass

        def lost(self, error):
            lost.append(error)
            raise KeyboardInterrupt

    def sleep(_):
        import os

        (tmp_path / "infer.py").write_text("v2")
        os.utime(tmp_path / "infer.py", (1e9 + len(lost), 1e9 + len(lost)))

    monkeypatch.setattr(hot.time, "sleep", sleep)
    monkeypatch.setattr(box, "push", lambda root, remote, rels: ["infer.py"])
    monkeypatch.setattr(box, "run", lambda *a, **k: RunResult(200, {}, b"{}", 0.1))
    with pytest.raises(KeyboardInterrupt):
        box.watch(tmp_path, "/work/x", {}, None, View())
    assert lost == ["preview broke"]


def test_long_exceptions_keep_their_full_text_in_the_panel():
    message = (
        "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. "
        "GPU 0 has a total capacity of 95 GiB of which 1.2 GiB is free"
    )
    body = json.dumps(
        {"error": "infer() raised", "traceback": f"Traceback:\n{message}\n"}
    ).encode()
    console = _console()
    _view(console).ran(RunResult(422, {}, body, 0.1), edited_at=0)
    text = console.export_text()
    assert "error  OutOfMemoryError " in text
    assert "1.2 GiB is free" in text


def test_header_card_when_the_hot_server_did_not_answer():
    console = _console()
    LoopView(console, None).watching(Path("klein"), run=True)
    assert "model unknown (the hot server did not answer)" in console.export_text()


def test_long_sync_and_network_errors_keep_their_hint():
    console = Console(record=True, width=60, force_terminal=True, color_system=None)
    view = _view(console)
    view.sync_failed(
        ["infer.py"], "sync failed (500): disk quota exceeded on /work " * 2
    )
    view.lost("HTTPSConnectionPool(host='x.fal.run', port=443): Read timed out. " * 2)
    text = console.export_text()
    assert "sent again with the next save" in text
    assert "the next save retries" in text
    assert "Read timed out" in text
