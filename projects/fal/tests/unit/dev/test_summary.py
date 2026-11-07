import io
from types import SimpleNamespace

from rich.console import Console

from dev.test_dash import NOW, FakeDashBox, _probe, _snapshot
from fal.dev import loading
from fal.dev.cli import box as box_cli
from fal.dev.cli import summary
from fal.dev.dash.model import Cost, IdleWarning


def _text(renderable) -> str:
    console = Console(record=True, width=110, force_terminal=False, color_system=None)
    console.print(renderable)
    return console.export_text()


def _idle_probe(remaining: float) -> dict:
    return _probe(
        idle_state={"last_active": NOW - (3600 - remaining), "idle_minutes": 60},
        now=NOW,
    )


def test_status_line_for_a_running_box():
    snap = _snapshot(
        _idle_probe(38 * 60), cost=Cost(3.15, 2.0), cost_runner="runner-abc"
    )
    line = summary.status_line(snap, FakeDashBox(), now=NOW).plain
    assert line.startswith("devbox-demo  RUNNING  1x NVIDIA H100 80GB HBM3  up 1h 00m")
    assert "$3.15 ($2.00/h)" in line
    assert "idle stop in 38m 00s" in line
    assert "/work saved" in line


def test_ready_card_says_how_to_connect_and_stop():
    text = _text(summary.ready_card(_snapshot(_idle_probe(3600)), FakeDashBox(), 98))
    assert "READY" in text.splitlines()[0]
    assert "ready in 1m38s" in text
    for command in ("fal dev shell", "fal dev dash", "ssh devbox-demo", "fal dev down"):
        assert command in text
    assert "stops itself after 60 min idle" in text


def test_ready_card_warns_when_idle_shutdown_is_off():
    probe = _probe(warnings=["idle shutdown is off: DEVBOX_ADMIN_KEY is not set"])
    text = _text(summary.ready_card(_snapshot(probe), FakeDashBox()))
    assert "idle shutdown is OFF" in text


def test_idle_warning_fires_once_per_idle_stretch():
    warning = IdleWarning()
    assert warning.check(_snapshot(_idle_probe(20 * 60)), NOW) is None
    message = warning.check(_snapshot(_idle_probe(9 * 60)), NOW)
    assert message and "stops itself in about 9 min" in message and "press h" in message
    assert warning.check(_snapshot(_idle_probe(8 * 60)), NOW) is None
    assert warning.check(_snapshot(_idle_probe(30 * 60)), NOW) is None
    assert warning.check(_snapshot(_idle_probe(5 * 60)), NOW)


def test_runner_phase_in_words():
    assert loading.runner_phase("pending") == "queued"
    assert (
        loading.runner_phase("docker_pull, setup")
        == "pulling the image, starting the box"
    )
    assert loading.runner_phase("requested") == "requested"


class _Tty(io.StringIO):
    encoding = "utf-8"

    def isatty(self):
        return True


def test_a_long_wait_rings_the_bell_and_a_short_one_does_not():
    for seconds, bell in ((25, True), (5, False)):
        out = _Tty()
        clock = iter([0.0, 0.0, seconds, seconds, seconds])
        wait = loading.Loading(
            "w", stream=out, animate=False, clock=lambda: next(clock)
        )
        wait.start()
        wait.done()
        assert ("\a" in out.getvalue()) is bell


class _BoxBase:
    name = "demo"
    account = SimpleNamespace(personal=True, team="troy")
    progress = None


def test_ctrl_c_during_up_cancels_and_says_it_stopped_billing(monkeypatch):
    class Box(_BoxBase):
        start_cancelled = True

        def app_info(self):
            return object()

        def up(self, timeout, machine_type, on_wait, starship=False):
            on_wait("pending")
            raise KeyboardInterrupt

    monkeypatch.setattr(box_cli, "get_box", lambda args: Box())
    monkeypatch.setattr(box_cli, "_cleanup_on_hangup", lambda: None)
    console = Console(record=True, width=100, force_terminal=False)
    args = SimpleNamespace(
        starship=False,
        timeout=900,
        machine_type=None,
        dash=False,
        console=console,
        yes=True,
    )
    assert box_cli.cmd_up(args) == 130
    assert "cancelled the start" in console.export_text()


def test_status_fails_loudly_when_the_platform_does_not_answer(monkeypatch):
    import pytest

    from fal.dev.errors import DevError

    class Poller:
        def __init__(self, box):
            from fal.dev.dash.model import Snapshot

            self.snapshot = Snapshot(errors={"state": "401 Unauthorized"})

        def prime(self):
            pass

    monkeypatch.setattr(summary, "Poller", Poller)
    with pytest.raises(DevError, match="401 Unauthorized"):
        summary.snapshot(FakeDashBox())


def test_up_says_the_box_is_billing_when_its_details_cannot_be_read(monkeypatch):
    from fal.dev.errors import DevError

    class Box(_BoxBase):
        start_cancelled = None

        def app_info(self):
            return object()

        def up(self, timeout, machine_type, on_wait, starship=False):
            return None

    def fail(box):
        raise DevError("401 Unauthorized")

    monkeypatch.setattr(box_cli, "get_box", lambda args: Box())
    monkeypatch.setattr(box_cli, "_cleanup_on_hangup", lambda: None)
    monkeypatch.setattr(summary, "snapshot", fail)
    console = Console(record=True, width=200, force_terminal=False)
    args = SimpleNamespace(
        starship=False,
        timeout=900,
        machine_type=None,
        dash=False,
        console=console,
        yes=True,
    )
    assert box_cli.cmd_up(args) is None
    text = console.export_text()
    assert "up and billing" in text and "401 Unauthorized" in text


def test_once_the_box_is_up_the_hint_no_longer_promises_to_stop_billing(monkeypatch):
    hints = []

    class Box(_BoxBase):
        start_cancelled = None

        def app_info(self):
            return object()

        def up(self, timeout, machine_type, on_wait, starship=False):
            on_wait("pending")
            hints.append(self.progress.hint)

    def read(box):
        hints.append(box.progress.hint)
        raise KeyboardInterrupt

    monkeypatch.setattr(box_cli, "get_box", lambda args: Box())
    monkeypatch.setattr(box_cli, "_cleanup_on_hangup", lambda: None)
    monkeypatch.setattr(summary, "snapshot", read)
    console = Console(record=True, width=200, force_terminal=False)
    args = SimpleNamespace(
        starship=False,
        timeout=900,
        machine_type=None,
        dash=False,
        console=console,
        yes=True,
    )
    assert box_cli.cmd_up(args) == 130
    assert "stops billing" in hints[0]
    assert "not the box" in hints[-1]
