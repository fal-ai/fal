import io
from types import SimpleNamespace

import pytest
from rich.console import Console

from dev.fakes import make_runner
from dev.test_dash import NOW, FakeDashBox, _probe, _snapshot
from fal.dev import loading
from fal.dev.cli import box as box_cli
from fal.dev.cli import summary
from fal.dev.dash.model import Cost, IdleWarning
from fal.dev.errors import DevError


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

        def up(self, timeout, machine_type, on_wait, starship=False, **kwargs):
            on_wait("pending")
            raise KeyboardInterrupt

    monkeypatch.setattr(box_cli, "get_box", lambda args, **kwargs: Box())
    monkeypatch.setattr(box_cli, "_cleanup_on_hangup", lambda: None)
    console = Console(record=True, width=100, force_terminal=False)
    args = SimpleNamespace(
        starship=False,
        secret=[],
        set_env=[],
        timeout=900,
        machine_type=None,
        dash=False,
        console=console,
        yes=True,
    )
    assert box_cli.cmd_up(args) == 130
    assert "cancelled the start" in console.export_text()


def test_status_fails_loudly_when_the_platform_does_not_answer(monkeypatch):
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

        def up(self, timeout, machine_type, on_wait, starship=False, **kwargs):
            return None

    def fail(box):
        raise DevError("401 Unauthorized")

    monkeypatch.setattr(box_cli, "get_box", lambda args, **kwargs: Box())
    monkeypatch.setattr(box_cli, "_cleanup_on_hangup", lambda: None)
    monkeypatch.setattr(summary, "snapshot", fail)
    console = Console(record=True, width=200, force_terminal=False)
    args = SimpleNamespace(
        starship=False,
        secret=[],
        set_env=[],
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

        def up(self, timeout, machine_type, on_wait, starship=False, **kwargs):
            on_wait("pending")
            hints.append(self.progress.hint)

    def read(box):
        hints.append(box.progress.hint)
        raise KeyboardInterrupt

    monkeypatch.setattr(box_cli, "get_box", lambda args, **kwargs: Box())
    monkeypatch.setattr(box_cli, "_cleanup_on_hangup", lambda: None)
    monkeypatch.setattr(summary, "snapshot", read)
    console = Console(record=True, width=200, force_terminal=False)
    args = SimpleNamespace(
        starship=False,
        secret=[],
        set_env=[],
        timeout=900,
        machine_type=None,
        dash=False,
        console=console,
        yes=True,
    )
    assert box_cli.cmd_up(args) == 130
    assert "stops billing" in hints[0]
    assert "not the box" in hints[-1]


def test_status_line_shows_an_active_hold_and_when_it_ends():
    probe = _probe(
        activity={"held": True, "hold_until": NOW + 5400},
        idle_state={"last_active": NOW, "idle_minutes": 60},
        now=NOW,
    )
    line = summary.status_line(_snapshot(probe), FakeDashBox(), now=NOW).plain
    assert "held until" in line
    assert "(1h 30m left)" in line
    card = _text(summary.ready_card(_snapshot(probe), FakeDashBox(), now=NOW))
    assert "fal dev hold --off" in card


def test_status_line_has_no_hold_text_without_a_hold():
    line = summary.status_line(
        _snapshot(_idle_probe(38 * 60)), FakeDashBox(), now=NOW
    ).plain
    assert "held" not in line


def _gpu_probe(util=0, processes=()):
    probe = _probe()
    probe["gpus"] = [{**probe["gpus"][0], "util_pct": util}]
    probe["system"] = {"processes": list(processes)}
    return probe


def test_busy_reasons_name_a_busy_gpu_and_gpu_jobs():
    assert box_cli.busy_reasons(_gpu_probe(util=0)) == []
    assert "GPU is busy (97% utilization)" in box_cli.busy_reasons(_gpu_probe(97))[0]
    job = {"pid": 7, "name": "python3", "gpu_mib": 900}
    shell = {"pid": 8, "name": "bash", "gpu_mib": 10}
    cpu_only = {"pid": 9, "name": "tar"}
    reasons = box_cli.busy_reasons(_gpu_probe(0, [job, shell, cpu_only]))
    assert reasons == ["python3 (pid 7) is using the GPU"]


class _DownBox(FakeDashBox):
    def __init__(self, probe):
        super().__init__(probe)
        self.downs = []

    def runners(self):
        return [make_runner("runner-abc")]

    def down(self, no_save):
        self.downs.append(no_save)


def _down(box, monkeypatch, *, tty, yes=False):
    monkeypatch.setattr(box_cli, "get_box", lambda args: box)
    monkeypatch.setattr(box_cli.sys.stdin, "isatty", lambda: tty, raising=False)
    box_cli.cmd_down(
        SimpleNamespace(yes=yes, no_save=False, console=Console(file=io.StringIO()))
    )


def test_down_refuses_a_busy_gpu_without_a_terminal(monkeypatch):
    box = _DownBox(_gpu_probe(95))
    with pytest.raises(DevError, match="--yes"):
        _down(box, monkeypatch, tty=False)
    assert box.downs == []


def test_down_asks_on_a_terminal_and_honors_the_answer(monkeypatch):
    box = _DownBox(_gpu_probe(95))
    monkeypatch.setattr(box_cli.Confirm, "ask", lambda *a, **k: False)
    with pytest.raises(DevError, match="cancelled"):
        _down(box, monkeypatch, tty=True)
    monkeypatch.setattr(box_cli.Confirm, "ask", lambda *a, **k: True)
    _down(box, monkeypatch, tty=True)
    assert box.downs == [False]


def test_down_yes_skips_the_check_and_an_idle_box_stops_quietly(monkeypatch):
    busy = _DownBox(_gpu_probe(95))
    _down(busy, monkeypatch, tty=False, yes=True)
    assert busy.downs == [False] and busy.run_calls == []
    idle = _DownBox(_gpu_probe(0))
    _down(idle, monkeypatch, tty=False)
    assert idle.downs == [False]


def test_down_still_stops_when_the_probe_fails(monkeypatch):
    box = _DownBox(_gpu_probe(0))
    box.run = lambda *a, **k: (1, "boom")
    _down(box, monkeypatch, tty=False)
    assert box.downs == [False]


def _hold(box, monkeypatch, **fields):
    monkeypatch.setattr(box_cli, "get_box", lambda args: box)
    out = io.StringIO()
    args = SimpleNamespace(
        off=False, hours=None, console=Console(file=out, width=200), **fields
    )
    return args, out


def test_hold_off_and_zero_clear_the_hold(monkeypatch):
    box = SimpleNamespace(clear_hold=lambda: True)
    for fields in ({"off": True}, {"hours": 0.0}):
        args, out = _hold(box, monkeypatch)
        vars(args).update(fields)
        box_cli.cmd_hold(args)
        assert "hold cleared" in out.getvalue()
    box = SimpleNamespace(clear_hold=lambda: False)
    args, out = _hold(box, monkeypatch)
    args.off = True
    box_cli.cmd_hold(args)
    assert "no hold was set" in out.getvalue()


def test_status_and_ready_card_show_secret_names_only():
    snap = _snapshot(
        _probe(user_env={"secrets": ["HF_TOKEN", "REGISTRY_TOKEN"], "env": ["FOO"]})
    )

    line = summary.status_line(snap, FakeDashBox(), now=NOW).plain
    card = _text(summary.ready_card(snap, FakeDashBox(), 98))

    assert "secrets: HF_TOKEN, REGISTRY_TOKEN" in line
    assert "HF_TOKEN, REGISTRY_TOKEN" in card
    assert "FOO" not in line


def test_status_without_secrets_or_from_an_older_probe_shows_none():
    assert (
        "secrets"
        not in summary.status_line(_snapshot(_probe()), FakeDashBox(), now=NOW).plain
    )
    older = _probe()
    older.pop("user_env", None)
    assert (
        "secrets"
        not in summary.status_line(_snapshot(older), FakeDashBox(), now=NOW).plain
    )
