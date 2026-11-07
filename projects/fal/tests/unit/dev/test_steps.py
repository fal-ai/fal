import io
from types import SimpleNamespace

from rich.console import Console

from fal.dev import box as box_module
from fal.dev import config
from fal.dev.steps import ACTIVE, DONE, FAILED, PENDING, Steps

SPECS = [
    ("volume", "create volume"),
    ("build", "build environment"),
    ("deploy", "deploy"),
    ("runner", "start runner", 100),
    ("ready", "ready"),
]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def tick(self, seconds):
        self.now += seconds


def make(clock=None, **kwargs):
    clock = clock or Clock()
    out = io.StringIO()
    steps = Steps(
        SPECS,
        title="voice",
        team="personal",
        machine="GPU-RTXPRO6000",
        stream=out,
        animate=kwargs.pop("animate", False),
        clock=clock,
        **kwargs,
    )
    return steps, clock, out


def text(renderable, width=90) -> str:
    console = Console(record=True, width=width, force_terminal=False, color_system=None)
    console.print(renderable)
    return console.export_text()


def test_steps_go_pending_active_done_with_timings():
    steps, clock, _ = make()
    assert [s.state for s in steps.steps] == [PENDING] * 5
    steps.begin("volume")
    assert steps.state("volume") == ACTIVE
    clock.tick(2)
    steps.finish("volume", "created, starts empty")
    assert steps.state("volume") == DONE
    assert steps.elapsed("volume") == 2
    steps.begin("build")
    clock.tick(75)
    assert steps.elapsed("build") == 75  # still counting while active
    steps.finish("build")
    clock.tick(30)
    assert steps.elapsed("build") == 75  # frozen once done


def test_finishing_a_step_that_never_began_takes_no_time():
    steps, clock, _ = make()
    clock.tick(5)
    steps.finish("build")
    assert steps.state("build") == DONE
    assert steps.elapsed("build") == 0


def test_unknown_steps_are_ignored():
    steps, _, _ = make()
    steps.begin("nope")
    steps.finish("nope")
    assert steps.active is None
    assert not steps.started("nope")


def test_failure_marks_the_active_step_and_keeps_the_others():
    steps, clock, _ = make()
    steps.begin("volume")
    steps.finish("volume")
    steps.begin("build")
    clock.tick(9)
    steps.fail("boom")
    states = {s.key: s.state for s in steps.steps}
    assert states == {
        "volume": DONE,
        "build": FAILED,
        "deploy": PENDING,
        "runner": PENDING,
        "ready": PENDING,
    }
    assert steps.elapsed("build") == 9


def test_an_error_before_any_step_began_marks_nothing():
    steps, _, _ = make()
    steps.fail("bad python")
    assert [s.state for s in steps.steps] == [PENDING] * 5


def test_a_finished_step_drops_its_working_detail():
    steps, _, _ = make()
    steps.begin("runner", "pulling the image")
    steps.finish("runner")
    assert steps.steps[3].detail == ""


def test_the_ctrl_c_hint_shows_only_while_a_runner_is_started():
    steps, _, _ = make(hint="Ctrl-C cancels")
    steps.begin("build")
    assert "Ctrl-C" not in text(steps.renderable(100, 40))
    steps.finish("build")
    steps.begin("runner")
    assert "Ctrl-C" in text(steps.renderable(100, 40))
    plain, _, out = make(hint="Ctrl-C cancels")
    plain.begin("runner")
    assert "Ctrl-C cancels" in out.getvalue()


def test_ansi_is_stripped_from_captured_logs():
    steps, _, _ = make()
    steps.capture("\x1b[2K\x1b[31mred\x1b[0m")
    assert steps.log_lines == ["red"]


def test_short_terminals_get_the_bare_steps():
    steps, _, _ = make()
    assert "╭" not in text(steps.renderable(100, 8))


def test_the_log_keeps_the_last_lines_unless_verbose():
    steps, _, _ = make()
    for i in range(40):
        steps.capture(f"line {i}")
    assert steps.log_lines[0] == "line 25" and steps.log_lines[-1] == "line 39"
    verbose, _, _ = make(verbose=True)
    for i in range(40):
        verbose.capture(f"line {i}")
    assert len(verbose.log_lines) == 40


def test_log_prints_under_the_panel():
    steps, _, _ = make()
    steps.capture("ERROR: no such package")
    console = Console(record=True, width=90, force_terminal=False)
    steps.print_log(console)
    out = console.export_text()
    assert "deploy log (last 1 lines)" in out and "ERROR: no such package" in out


def test_plain_mode_prints_the_same_steps_as_lines():
    steps, clock, out = make()
    steps.open()
    steps.begin("volume")
    clock.tick(2)
    steps.finish("volume", "created, starts empty")
    steps.begin("runner")
    clock.tick(61)
    steps.fail("no GPU")
    lines = out.getvalue().splitlines()
    assert lines == [
        "voice: team personal, GPU-RTXPRO6000",
        "[1/5] create volume ...",
        "[1/5] create volume: done (2s) - created, starts empty",
        "[4/5] start runner ... (typically ~1m40s)",
        "[4/5] start runner: failed (1m01s) - no GPU",
    ]


def test_panel_shows_header_steps_timings_and_hint():
    steps, clock, _ = make(hint="Ctrl-C cancels")
    steps.begin("volume")
    clock.tick(2)
    steps.finish("volume")
    steps.begin("build")
    clock.tick(14)
    steps.begin("runner")
    steps.finish("build")
    out = text(steps.renderable(100, 40))
    first = out.splitlines()
    assert "voice" in first[0]
    assert "team personal" in out and "GPU-RTXPRO6000" in out
    for label in ("create volume", "build environment", "deploy", "ready"):
        assert label in out
    assert "✓ create volume" in out and "2s" in out
    assert "build environment" in out and "14s" in out
    assert "typically ~1m40s" in out
    assert "Ctrl-C cancels" in out
    assert any(0x2800 <= ord(c) <= 0x28FF for c in out)  # the dust


def test_panel_marks_a_failed_step():
    steps, clock, _ = make(animate=True)
    steps.begin("build")
    clock.tick(3)
    steps.fail("exit 1")
    out = text(steps.renderable(100, 40))
    assert "✗ build environment" in out and "exit 1" in out
    assert "○ deploy" in out
    assert "Ctrl-C" not in out


def test_narrow_terminals_get_no_frame():
    steps, _, _ = make()
    out = text(steps.renderable(40, 40), width=40)
    assert "╭" not in out and "create volume" in out


def test_animated_run_draws_a_panel_and_leaves_it_on_screen():
    class Tty(io.StringIO):
        encoding = "utf-8"

        def isatty(self):
            return True

    out = Tty()
    steps = Steps(
        SPECS[3:],
        title="voice",
        team="personal",
        machine="GPU-H100",
        stream=out,
        animate=True,
    )
    steps.open()
    steps.begin("runner")
    steps.finish("runner")
    steps.finish("ready")
    steps.done()
    assert "start runner" in out.getvalue() and "✓" in out.getvalue()


def test_the_switches_pick_the_plain_mode():
    for env in ({"NO_COLOR": "1"}, {"TERM": "dumb"}, {"FAL_DEV_NO_ANIMATION": "1"}):
        steps = Steps(
            SPECS, title="t", team="x", machine="m", stream=io.StringIO(), environ=env
        )
        steps.begin("volume")
        assert "[1/5] create volume ..." in steps._stream.getvalue()


def test_box_turns_deploy_logs_into_step_changes():
    steps, clock, _ = make()
    box = box_module.Box.__new__(box_module.Box)
    box.progress = steps
    handlers = box._deploy_handlers()
    build, register = handlers["build_result_handler"], handlers["result_handler"]
    log = SimpleNamespace(message="Installing torch")
    build(SimpleNamespace(logs=[log], service_urls=None))
    assert steps.state("build") == ACTIVE
    clock.tick(30)
    register(SimpleNamespace(logs=[SimpleNamespace(message="Registered")]))
    assert steps.state("build") == DONE and steps.elapsed("build") == 30
    assert steps.state("deploy") == ACTIVE
    assert steps.log_lines == ["Installing torch", "Registered"]


def test_box_without_progress_leaves_the_deploy_alone():
    box = box_module.Box.__new__(box_module.Box)
    box.progress = None
    assert box._deploy_handlers() == {}
    box._begin("build")  # no-ops
    box._finish("build")


def test_up_steps_for_a_first_deploy_and_for_an_existing_box():
    from fal.dev.cli import box as box_cli

    box = SimpleNamespace(
        name="voice", account=SimpleNamespace(personal=True, team="t")
    )
    args = SimpleNamespace(machine_type=None, verbose=False)
    first = box_cli._up_steps(box, args, True, None)
    assert [s.key for s in first.steps] == [
        "volume",
        "build",
        "deploy",
        "runner",
        "ready",
    ]
    assert (first.team, first.machine) == ("personal", config.DEFAULT_GPU)
    info = SimpleNamespace(machine_types=["GPU-RTXPRO6000"])
    team = SimpleNamespace(
        name="voice", account=SimpleNamespace(personal=False, team="red")
    )
    again = box_cli._up_steps(team, args, False, info)
    assert [s.key for s in again.steps] == ["runner", "ready"]
    assert (again.team, again.machine) == ("red", "GPU-RTXPRO6000")


def test_a_failed_up_marks_the_step_and_prints_the_log_tail():
    from fal.dev.cli import box as box_cli
    from fal.dev.errors import DevError

    class FakeBox:
        name = "voice"
        account = SimpleNamespace(personal=True, team="t")
        progress = None

        def app_info(self):
            return object()

        def up(self, timeout, machine_type, on_wait, starship=False, **kwargs):
            self.progress.begin("runner")
            self.progress.capture("image pull failed")
            raise DevError("setup() failed")

    holder = {}
    real = box_cli._up_steps

    def spy(*a):
        holder["steps"] = real(*a)
        return holder["steps"]

    import pytest

    mp = pytest.MonkeyPatch()
    mp.setattr(box_cli, "get_box", lambda args, **kwargs: FakeBox())
    mp.setattr(box_cli, "_cleanup_on_hangup", lambda: None)
    mp.setattr(box_cli, "_up_steps", spy)
    try:
        console = Console(record=True, width=100, force_terminal=False)
        args = SimpleNamespace(
            timeout=9,
            machine_type=None,
            dash=False,
            console=console,
            yes=True,
            starship=False,
            secret=[],
            set_env=[],
        )
        with pytest.raises(DevError):
            box_cli.cmd_up(args)
    finally:
        mp.undo()
    steps = holder["steps"]
    assert steps.state("runner") == FAILED
    assert steps.steps[0].detail == "setup() failed"
    assert "image pull failed" in console.export_text()


def test_box_deploy_walks_the_steps_in_order(monkeypatch, tmp_path):
    from fal.dev import state

    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    steps, clock, _ = make()
    box = box_module.Box.__new__(box_module.Box)
    box.progress = steps
    box.name, box.app, box.volume_created = "voice", "devbox-voice", False
    box.host = None
    seen = []

    class Client:
        def deploy(self, *a, **kw):
            build, register = kw["build_result_handler"], kw["result_handler"]
            build(SimpleNamespace(logs=[SimpleNamespace(message="b")]))
            seen.append(dict(steps_state(steps)))
            register(SimpleNamespace(logs=[SimpleNamespace(message="r")]))
            seen.append(dict(steps_state(steps)))

    def steps_state(st):
        return {s.key: s.state for s in st.steps}

    box.client = Client()
    box.app_info = lambda: None
    box.ensure_volume = lambda: "devbox-voice"
    box.stop = lambda: True
    box.wait_until_stopped = lambda timeout: True
    box.scale = lambda *a: None
    box.deploy(machine_type="GPU-H100")
    assert seen[0]["volume"] == DONE and seen[0]["build"] == ACTIVE
    assert seen[1]["build"] == DONE and seen[1]["deploy"] == ACTIVE
    assert steps.state("deploy") == DONE


def test_plain_ready_line_carries_the_billing_hint_not_the_cancel_hint(monkeypatch):
    from fal.dev.cli import box as box_cli
    from fal.dev.cli import summary

    class FakeBox:
        name = "voice"
        account = SimpleNamespace(personal=True, team="t")
        progress = None

        def app_info(self):
            return object()

        def up(self, timeout, machine_type, on_wait, starship=False, **kwargs):
            self.progress.begin("runner")
            self.progress.finish("runner")

    def read(box):
        raise KeyboardInterrupt

    monkeypatch.setattr(box_cli, "get_box", lambda args, **kwargs: FakeBox())
    monkeypatch.setattr(box_cli, "_cleanup_on_hangup", lambda: None)
    monkeypatch.setattr(summary, "snapshot", read)
    err = io.StringIO()
    monkeypatch.setattr("sys.stderr", err)
    args = SimpleNamespace(
        timeout=9,
        machine_type=None,
        dash=False,
        console=Console(file=io.StringIO()),
        yes=True,
        starship=False,
        secret=[],
        set_env=[],
    )
    assert box_cli.cmd_up(args) == 130
    out = err.getvalue()
    assert "the box is up and billing" in out
    assert out.count("stops billing") == 1  # the runner's hint, before it was up
