import sys
from pathlib import Path

import pytest

from fal.dev.remote.holder import DevAppHolder
from fal.dev.remote.process import LogBuffer

APP_SOURCE = """
import fal
import helpers

MODEL = "model-a"
STEPS = 4
OPTIONS = {"dtype": "bf16"}
PIPE = None


def load(name):
    return {"model": name, "loads": 1}


class Demo(fal.App):
    machine_type = "GPU-H100"
    requirements = ["torch"]
    keep_alive = 60
    max_concurrency = 5

    def setup(self):
        global PIPE
        self.state = load(MODEL)
        self.options = dict(OPTIONS)
        PIPE = "loaded"
        self.setup_calls = getattr(self, "setup_calls", 0) + 1

    @fal.endpoint("/")
    def run(self) -> dict:
        model = self.state["model"]
        return {
            "answer": helpers.answer(),
            "model": model,
            "steps": STEPS,
            "pipe": PIPE,
        }
"""


HELPERS_SOURCE = """
def answer():
    return 1
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "demo_app.py").write_text(APP_SOURCE)
    (tmp_path / "helpers.py").write_text(HELPERS_SOURCE)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for name in ("demo_app", "helpers"):
        sys.modules.pop(name, None)


def make_dist(target: Path, name: str, version: str) -> None:
    info = target / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    )


class FakeInstaller:
    """Records uv commands; --target installs write dist-info folders."""

    def __init__(self, plan: list[str]) -> None:
        self.plan = plan
        self.commands: list[list[str]] = []

    def run(self, command: list[str]) -> str:
        self.commands.append(command)
        if "--dry-run" in command:
            return "\n".join(f" + {pin}" for pin in self.plan)
        target = Path(command[command.index("--target") + 1])
        for pin in command[command.index("--no-deps") + 1 :]:
            if "==" in pin:
                name, version = pin.split("==")
                make_dist(target, name, version)
        return ""


@pytest.fixture
def roomy(monkeypatch):
    """The store's free-space floor, off: test disks are small."""
    from fal.dev.remote import objects as app_objects

    monkeypatch.setattr(app_objects, "MIN_FREE_BYTES", 0)


def bare_holder(**attrs) -> DevAppHolder:
    """A holder without setup(): token "t", a log buffer, and `_<name>` attributes."""
    holder = object.__new__(DevAppHolder)
    holder._fal_dev_token = "t"
    holder._log_buffer = LogBuffer()
    for name, value in attrs.items():
        setattr(holder, f"_{name}", value)
    return holder
