import sys

import pytest

from fal.cli.main import parse_args
from fal.dev.account import Account
from fal.dev.app import bench as app_bench
from fal.dev.app import client
from fal.dev.app import session as app_session
from fal.dev.app.session import Session
from fal.dev.errors import DevError

REPO_APP_SOURCE = """
import fal

from pkg.helpers import answer


class Demo(fal.App):
    # A named package index in the class requirements, as many apps use.
    requirements = [
        "--index",
        "private=https://pkgs.example.com/simple",
        "private-pkg",
    ]
    keep_alive = 60

    def setup(self):
        self.value = answer()

    @fal.endpoint("/")
    def run(self) -> dict:
        return {"value": self.value}
"""


PYPROJECT = r"""
[tool.fal.apps.demo]
ref = "pkg/models/app.py::Demo"
team = "someone-else"
auth = "shared"
machine_type = "GPU-A100"
keep_alive = 600
max_concurrency = 10
app_files = ["assets"]
# Relative to pyproject.toml, as `fal deploy` reads it.
app_files_context_dir = "."
app_files_ignore = ["\\.bin$"]
exposed_port = 9000
"""


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repo with a package, an app two levels down and a pyproject entry."""
    models = tmp_path / "pkg" / "models"
    models.mkdir(parents=True)
    for folder in (tmp_path / "pkg", models):
        (folder / "__init__.py").write_text("")
    (tmp_path / "pkg" / "helpers.py").write_text("def answer():\n    return 1\n")
    (models / "app.py").write_text(REPO_APP_SOURCE)
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "prompt.txt").write_text("a cat")
    (tmp_path / "assets" / "weights.bin").write_text("x")
    (tmp_path / "pyproject.toml").write_text(PYPROJECT)
    monkeypatch.chdir(tmp_path)
    from fal import project

    # pyproject.toml lookups are cached per process and keyed on the cwd.
    for cached in (project.find_project_root, project._cached_resolve):
        cached.cache_clear()
    from fal.api.client import SyncServerlessClient

    # No credential lookups (they call the REST API for team keys).
    monkeypatch.setattr(SyncServerlessClient, "_credentials", property(lambda s: None))
    yield tmp_path
    for name in [m for m in sys.modules if m == "pkg" or m.startswith("pkg.")]:
        sys.modules.pop(name, None)


def make_tree_session(tmp_path, *names) -> Session:
    root = tmp_path / "tree"
    root.mkdir(exist_ok=True)
    for name in names:
        (root / name).write_text(name)
    return Session(
        name="dev-x",
        file_path=str(root / names[0]),
        root=str(root),
        module="a",
        class_name="A",
        run_url="https://fal.run/me/dev-x",
        token="t",
        account=Account("me", True),
    )


class FakeDepsBox:
    """The holder's deps route, as far as the CLI sees it."""

    def __init__(
        self,
        fail_on: str | None = None,
        error: str = "deps failed (400): index unreachable",
    ) -> None:
        self.error = error
        self.state: dict = {"packages": []}
        self.sent: list[tuple[str, dict]] = []
        self.fail_on = fail_on

    def call(self, _session, action, payload: dict | None = None, **_):
        payload = payload or {}
        self.sent.append((action, payload))
        if action == "status":
            return {"phase": "ready", "module": "demo_app"}
        assert action == "deps"
        pins = []
        if payload["action"] == "install":
            if self.fail_on and self.fail_on in payload["packages"][0]:
                raise DevError(self.error)
            pins = payload["packages"]
            if not payload.get("no_deps"):
                pins = [*pins, "safetensors==0.5.0"]
            self.state = {
                "packages": [*self.state["packages"], *pins],
                "pre": bool(payload.get("pre")),
                "index_url": payload.get("index_url"),
                "extra_index_urls": payload.get("extra_index_urls") or [],
            }
        elif payload["action"] == "reset":
            self.state = {"packages": []}
        return {
            "pins": pins,
            "packages": [
                {"name": pin.split("==")[0], "version": "1", "base_version": None}
                for pin in self.state["packages"]
            ],
            "replay": dict(self.state),
            "removed": 0,
            "restarted": False,
        }


def make_cli_session(tmp_path, monkeypatch, box):
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(client, "call", box.call)
    app_session.Session(
        name="dev-demo-me",
        file_path="/src/demo_app.py",
        root="/src",
        module="demo_app",
        class_name="Demo",
        run_url="https://fal.run/me/dev-demo-me",
        token="t",
        account=Account("me", True),
    ).save()


def make_session(**kwargs) -> Session:
    values = {
        "name": "dev-demo-me",
        "file_path": "/src/demo_app.py",
        "root": "/src",
        "module": "demo_app",
        "class_name": "Demo",
        "run_url": "https://fal.run/me/dev-demo-me",
        "token": "t",
        "account": Account("me", True),
        "machine_type": "GPU-H100",
        "manifest": {"demo_app.py": "abc"},
    }
    return Session(**{**values, **kwargs})


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(app_bench, "BENCHMARKS_DIR", tmp_path / "benchmarks")
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path / "sessions")
    return tmp_path


def make_request(*extra):
    args = parse_args(
        ["dev", "app", "request", "/", "--data", '{"prompt": "a"}', *extra]
    )
    return args.func(args)
