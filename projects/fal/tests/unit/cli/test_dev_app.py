import base64
import io
import os
import sys
import tarfile
import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import fal
from fal.cli.main import parse_args
from fal.dev.app import sync as app_sync
from fal.dev.app.session import NAME_PREFIX, session_name
from fal.dev.remote import reload as app_reload
from fal.dev.remote.child import DevAppServer
from fal.dev.remote.holder import (
    DevAppHolder,
    _ChildProcess,
    _extract,
    _forward,
    make_holder,
)

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


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))


@pytest.fixture
def server(project, monkeypatch):
    monkeypatch.setattr(fal.ref, "current_app", None)
    server = DevAppServer(project, "demo_app", "Demo")
    with TestClient(server) as client:
        yield server, client


def test_setup_runs_once_and_endpoint_reload_keeps_state(server, project):
    server, client = server
    assert client.post("/", json={}).json() == {
        "answer": 1,
        "model": "model-a",
        "steps": 4,
        "pipe": "loaded",
    }
    assert server.instance.setup_calls == 1

    _edit(project / "helpers.py", "return 1", "return 2")
    _edit(project / "demo_app.py", "STEPS = 4", "STEPS = 8")
    result = client.post(
        "/__fal_dev_child/reload", json={"files": ["helpers.py"]}
    ).json()

    assert result["action"] == "reload"
    assert result["code_version"] == 1
    assert client.post("/", json={}).json()["answer"] == 2
    assert client.post("/", json={}).json()["steps"] == 8
    # A module global that setup() assigned survives the re-import.
    assert client.post("/", json={}).json()["pipe"] == "loaded"
    assert server.instance.setup_calls == 1


def test_setup_change_asks_for_restart(server, project):
    server, client = server
    _edit(
        project / "demo_app.py",
        "self.setup_calls =",
        "self.extra = 1\n        self.setup_calls =",
    )
    result = client.post("/__fal_dev_child/reload", json={}).json()
    assert result["action"] == "restart"
    assert "setup" in result["reason"]
    assert server.generation == 0


def test_constant_used_by_setup_asks_for_restart(server, project):
    server, client = server
    _edit(project / "demo_app.py", '"model-a"', '"model-b"')
    result = client.post("/__fal_dev_child/reload", json={}).json()
    assert result["action"] == "restart"
    assert "MODEL" in result["reason"]


def test_mutable_constant_used_by_setup_asks_for_restart(server, project):
    server, client = server
    _edit(project / "demo_app.py", '"bf16"', '"fp16"')
    result = client.post("/__fal_dev_child/reload", json={}).json()
    assert result["action"] == "restart"
    assert "OPTIONS" in result["reason"]


def test_parent_setup_is_fingerprinted(tmp_path):
    (tmp_path / "base_app.py").write_text(
        "import fal\nSIZE = 1\n\n\nclass Base(fal.App):\n"
        "    def setup(self):\n        self.size = SIZE\n\n\n"
        "class Child(Base):\n    def setup(self):\n        super().setup()\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        import base_app

        fingerprint = app_reload.setup_fingerprint(base_app.Child, tmp_path)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("base_app", None)
    assert any(key.endswith(".SIZE") for key in fingerprint)


def test_helper_called_by_setup_asks_for_restart(server, project):
    server, client = server
    _edit(project / "demo_app.py", '"loads": 1', '"loads": 2')
    result = client.post("/__fal_dev_child/reload", json={}).json()
    assert result["action"] == "restart"
    assert "load" in result["reason"]


def test_requirements_change_asks_for_rebuild(server, project):
    server, client = server
    _edit(project / "demo_app.py", '["torch"]', '["torch", "diffusers"]')
    result = client.post("/__fal_dev_child/reload", json={}).json()
    assert result["action"] == "rebuild"
    assert "requirements" in result["reason"]


def test_import_error_returns_422_until_fixed(server, project):
    server, client = server
    _edit(project / "helpers.py", "return 1", "return (")
    result = client.post("/__fal_dev_child/reload", json={}).json()
    assert result["action"] == "error"
    response = client.post("/", json={})
    assert response.status_code == 422
    assert "SyntaxError" in response.json()["traceback"]

    _edit(project / "helpers.py", "return (", "return 3")
    assert client.post("/__fal_dev_child/reload", json={}).json()["action"] == "reload"
    assert client.post("/", json={}).json()["answer"] == 3
    # The global setup() assigned survives the failed reload in between.
    assert client.post("/", json={}).json()["pipe"] == "loaded"


def test_decide_prefers_rebuild_over_restart():
    action, _ = app_reload.decide({"a": "1"}, {"a": "2"}, {"r": "1"}, {"r": "2"})
    assert action == "rebuild"
    assert app_reload.decide({}, {}, {}, {})[0] == "reload"


def test_make_holder_inherits_environment_but_not_routes_or_scaling(project):
    import demo_app

    holder = make_holder(
        demo_app.Demo,
        app_name="dev-demo-me",
        session="dev-demo-me",
        token="t",
        keep_alive=900,
    )
    assert issubclass(holder, demo_app.Demo)
    assert holder.app_name == "dev-demo-me"
    assert holder.machine_type == "GPU-H100"
    assert holder.requirements == ["torch"]
    assert holder.host_kwargs["keep_alive"] == 900
    assert holder.host_kwargs["max_concurrency"] == 1
    assert holder.host_kwargs["min_concurrency"] == 0
    assert holder.host_kwargs["health_check_config"] is None
    assert holder.setup is DevAppHolder.setup

    routes = holder(_allow_init=True).collect_routes()
    paths = {signature.path for signature in routes}
    assert "/" not in paths
    assert "/__fal_dev/request" in paths

    overridden = make_holder(
        demo_app.Demo,
        app_name="x",
        session="x",
        token="t",
        keep_alive=1,
        machine_type="GPU-A100",
    )
    assert overridden.machine_type == "GPU-A100"


def test_holder_rejects_bad_token(project):
    import demo_app
    from fastapi import HTTPException

    holder = make_holder(
        demo_app.Demo, app_name="x", session="x", token="secret", keep_alive=1
    )
    instance = holder(_allow_init=True)
    for bad in ("wrong", "", "\u00e9"):
        with pytest.raises(HTTPException) as exc:
            instance._check({"token": bad})
        assert exc.value.status_code == 403
    assert instance._check({"token": "secret", "a": 1})["a"] == 1


def test_forward_refuses_child_control_paths():
    for path in ("/__fal_dev_child/reload", "/%5F_fal_dev_child/reload", "x"):
        with pytest.raises(ValueError):
            _forward({"path": path})


def test_child_process_reload_states(tmp_path):
    child = _ChildProcess(tmp_path)
    assert child.start() is False
    assert child.reload([])["action"] == "not_running"
    child.phase = "setup"
    assert child.reload([])["action"] == "pending"
    assert child.pending_reload


def test_session_names_are_always_dev_boxes():
    assert session_name("my-prod-app", "Demo").startswith(NAME_PREFIX)
    assert session_name("dev-x", "Demo") == "dev-x"
    assert "/" not in session_name("../../etc", "Demo")
    assert session_name(None, "Demo").startswith("dev-demo-")


def test_extract_refuses_escaping_paths(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        data = b"x"
        info = tarfile.TarInfo("../evil.py")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError):
        _extract(base64.b64encode(buffer.getvalue()).decode(), tmp_path / "dest")


def test_sync_roundtrip(tmp_path):
    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "a.py").write_text("a = 1\n")
    (src / "pkg" / "b.py").write_text("b = 1\n")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"x")

    files = app_sync.collect(src)
    assert sorted(files) == ["a.py", "pkg/b.py"]
    first = app_sync.manifest(files)

    (src / "a.py").write_text("a = 2\n")
    (src / "pkg" / "b.py").unlink()
    (src / "c.py").write_text("c = 1\n")
    files = app_sync.collect(src)
    changed, deleted = app_sync.diff(first, app_sync.manifest(files))
    assert changed == ["a.py", "c.py"]
    assert deleted == ["pkg/b.py"]

    written = _extract(app_sync.pack(files, changed), tmp_path / "dest")
    assert sorted(written) == ["a.py", "c.py"]
    assert (tmp_path / "dest" / "a.py").read_text() == "a = 2\n"


def test_sync_sends_symlinks_and_skips_env(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("a = 1\n")
    (src / ".env").write_text("SECRET=1\n")
    os.symlink(src / "a.py", src / "b.py")
    (tmp_path / "outside.py").write_text("x = 1\n")
    os.symlink(tmp_path / "outside.py", src / "c.py")
    os.symlink(src / ".env", src / "config.txt")
    files = app_sync.collect(src)
    assert sorted(files) == ["a.py", "b.py"]
    written = _extract(app_sync.pack(files, sorted(files)), tmp_path / "dest")
    assert sorted(written) == ["a.py", "b.py"]
    assert (tmp_path / "dest" / "b.py").read_text() == "a = 1\n"


def test_sdk_files_ship_the_child():
    files = app_sync.sdk_files()
    assert "fal/dev/remote/child.py" in files
    assert "fal/app.py" in files
    assert not any("__pycache__" in name for name in files)


def test_parser_registers_commands():
    args = parse_args(
        ["dev", "app", "start", "app.py::Demo", "--machine-type", "GPU-H100"]
    )
    assert args.app_ref == ("app.py", "Demo")
    assert args.machine_type == "GPU-H100"
    args = parse_args(["dev", "app", "sync", "--watch", "--request", "/"])
    assert args.watch and args.request == "/"


def test_setup_fingerprint_ignores_endpoint_only_constants(project):
    import demo_app

    fingerprint = app_reload.setup_fingerprint(demo_app.Demo, project)
    assert any(key.endswith(".MODEL") for key in fingerprint)
    assert not any(key.endswith(".STEPS") for key in fingerprint)
    assert textwrap.dedent("def setup(self):") in "".join(fingerprint.values())


def test_config_object_used_by_setup_is_fingerprinted(tmp_path):
    source = (
        "import dataclasses\nimport fal\n\n\n@dataclasses.dataclass\nclass Cfg:\n"
        "    model: str\n\n\nCONFIG = Cfg(model={model!r})\n\n\n"
        "class Demo(fal.App):\n    def setup(self):\n        self.m = CONFIG.model\n"
    )
    fingerprints = []
    for model in ("A", "B"):
        (tmp_path / "cfg_app.py").write_text(source.format(model=model))
        sys.path.insert(0, str(tmp_path))
        try:
            import cfg_app

            fingerprints.append(app_reload.setup_fingerprint(cfg_app.Demo, tmp_path))
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("cfg_app", None)
    action, reason = app_reload.decide(fingerprints[0], fingerprints[1], {}, {})
    assert action == "restart"
    assert "CONFIG" in reason


def test_child_process_ignores_stale_restart(tmp_path):
    child = _ChildProcess(tmp_path)
    child.module, child.class_name = "app", "App"
    stale = child.generation
    child.stop()
    assert child.start(generation=stale) is False


def test_sync_batches_large_trees(tmp_path, monkeypatch):
    monkeypatch.setattr(app_sync, "BATCH_BYTES", 10)
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).write_text("x = 1\n")  # 6 bytes each
    files = app_sync.collect(tmp_path)
    assert app_sync.batches(files, sorted(files)) == [["a.py"], ["b.py"], ["c.py"]]
    assert app_sync.batches(files, []) == [[]]


def test_stop_box_refuses_non_dev_apps():
    from fal.dev.app.session import DevAppError, Session, stop_box

    session = Session(
        name="prod",
        app_name="prod",
        file_path="app.py",
        root=".",
        module="app",
        class_name="App",
        run_url="https://example.invalid",
        token="t",
    )
    with pytest.raises(DevAppError):
        stop_box(session)


def test_start_refuses_to_take_over_an_unknown_app(monkeypatch, tmp_path):
    from fal.dev.app import session as app_session
    from fal.dev.cli import app as dev_app

    (tmp_path / "app.py").write_text("import fal\n")
    monkeypatch.setattr(dev_app, "box_exists", lambda *a, **k: True)
    monkeypatch.setattr(
        dev_app,
        "find_session",
        lambda name: (_ for _ in ()).throw(app_session.DevAppError("none")),
    )
    deployed = []
    monkeypatch.setattr(dev_app, "deploy_box", lambda *a, **k: deployed.append(1))
    args = parse_args(
        ["dev", "app", "start", f"{tmp_path / 'app.py'}::App", "--team", "t"]
    )
    with pytest.raises(app_session.DevAppError, match="--replace"):
        dev_app.cmd_start(args)
    assert not deployed


def test_sync_handles_file_becoming_folder_and_back(tmp_path):
    from fal.dev.remote.holder import _delete

    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    (src / "data").write_text("file\n")
    files = app_sync.collect(src)
    _extract(app_sync.pack(files, ["data"]), dest)

    (src / "data").unlink()
    (src / "data").mkdir()
    (src / "data" / "x.txt").write_text("x\n")
    files = app_sync.collect(src)
    assert _delete(["data"], dest) == ["data"]
    _extract(app_sync.pack(files, ["data/x.txt"]), dest)
    assert (dest / "data" / "x.txt").read_text() == "x\n"

    assert _delete(["data/x.txt"], dest) == ["data/x.txt"]
    assert not (dest / "data").exists()
