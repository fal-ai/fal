import sys
import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import fal
from dev.remote import fakes
from fal.dev.remote import reload as app_reload
from fal.dev.remote.child import DevAppServer

project = fakes.project

pytestmark = pytest.mark.usefixtures("dev_account")


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

        fingerprint, _ = app_reload.analyze_setup(base_app.Child, tmp_path)
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


def test_setup_fingerprint_ignores_endpoint_only_constants(project):
    import demo_app

    fingerprint, _ = app_reload.analyze_setup(demo_app.Demo, project)
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

            fingerprints.append(app_reload.analyze_setup(cfg_app.Demo, tmp_path)[0])
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("cfg_app", None)
    action, reason = app_reload.decide(fingerprints[0], fingerprints[1], {}, {})
    assert action == "restart"
    assert "CONFIG" in reason
