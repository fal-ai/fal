import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dev.app import fakes
from dev.app.fakes import REPO_APP_SOURCE
from fal.dev.account import Account
from fal.dev.app import deploy
from fal.dev.app.session import Session
from fal.dev.errors import DevError

repo = fakes.repo

pytestmark = pytest.mark.usefixtures("dev_account")


def test_stop_box_refuses_non_dev_apps():
    session = Session(
        name="prod",
        file_path="app.py",
        root=".",
        module="app",
        class_name="App",
        run_url="https://example.invalid",
        token="t",
        account=Account("me", True),
    )
    with pytest.raises(DevError):
        deploy.stop_box(session)


def _fake_deploy(monkeypatch):
    from fal.api import deploy

    captured = {}

    def execute(prepared, **kwargs):
        captured["prepared"] = prepared
        routes = prepared.loaded.function.endpoints
        base = "https://fal.run/me/dev-demo"
        return SimpleNamespace(urls={"sync": {r: base + r for r in routes}})

    monkeypatch.setattr(deploy, "execute_prepared_deployment", execute)
    return captured


def test_resolve_app_reads_pyproject_like_fal_deploy(repo):
    from fal.dev.app.deploy import resolve_app

    file_path, class_name, app_data = resolve_app(("demo", None))
    assert file_path == str(repo / "pkg" / "models" / "app.py")
    assert class_name == "Demo"
    assert app_data.team == "someone-else"
    assert app_data.options.host["machine_type"] == "GPU-A100"

    # A file ref does not read pyproject.toml, as with `fal deploy`.
    _, _, plain = resolve_app((file_path, "Demo"))
    assert plain.options.host == {}


def test_package_root_is_above_the_top_level_package(repo):
    from fal.dev.app.deploy import package_root

    assert package_root(repo / "pkg" / "models" / "app.py") == repo
    assert package_root(repo / "pyproject.toml") == repo


def test_deploy_box_matches_fal_deploy_options_except_box_settings(repo, monkeypatch):
    from fal.api.client import SyncServerlessClient
    from fal.dev.app.deploy import deploy_box, resolve_app
    from fal.utils import load_function_from

    captured = _fake_deploy(monkeypatch)
    file_path, class_name, app_data = resolve_app(("demo", None))
    deployed = deploy_box(
        file_path,
        class_name,
        app_data=app_data,
        root=repo,
        name="dev-demo",
        token="t",
        keep_alive=300,
        machine_type=None,
        team="mine",
        host=None,
        environment=None,
    )
    box = captured["prepared"].loaded.function.options

    # What `fal deploy demo` would build, from the same loader.
    _, _, app_data = resolve_app(("demo", None))
    for name in [m for m in sys.modules if m == "pkg" or m.startswith("pkg.")]:
        sys.modules.pop(name, None)
    host = SyncServerlessClient()._create_host(local_file_path=file_path)
    real = load_function_from(
        host, file_path, class_name, options=app_data.options
    ).function.options

    assert box.environment == real.environment
    assert (
        "private=https://pkgs.example.com/simple" in (box.environment["requirements"])
    )
    assert box.host["machine_type"] == real.host["machine_type"] == "GPU-A100"
    assert box.host["keep_alive"] == 300
    assert box.host["max_concurrency"] == 1
    assert box.host["health_check_config"] is None
    assert not {"app_files", "app_files_context_dir"} & set(box.host)
    assert captured["prepared"].loaded.app_auth == "private"
    assert captured["prepared"].app_data.name == "dev-demo"

    # Every App is served on 8080; a pyproject exposed_port is not the box's.
    assert box.gateway["exposed_port"] == 8080

    assert deployed.class_name == "Demo"
    assert deployed.run_url == "https://fal.run/me/dev-demo"
    assert deployed.layout.root == repo
    assert deployed.layout.paths == [".", "pkg/models"]
    assert deployed.layout.workdir == "pkg/models"
    assert deployed.layout.ignore == [r"\.bin$"]


def test_layout_problems_stop_before_anything_is_deployed(repo, monkeypatch):
    from fal.dev.app.deploy import deploy_box, resolve_app

    captured = _fake_deploy(monkeypatch)
    file_path, class_name, app_data = resolve_app(("demo", None))
    with pytest.raises(DevError, match="--root"):
        deploy_box(
            file_path,
            class_name,
            app_data=app_data,
            root=repo / "pkg",
            explicit_root=True,
            name="dev-demo",
            token="t",
            keep_alive=300,
            machine_type=None,
            team=None,
            host=None,
            environment=None,
        )
    assert "prepared" not in captured


def test_machine_type_flag_beats_pyproject(repo, monkeypatch):
    from fal.dev.app.deploy import deploy_box, resolve_app

    captured = _fake_deploy(monkeypatch)
    file_path, class_name, app_data = resolve_app(("demo", None))
    deploy_box(
        file_path,
        class_name,
        app_data=app_data,
        root=repo,
        name="dev-demo",
        token="t",
        keep_alive=300,
        machine_type="GPU-H200",
        team=None,
        host=None,
        environment=None,
    )
    options = captured["prepared"].loaded.function.options
    assert options.host["machine_type"] == "GPU-H200"


def test_layout_mirrors_a_deployed_runner(repo):
    file = repo / "pkg" / "models" / "app.py"
    host = {
        "app_files": ["assets"],
        "app_files_context_dir": str(repo),
        "app_files_ignore": [r"\.bin$"],
    }
    found = deploy.layout(repo, str(file), host, explicit_root=False)
    # /app is the context folder; the app file's folder is the cwd.
    assert found.root == repo
    assert found.paths == [".", "pkg/models"]
    assert found.workdir == "pkg/models"
    assert found.ignore_base == "."
    assert found.ignore == [r"\.bin$"]

    # A context folder above the import root widens the synced tree.
    host["app_files_context_dir"] = "../.."
    found = deploy.layout(repo / "pkg", str(file), host, explicit_root=False)
    assert found.root == repo
    assert found.paths == ["pkg", "pkg/models", "."]
    with pytest.raises(DevError, match="--root"):
        deploy.layout(repo / "pkg", str(file), host, explicit_root=True)

    # ...but never to a home folder or above, widened or not.
    home = {"app_files": ["x"], "app_files_context_dir": str(Path.home())}
    with pytest.raises(DevError, match="narrower"):
        deploy.layout(repo, str(file), home, explicit_root=False)
    for root in (Path.home(), Path("/")):
        with pytest.raises(DevError, match="narrower"):
            deploy.layout(root, str(file), {}, explicit_root=True)

    plain = {"app_files_ignore": [r"\.pyc$"]}
    found = deploy.layout(repo, str(file), plain, explicit_root=False)
    assert (found.root, found.paths, found.workdir) == (repo, ["."], ".")
    # app_files_ignore means nothing without app_files, as on a deploy.
    assert found.ignore == []

    broken = {"app_files": ["x"], "app_files_ignore": ["("]}
    with pytest.raises(DevError, match="app_files_ignore"):
        deploy.layout(repo, str(file), broken, explicit_root=False)


def test_class_app_files_with_a_pyproject_context_folder(repo, monkeypatch):
    """The class's app_files win over pyproject.toml's, but the folder they are
    relative to still comes from pyproject.toml: as on a deploy."""
    from fal.dev.app.deploy import deploy_box, resolve_app

    app = repo / "pkg" / "models" / "app.py"
    app.write_text(
        REPO_APP_SOURCE.replace(
            "    keep_alive = 60\n", "    keep_alive = 60\n    app_files = ['assets']\n"
        )
    )
    _fake_deploy(monkeypatch)
    file_path, class_name, app_data = resolve_app(("demo", None))
    deployed = deploy_box(
        file_path,
        class_name,
        app_data=app_data,
        root=repo,
        name="dev-demo",
        token="t",
        keep_alive=300,
        machine_type=None,
        team=None,
        host=None,
        environment=None,
    )
    assert deployed.layout.workdir == "pkg/models"
    assert deployed.layout.ignore_base == "."


def test_box_imports_the_class_by_the_name_the_module_exposes(repo, monkeypatch):
    from fal.dev.app.deploy import deploy_box, resolve_app

    app = repo / "pkg" / "models" / "app.py"
    app.write_text(REPO_APP_SOURCE + "\nExposed = Demo\ndel Demo\n")
    _fake_deploy(monkeypatch)
    _, _, app_data = resolve_app((str(app), "Exposed"))
    deployed = deploy_box(
        str(app),
        "Exposed",
        app_data=app_data,
        root=repo,
        name="dev-demo",
        token="t",
        keep_alive=300,
        machine_type=None,
        team=None,
        host=None,
        environment=None,
    )
    assert deployed.class_name == "Exposed"


def test_start_refuses_a_box_that_is_already_running_without_touching_the_session(
    monkeypatch, tmp_path
):
    from unittest.mock import MagicMock

    from fal.dev.account import Account
    from fal.dev.app import deploy as app_deploy
    from fal.dev.errors import DevError
    from fal.sdk import RunnerState

    monkeypatch.setattr(app_deploy, "box_exists", lambda *a, **k: True)
    ours = SimpleNamespace(
        account=Account("me", True),
        host=None,
        environment=None,
        file_path=str(tmp_path / "app.py"),
    )
    monkeypatch.setattr(app_deploy, "find_session", lambda name: ours)
    live = MagicMock(state=RunnerState.RUNNING)
    monkeypatch.setattr(
        "fal.api.client.SyncServerlessClient",
        lambda **k: MagicMock(apps=MagicMock(runners=lambda name, **kw: [live])),
    )
    deployed = []
    monkeypatch.setattr(app_deploy, "deploy_box", lambda *a, **k: deployed.append(1))

    with pytest.raises(DevError, match="already running.*fal dev app sync.*--replace"):
        app_deploy.start_box(
            tmp_path / "app.py",
            None,
            None,
            name="dev-x",
            root=tmp_path,
            explicit_root=False,
            replace=False,
            keep_alive=60,
            machine_type=None,
            account=Account("me", True),
            host=None,
            environment=None,
            session_env=[],
        )
    assert not deployed
