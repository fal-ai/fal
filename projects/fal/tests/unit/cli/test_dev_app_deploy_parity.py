"""`fal dev app` runs an app the way `fal deploy` ships it: pyproject.toml refs,
app_files layouts, package indexes, shared caches and synced trees."""

import base64
import hashlib
import io
import os
import sys
import tarfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from fal.cli.main import parse_args
from fal.dev.account import Account
from fal.dev.app import client, deploy
from fal.dev.app import session as app_session
from fal.dev.app import sync as app_sync
from fal.dev.app.session import Session
from fal.dev.errors import DevError
from fal.dev.remote import process
from fal.dev.remote import tree as app_tree
from fal.dev.remote.objects import Store
from fal.dev.remote.process import ChildProcess, LogBuffer
from fal.dev.remote.tree import SourceTree

pytestmark = pytest.mark.usefixtures("dev_account")

APP_SOURCE = """
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
    (models / "app.py").write_text(APP_SOURCE)
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


def test_sync_applies_app_files_ignore_from_the_context_folder(repo):
    files = app_sync.collect(repo, ignore=[r"\.bin$", r"^assets/"])
    assert "assets/prompt.txt" not in files
    files = app_sync.collect(repo, ignore=[r"^prompt"], ignore_base="assets")
    assert "assets/prompt.txt" not in files
    assert "assets/weights.bin" in files
    # Patterns only apply under their base, as on a deploy.
    assert "pkg/helpers.py" in files


def test_start_ignores_the_pyproject_team(repo, monkeypatch):
    from fal.dev.cli import app as dev_app

    calls = {}

    def deploy_box(file_path, class_name, **kwargs):
        calls.update(kwargs, file_path=file_path)
        return deploy.DeployedBox(
            class_name="Demo",
            run_url="https://fal.run/me/dev-demo",
            layout=deploy.Layout(
                repo, [".", "pkg/models"], "pkg/models", ".", [r"\.bin$"]
            ),
        )

    monkeypatch.setattr(deploy, "deploy_box", deploy_box)
    monkeypatch.setattr(deploy, "box_exists", lambda *a, **k: False)
    monkeypatch.setattr(app_session, "SESSIONS_DIR", repo / ".sessions")
    monkeypatch.setattr(client, "boot", lambda session: {"app": {}})
    args = parse_args(["dev", "app", "start", "demo", "--team", "mine"])
    dev_app.cmd_start(args)

    assert calls["team"] == "mine"
    assert calls["root"] == repo
    assert calls["explicit_root"] is False
    session = app_session.find_session(None)
    assert session.name.startswith("dev-demo-")
    assert session.account == Account("mine", False)
    assert session.module == "pkg.models.app"
    assert session.paths == [".", "pkg/models"]
    assert session.workdir == "pkg/models"
    assert session.ignore == [r"\.bin$"]


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


def test_class_app_files_with_a_pyproject_context_folder(repo, monkeypatch):
    """The class's app_files win over pyproject.toml's, but the folder they are
    relative to still comes from pyproject.toml: as on a deploy."""
    from fal.dev.app.deploy import deploy_box, resolve_app

    app = repo / "pkg" / "models" / "app.py"
    app.write_text(
        APP_SOURCE.replace(
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
    app.write_text(APP_SOURCE + "\nExposed = Demo\ndel Demo\n")
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


def test_holder_refuses_a_bad_layout_without_changing_state(tmp_path):
    child = ChildProcess(tmp_path, LogBuffer())
    with pytest.raises(ValueError, match="refusing"):
        child.start("m", "C", layout={"paths": ["."], "workdir": "/etc"})
    assert (child.module, child.paths, child.workdir) == ("", ["."], ".")


def test_an_index_that_refuses_the_box_is_explained(monkeypatch):
    from fal.dev.cli import app as dev_app

    session = SimpleNamespace(account=Account("mine", False))
    monkeypatch.setattr(dev_app, "_booted", lambda args: session)
    errors = iter(
        [
            DevError(
                "deps failed (400): HTTP status client error (401 Unauthorized) for url"
            ),
            DevError("deps failed (400): no matching distribution for nope"),
            DevError("deps failed (403): Forbidden"),
        ]
    )

    def deps_call(session, payload, **kwargs):
        raise next(errors)

    monkeypatch.setattr(client, "deps_call", deps_call)
    args = parse_args(["dev", "app", "deps", "install", "private-pkg"])
    with pytest.raises(DevError, match="fal_wheels_access") as info:
        dev_app.cmd_deps_install(args)
    assert "--team" in str(info.value)
    assert "runs as team mine" in str(info.value)
    assert "401 Unauthorized" in str(info.value)
    # Other failures, and a refused CLI key, are passed through as they are.
    for _ in range(2):
        with pytest.raises(DevError) as info:
            dev_app.cmd_deps_install(args)
        assert "fal_wheels_access" not in str(info.value)


@pytest.mark.parametrize(
    "message",
    [
        "401 Client Error: Unauthorized for url: https://huggingface.co/api/x",
        "Downloaded 401 packages from https://pypi.example/simple",
        "deploy failed (403): Forbidden",
        "start failed (401): Unauthorized",
    ],
)
def test_other_401s_get_no_index_hint(message):
    assert client.index_hint(message, Account(None, True)) == message


def test_start_runs_as_the_remembered_team_else_your_personal_account(
    repo, monkeypatch, capsys
):
    from fal.dev.cli import app as dev_app

    seen = []
    monkeypatch.setattr(
        deploy,
        "deploy_box",
        lambda *a, **k: deploy.DeployedBox(
            "Demo", "u", deploy.Layout(repo, ["."], ".", ".", [])
        ),
    )
    monkeypatch.setattr(deploy, "box_exists", lambda *a, **k: False)
    monkeypatch.setattr(app_session, "SESSIONS_DIR", repo / ".sessions")
    monkeypatch.setattr(client, "boot", lambda session: {"app": {}})
    monkeypatch.setattr(
        dev_app, "confirm_deploy", lambda *a, **k: seen.append((a[1], k["yes"]))
    )
    start = ["dev", "app", "start", "demo", "--name", "box"]

    dev_app.cmd_start(parse_args(start))
    assert app_session.find_session("box").account == Account("me", True)

    dev_app.cmd_start(parse_args([*start, "--team", "mine", "--replace", "--yes"]))
    dev_app.cmd_start(parse_args([*start, "--replace"]))
    assert app_session.find_session("box").account == Account("mine", False)
    assert seen == [
        (Account("me", True), False),
        (Account("mine", False), True),
        (Account("mine", False), False),
    ]
    assert capsys.readouterr().err.splitlines() == [
        "Team: personal (me)",
        "Team: mine",
        "Team: mine",
    ]


def test_start_explains_an_index_that_refuses_the_environment_build(repo, monkeypatch):
    from fal.dev.cli import app as dev_app

    def refuse(*args, **kwargs):
        raise RuntimeError(
            "Failed to build the environment: error: HTTP status client error "
            "(401 Unauthorized) for url (https://wheels.example/simple/pkg/)"
        )

    monkeypatch.setattr(deploy, "start_box", refuse)
    args = parse_args(["dev", "app", "start", "demo", "--team", "mine"])
    with pytest.raises(DevError, match="(?s)fal_wheels_access.*runs as team mine"):
        dev_app.cmd_start(args)
    args = parse_args(["dev", "app", "start", "demo", "--name", "other"])
    with pytest.raises(DevError, match="runs as your personal account"):
        dev_app.cmd_start(args)


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


def test_start_saves_session_env(repo, monkeypatch):
    from fal.dev.cli import app as dev_app

    monkeypatch.setattr(
        deploy,
        "deploy_box",
        lambda *a, **k: deploy.DeployedBox(
            "Demo", "u", deploy.Layout(repo, ["."], ".", ".", [])
        ),
    )
    monkeypatch.setattr(deploy, "box_exists", lambda *a, **k: False)
    monkeypatch.setattr(app_session, "SESSIONS_DIR", repo / ".sessions")
    monkeypatch.setattr(client, "boot", lambda session: {"app": {}})
    args = parse_args(
        ["dev", "app", "start", "demo", "--team", "t", "--session-env", "MY_NS"]
    )
    dev_app.cmd_start(args)
    assert app_session.find_session(None).session_env == ["MY_NS"]

    # A name the box would refuse stops the start before anything deploys.
    deployed = []
    monkeypatch.setattr(deploy, "deploy_box", lambda *a, **k: deployed.append(1))
    args = parse_args(
        ["dev", "app", "start", "demo", "--team", "t", "--session-env", "LD_PRELOAD"]
    )
    with pytest.raises(DevError, match="cannot set"):
        dev_app.cmd_start(args)
    assert not deployed


@pytest.fixture
def roomy(monkeypatch):
    """The store's free-space floor, off: test disks are small."""
    from fal.dev.remote import objects as app_objects

    monkeypatch.setattr(app_objects, "MIN_FREE_BYTES", 0)


def test_object_store_keeps_checked_copies(tmp_path, roomy, monkeypatch):
    from fal.dev.remote import objects as app_objects

    assert Store.open(tmp_path / "missing") is None
    store = Store.open(tmp_path)
    assert store is not None
    source = tmp_path / "a.py"
    source.write_text("print(1)\n")
    digest = store.put(source.read_bytes())
    assert digest == hashlib.sha256(b"print(1)\n").hexdigest()

    target = tmp_path / "src" / "pkg" / "a.py"
    assert store.get(digest, target, executable=True)
    assert target.read_text() == "print(1)\n"
    assert target.stat().st_mode & 0o111

    # A damaged object is never copied out, and is dropped so the next put
    # repairs it.
    store.path(digest).write_text("tampered")
    assert not store.get(digest, tmp_path / "other.py")
    assert not (tmp_path / "other.py").exists()
    assert not store.path(digest).exists()
    assert store.put(source.read_bytes()) == digest
    assert store.get(digest, tmp_path / "other.py")

    # Links planted in the store are never followed.
    planted = tmp_path / "planted.py"
    planted.write_text("print(2)\n")
    planted_digest = hashlib.sha256(b"print(2)\n").hexdigest()
    store.path(planted_digest).parent.mkdir(exist_ok=True)
    store.path(planted_digest).symlink_to(planted)
    assert not store.get(planted_digest, tmp_path / "x.py")

    # A FIFO planted in the store neither hangs nor gets copied.
    fifo_digest = "f" * 64
    store.path(fifo_digest).parent.mkdir(exist_ok=True)
    os.mkfifo(store.path(fifo_digest))
    assert not store.get(fifo_digest, tmp_path / "fifo.py")

    # Nothing bigger than the sync's file limit is copied out.
    big = b"x" * (app_objects.MAX_FILE_BYTES + 1)
    big_digest = hashlib.sha256(big).hexdigest()
    store.path(big_digest).parent.mkdir(exist_ok=True)
    store.path(big_digest).write_bytes(big)
    assert not store.get(big_digest, tmp_path / "big.bin")
    assert not (tmp_path / "big.bin").exists()

    # A nearly full volume is left alone.
    monkeypatch.setattr(app_objects, "MIN_FREE_BYTES", 1 << 62)
    assert store.put(planted.read_bytes()) is None
    with pytest.raises(ValueError):
        store.path("../../etc/passwd")


def _age(path: Path, days: float) -> None:
    import time

    then = time.time() - days * 86400
    os.utime(path, (then, then), follow_symlinks=False)


def test_object_store_expires_objects_nobody_uses(tmp_path, roomy):
    store = Store.open(tmp_path)
    assert store is not None
    digests = {}
    for name in ("old", "used", "kept", "fresh"):
        source = tmp_path / f"{name}.py"
        source.write_text(name)
        digests[name] = store.put(source.read_bytes())
    for name in ("old", "used", "kept"):
        _age(store.path(digests[name]), 31)
    # Copying an object out, or finding it already stored, counts as use.
    assert store.get(digests["used"], tmp_path / "out" / "used.py")
    assert store.put((tmp_path / "kept.py").read_bytes()) == digests["kept"]
    # Half-written copies go much sooner.
    tmp_copy = store.path(digests["fresh"]).parent / ".tmp-abc"
    tmp_copy.write_text("x")
    _age(tmp_copy, 0.1)
    young = store.path(digests["fresh"]).parent / ".tmp-new"
    young.write_text("x")

    store.expire()

    assert not store.path(digests["old"]).exists()
    for name in ("used", "kept", "fresh"):
        assert store.path(digests[name]).exists()
    assert not tmp_copy.exists() and young.exists()


def test_object_store_expiry_does_not_follow_planted_folders(tmp_path, roomy):
    store = Store.open(tmp_path / "data") or Store.open(tmp_path)
    assert store is not None
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "precious").write_text("keep")
    _age(victim / "precious", 400)
    (store.root / "ab").symlink_to(victim)
    store.expire()
    assert (victim / "precious").exists()


def _synced(tmp_path, roomy_store=True):
    (tmp_path / "data").mkdir()
    store = Store.open(tmp_path / "data")
    assert store is not None
    return store


def test_tree_places_stored_files_and_hands_synced_ones_to_the_store(
    tmp_path, roomy, monkeypatch
):
    store = _synced(tmp_path)
    kept = []
    monkeypatch.setattr(
        app_tree,
        "Keeper",
        lambda store, log: SimpleNamespace(keep=kept.append),
    )
    tree = SourceTree(tmp_path / "session" / "src", store, print)
    local = tmp_path / "local"
    (local / "pkg").mkdir(parents=True)
    (local / "pkg" / "a.py").write_text("A = 1\n")
    files = app_sync.collect(local)
    digest = app_sync.manifest(files)["pkg/a.py"]

    # Nothing stored yet: nothing is placed.
    assert tree.place({"pkg/a.py": [digest, False]}) == []

    # A synced file is handed to the store's keeper...
    written, _ = tree.sync(archive=app_sync.pack(files, ["pkg/a.py"]), deleted=[])
    assert kept == [b"A = 1\n"]
    store.put(kept[0])
    # ...so a fresh runner can copy it into place.
    fresh = SourceTree(tmp_path / "fresh" / "src", store, print)
    assert fresh.place({"pkg/a.py": [digest, False]}) == ["pkg/a.py"]
    assert (fresh.root / "pkg" / "a.py").read_text() == "A = 1\n"
    assert "pkg/a.py" in fresh.synced
    with pytest.raises(ValueError, match="refusing"):
        fresh.place({"../x": [digest, 0]})
    with pytest.raises(ValueError, match="digest"):
        fresh.place({"x.py": ["../../etc", 0]})

    # No store (no /data volume): nothing placed, everything is uploaded.
    assert SourceTree(tmp_path / "bare", None, print).place({"a.py": [digest, 0]}) == []


def _session(tmp_path, *names) -> Session:
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


def _uploaded(archive: str) -> list[str]:
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(archive))) as tar:
        return sorted(tar.getnames())


def test_calls_to_the_box_switch_request_io_storage_off(tmp_path, monkeypatch):
    import httpx

    import fal.sdk

    session = _session(tmp_path, "a.py")
    sent = {}

    def post(url, **kwargs):
        sent.update(kwargs)
        return httpx.Response(200, json={})

    monkeypatch.setattr("fal.dev.transport.httpx.post", post)
    monkeypatch.setattr(
        fal.sdk,
        "get_credentials",
        lambda team: SimpleNamespace(to_headers=lambda: {"Authorization": "Key k"}),
    )
    app_session.call(session, "status")

    assert sent["headers"]["X-Fal-Store-IO"] == "0"
    assert sent["json"]["token"] == "t"


@pytest.mark.parametrize(
    "account, key, team, acts, outcome",
    [
        (Account("me", True), None, "red", True, "--team red is ignored"),
        (Account("me", True), None, "ME", True, ""),
        (
            Account("acme", False, via_key=True),
            Account("ACME", False, True),
            None,
            True,
            "",
        ),
        (Account("acme", False, via_key=True), None, None, True, "set FAL_KEY"),
        (
            Account("acme", False, via_key=True),
            Account("other", False, True),
            None,
            True,
            "belongs to 'other'",
        ),
        (
            Account(None, False, via_key=True),
            Account("any", False, True),
            None,
            True,
            "",
        ),
        # Looking at or stopping a box is never refused.
        (Account("acme", False, via_key=True), None, None, False, ""),
        (
            Account("acme", False, via_key=True),
            Account("other", False, True),
            None,
            False,
            "",
        ),
    ],
)
def test_a_started_box_keeps_its_account(
    tmp_path, monkeypatch, capsys, account, key, team, acts, outcome
):
    from fal.dev.cli import app as dev_app

    session = _session(tmp_path, "a.py")
    session.account = account
    monkeypatch.setattr(dev_app, "find_session", lambda name: session)
    monkeypatch.setattr(dev_app, "key_account", lambda: key)
    args = SimpleNamespace(name=None, team=team)

    if outcome in ("set FAL_KEY", "belongs to 'other'"):
        with pytest.raises(DevError, match=outcome):
            dev_app._session(args, acts_as_account=acts)
    else:
        assert dev_app._session(args, acts_as_account=acts) is session
        assert outcome in capsys.readouterr().err


def test_sync_uploads_only_what_the_store_lacks(tmp_path, monkeypatch):
    session = _session(tmp_path, "a.py", "b.py", "c.py")
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path / "sessions")
    calls = []

    def call(session, action, payload=None, *, timeout=60):
        calls.append((action, payload))
        if action == "place":
            return {"placed": sorted(set(payload["files"]) - {"c.py"})}
        return {}

    monkeypatch.setattr(client, "call", call)
    result = client.sync_files(session, full=True, reload=False)

    assert result.placed == 2
    (files_call,) = (p for action, p in calls if action == "files")
    assert _uploaded(files_call["archive"]) == ["c.py"]
    assert files_call["all_files"] == ["a.py", "b.py", "c.py"]
    assert session.manifest.keys() == {"a.py", "b.py", "c.py"}

    # A small edit is uploaded directly: no extra round trip to the store.
    calls.clear()
    (tmp_path / "tree" / "a.py").write_text("edited")
    result = client.sync_files(session)
    assert [action for action, _ in calls] == ["files"]
    assert _uploaded(calls[0][1]["archive"]) == ["a.py"]
    assert result.changed == ["a.py"] and result.placed == 0


def test_a_big_incremental_sync_asks_the_store(tmp_path, monkeypatch):
    session = _session(tmp_path, "a.py", "b.py", "c.py")
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path / "sessions")
    calls = []

    def call(session, action, payload=None, *, timeout=60):
        calls.append(action)
        return {"placed": []} if action == "place" else {}

    monkeypatch.setattr(client, "call", call)
    client.sync_files(session, reload=False)
    calls.clear()
    monkeypatch.setattr(client, "PLACE_MIN_FILES", 1)
    for name in ("a.py", "b.py"):
        (tmp_path / "tree" / name).write_text("edit " + name)
    client.sync_files(session, reload=False)
    assert calls == ["place", "files"]
    # Bytes count too.
    calls.clear()
    monkeypatch.setattr(client, "PLACE_MIN_FILES", 100)
    monkeypatch.setattr(client, "PLACE_MIN_BYTES", 5)
    (tmp_path / "tree" / "c.py").write_text("a longer edit")
    client.sync_files(session, reload=False)
    assert calls == ["place", "files"]


def test_sync_falls_back_to_uploading(tmp_path, monkeypatch):
    """No store, a failed place, or a damaged object: those files are sent."""
    session = _session(tmp_path, "a.py", "b.py", "c.py")
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(client, "MAX_PLACE_FILES", 1)
    places = []

    def call(session, action, payload=None, *, timeout=60):
        if action == "place":
            places.append(sorted(payload["files"]))
            (rel,) = payload["files"]
            if rel == "b.py":
                raise DevError("place failed (400): disk error")
            return {"placed": [] if rel == "c.py" else [rel]}
        return {}

    monkeypatch.setattr(client, "call", call)
    result = client.sync_files(session, full=True, reload=False)
    # One file per place request, as MAX_PLACE_FILES says.
    assert places == [["a.py"], ["b.py"], ["c.py"]]
    assert result.placed == 1
    assert result.changed == ["a.py", "b.py", "c.py"]

    # A box without a store places nothing, and gets the whole tree.
    monkeypatch.setattr(
        client, "call", lambda s, a, p=None, **k: {"placed": []} if a == "place" else {}
    )
    result = client.sync_files(session, full=True, reload=False)
    assert result.placed == 0


def test_sync_announces_withheld_files_once_per_change(tmp_path, monkeypatch, capsys):
    session = _session(tmp_path, "a.py", "server.pem")
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(client, "call", lambda *a, **k: {"placed": []})

    def announced() -> bool:
        return "server.pem" in capsys.readouterr().out

    client.sync_files(session, reload=False)
    assert announced()
    client.sync_files(session, reload=False)
    assert not announced()
    (tmp_path / "tree" / "tls.key").write_text("k")
    client.sync_files(session, reload=False)
    assert announced()
    # A full sync (a new runner) says it again.
    client.sync_files(session, full=True, reload=False)
    assert announced()


def test_a_sync_that_needs_a_rebuild_is_repeated(tmp_path, monkeypatch):
    session = _session(tmp_path, "a.py")
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path / "sessions")

    def call(session, action, payload=None, *, timeout=60):
        if action == "place":
            return {"placed": []}
        return {"reload": {"action": "rebuild", "reason": "requirements changed"}}

    monkeypatch.setattr(client, "call", call)
    client.sync_files(session)
    assert session.manifest == {}


def test_boot_ships_the_sdk_and_waits_for_setup(tmp_path, monkeypatch):
    session = _session(tmp_path, "a.py")
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path / "sessions")
    sent = {}

    def call(session, action, payload=None, *, timeout=60):
        sent[action] = payload
        if action == "logs":
            return {"lines": [], "total": 0}
        if action == "status":
            return {"phase": "ready", "detail": "", "module": "a"}
        return {"placed": []}

    monkeypatch.setattr(client, "call", call)
    assert client.boot(session)["phase"] == "ready"
    start = sent["start"]
    assert (start["module"], start["class_name"]) == ("a", "A")
    assert start["layout"] == {"paths": ["."], "workdir": "."}
    sdk = _uploaded(start["sdk_archive"])
    assert "fal/dev/remote/child.py" in sdk and "fal/app.py" in sdk
    assert not any("__pycache__" in name for name in sdk)
    # The shipped SDK's own dependencies, minus optional extras.
    assert start["sdk_requirements"]
    assert not any("extra ==" in req for req in start["sdk_requirements"])


def test_a_box_on_a_new_runner_is_booted_again(tmp_path, monkeypatch):
    session = _session(tmp_path, "a.py")
    booted = []
    monkeypatch.setattr(client, "boot", lambda session: booted.append(1))
    status = {"phase": "idle", "module": ""}
    monkeypatch.setattr(client, "call", lambda *a, **k: status)
    assert client.ensure_booted(session) is True
    status["module"] = "a"
    assert client.ensure_booted(session) is False
    assert booted == [1]


def test_sync_never_sends_keys_or_credentials(tmp_path):
    for name in ("app.py", "server.pem", "tls.key", "credentials.json", "id_ed25519"):
        (tmp_path / name).write_text("x")
    for name in (
        "conf/.npmrc",
        ".aws/credentials",
        "certs/ca.p8",
        ".SSH/config",
        "deploy/kubeconfig",
        "conf/secrets.yaml",
    ):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text("x")
    # Not even through a link with an innocent name.
    (tmp_path / "ok.txt").symlink_to(tmp_path / "server.pem")
    withheld: list[str] = []
    (tmp_path / "settings").mkdir()
    (tmp_path / "settings" / "secrets.py").write_text("x")
    assert sorted(app_sync.collect(tmp_path, withheld=withheld)) == [
        "app.py",
        "settings/secrets.py",
    ]
    assert "server.pem" in withheld
    assert ".aws/credentials" in withheld


def test_session_env_never_overrides_what_the_sdk_reads():
    """Every FAL_ variable the SDK reads is refused, so new ones can't slip by."""
    import re

    import fal
    from fal.dev.remote.protocol import check_session_env

    names = set()
    for path in Path(fal.__file__).parent.rglob("*.py"):
        for line in path.read_text(errors="ignore").splitlines():
            if any(read in line for read in ("environ", "getenv", "bool_envvar(")):
                names.update(re.findall(r"[\"'](FAL_[A-Z0-9_]+)[\"']", line))
    assert names
    for name in sorted(names):
        with pytest.raises(DevError, match="cannot set"):
            check_session_env([name])


def test_object_store_refuses_oversized_content(tmp_path, roomy):
    (tmp_path / "data").mkdir()
    store = Store.open(tmp_path / "data")
    assert store is not None
    assert store.put(b"x" * (10 * 1024 * 1024 + 1)) is None
    assert not list(store.root.glob("*/*"))


def test_start_with_a_bad_layout_is_a_400_and_leaves_the_app_running(
    tmp_path, monkeypatch
):
    import asyncio

    from fastapi import HTTPException

    from fal.dev.remote.holder import DevAppHolder

    holder = object.__new__(DevAppHolder)
    holder._fal_dev_token = "t"
    holder._log_buffer = LogBuffer()
    holder._child = ChildProcess(tmp_path, holder._log_buffer)
    holder._child.phase = "ready"
    swapped = []
    monkeypatch.setattr(holder._child, "replace_sdk", swapped.append)
    for bad in (
        {"layout": "nope"},
        {"session_env": "LD_PRELOAD"},
        {"module": 5},
        {"class_name": "-c"},
    ):
        with pytest.raises(HTTPException) as info:
            asyncio.run(
                holder.dev_start(
                    {"token": "t", "sdk_archive": "x", "module": "m", **bad}
                )
            )
        assert info.value.status_code == 400
    assert swapped == [] and holder._child.phase == "ready"

    # A bug in a handler is a 400 too, and is logged.
    def broken(payload):
        raise KeyError("boom")

    with pytest.raises(HTTPException) as info:
        asyncio.run(holder._run({"token": "t"}, broken))
    assert info.value.status_code == 400
    assert "KeyError" in "\n".join(holder._log_buffer.lines)


def test_files_reload_includes_files_the_box_placed(tmp_path):
    import asyncio

    from fal.dev.remote.holder import DevAppHolder

    reloaded = []
    holder = object.__new__(DevAppHolder)
    holder._fal_dev_token = "t"
    holder._log_buffer = LogBuffer()
    holder._placed = set()
    holder._placed_lock = threading.Lock()
    holder._tree = SimpleNamespace(
        place=lambda files: ["a.py"],
        sync=lambda **kwargs: (["b.py"], ["c.py"]),
    )
    answers = [{"action": "error"}, {"action": "reload"}, {"action": "reload"}]
    holder._child = SimpleNamespace(
        reload=lambda files: reloaded.append(files) or answers.pop(0)
    )
    asyncio.run(holder.dev_place({"token": "t", "files": {"a.py": ["0" * 64, False]}}))
    for _ in range(3):
        asyncio.run(holder.dev_files({"token": "t", "reload": True}))
    # A failed reload keeps the placed file for the next one.
    assert reloaded == [["b.py", "c.py", "a.py"]] * 2 + [["b.py", "c.py"]]


def test_an_old_session_file_is_called_out(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path)
    (tmp_path / "dev-old.json").write_text('{"name": "dev-old", "app_name": "x"}')
    assert app_session.load_sessions() == []
    err = capsys.readouterr().err
    assert "dev-old.json" in err and "fal apps delete dev-old" in err
    assert err.count("\n") == 1


def test_a_bad_sdk_archive_leaves_the_old_sdk_in_place(tmp_path):
    child = ChildProcess(tmp_path, LogBuffer())
    child.sdk.mkdir(parents=True)
    (child.sdk / "marker").write_text("old")
    with pytest.raises(Exception):
        child.replace_sdk(base64.b64encode(b"not a tarball").decode())
    assert (child.sdk / "marker").read_text() == "old"
    assert not (tmp_path / "sdk.new").exists()


def test_files_written_before_an_archive_fails_are_recorded(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("a")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in ("a.py", "../evil.py"):
            info = tarfile.TarInfo(name)
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
    tree = SourceTree(tmp_path / "dest", None, print)
    with pytest.raises(ValueError):
        tree.sync(archive=base64.b64encode(buffer.getvalue()).decode(), deleted=[])
    assert tree.synced == {"a.py"}


def test_layout_paths_must_be_a_list_of_strings(tmp_path):
    child = ChildProcess(tmp_path, LogBuffer())
    with pytest.raises(ValueError, match="list of strings"):
        child.check_start({"paths": "pkg"}, None)


def test_holder_logs_hide_credentials(tmp_path):
    import asyncio

    from fastapi import HTTPException

    from fal.dev.remote.holder import DevAppHolder

    holder = object.__new__(DevAppHolder)
    holder._fal_dev_token = "t"
    holder._log_buffer = LogBuffer()

    def broken(payload):
        raise KeyError("https://user:secret@pypi.example/simple")

    with pytest.raises(HTTPException):
        asyncio.run(holder._run({"token": "t"}, broken))
    assert "secret" not in "\n".join(holder._log_buffer.lines)


def test_overlay_install_shares_one_deadline(tmp_path, monkeypatch):
    from fal.dev.remote import deps as app_deps

    monkeypatch.setattr(
        app_deps.Overlay, "_installer", lambda self: ["uv", "pip", "install"]
    )
    timeouts = []

    def run(command, **kwargs):
        timeouts.append(kwargs["timeout"])
        if "--target" in command:
            Path(command[command.index("--target") + 1]).mkdir()
        return SimpleNamespace(returncode=0, stdout=" + six==1.0", stderr="")

    monkeypatch.setattr(app_deps.subprocess, "run", run)
    app_deps.Overlay(tmp_path / "o").install(["six"])
    plan, build = timeouts
    assert build <= plan <= app_deps.INSTALL_TIMEOUT_SECONDS


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
