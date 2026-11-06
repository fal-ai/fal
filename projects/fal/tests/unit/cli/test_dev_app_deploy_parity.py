"""`fal dev app` runs an app the way `fal deploy` ships it: pyproject.toml refs,
app_files layouts, package indexes, shared caches and synced trees."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fal.cli.main import parse_args
from fal.dev.app import sync as app_sync

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
    from fal.dev.app.session import resolve_app

    file_path, class_name, app_data = resolve_app(("demo", None))
    assert file_path == str(repo / "pkg" / "models" / "app.py")
    assert class_name == "Demo"
    assert app_data.team == "someone-else"
    assert app_data.options.host["machine_type"] == "GPU-A100"

    # A file ref does not read pyproject.toml, as with `fal deploy`.
    _, _, plain = resolve_app((file_path, "Demo"))
    assert plain.options.host == {}


def test_package_root_is_above_the_top_level_package(repo):
    from fal.dev.app.session import package_root

    assert package_root(repo / "pkg" / "models" / "app.py") == repo
    assert package_root(repo / "pyproject.toml") == repo


def test_deploy_box_matches_fal_deploy_options_except_box_settings(repo, monkeypatch):
    from fal.api.client import SyncServerlessClient
    from fal.dev.app.session import deploy_box, resolve_app
    from fal.utils import load_function_from

    captured = _fake_deploy(monkeypatch)
    file_path, class_name, app_data = resolve_app(("demo", None))
    deployed = deploy_box(
        file_path,
        class_name,
        app_data=app_data,
        root=repo,
        app_name="dev-demo",
        session="dev-demo",
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
    from fal.dev.app.session import DevAppError, deploy_box, resolve_app

    captured = _fake_deploy(monkeypatch)
    file_path, class_name, app_data = resolve_app(("demo", None))
    with pytest.raises(DevAppError, match="--root"):
        deploy_box(
            file_path,
            class_name,
            app_data=app_data,
            root=repo / "pkg",
            explicit_root=True,
            app_name="dev-demo",
            session="dev-demo",
            token="t",
            keep_alive=300,
            machine_type=None,
            team=None,
            host=None,
            environment=None,
        )
    assert "prepared" not in captured


def test_machine_type_flag_beats_pyproject(repo, monkeypatch):
    from fal.dev.app.session import deploy_box, resolve_app

    captured = _fake_deploy(monkeypatch)
    file_path, class_name, app_data = resolve_app(("demo", None))
    deploy_box(
        file_path,
        class_name,
        app_data=app_data,
        root=repo,
        app_name="dev-demo",
        session="dev-demo",
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
    from fal.dev.app.session import AppFiles, DevAppError, layout

    file = repo / "pkg" / "models" / "app.py"
    app_files = AppFiles(["assets"], [r"\.bin$"], repo, file.parent)
    found = layout(repo, app_files, explicit_root=False)
    # /app is the context folder; the app file's folder is the cwd.
    assert found.root == repo
    assert found.paths == [".", "pkg/models"]
    assert found.workdir == "pkg/models"
    assert found.ignore_base == "."
    assert found.ignore == [r"\.bin$"]

    # A context folder above the import root widens the synced tree.
    found = layout(repo / "pkg", app_files, explicit_root=False)
    assert found.root == repo
    assert found.paths == ["pkg", "pkg/models", "."]
    with pytest.raises(DevAppError, match="--root"):
        layout(repo / "pkg", app_files, explicit_root=True)

    # ...but never to a home folder or above, widened or not.
    home = AppFiles(["x"], [], Path.home(), file.parent)
    with pytest.raises(DevAppError, match="narrower"):
        layout(repo, home, explicit_root=False)
    plain = AppFiles([], [], file.parent, file.parent)
    for root in (Path.home(), Path("/")):
        with pytest.raises(DevAppError, match="narrower"):
            layout(root, plain, explicit_root=True)

    plain = AppFiles([], [r"\.pyc$"], file.parent, file.parent)
    found = layout(repo, plain, explicit_root=False)
    assert (found.root, found.paths, found.workdir) == (repo, ["."], ".")
    # app_files_ignore means nothing without app_files, as on a deploy.
    assert found.ignore == []

    broken = AppFiles(["x"], ["("], repo, file.parent)
    with pytest.raises(DevAppError, match="app_files_ignore"):
        layout(repo, broken, explicit_root=False)


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
    from fal.dev.app import session as app_session
    from fal.dev.app.session import DeployedBox, Layout

    calls = {}

    def deploy(file_path, class_name, **kwargs):
        calls.update(kwargs, file_path=file_path)
        return DeployedBox(
            class_name="Demo",
            run_url="https://fal.run/me/dev-demo",
            layout=Layout(repo, [".", "pkg/models"], "pkg/models", ".", [r"\.bin$"]),
        )

    monkeypatch.setattr(app_session, "deploy_box", deploy)
    monkeypatch.setattr(app_session, "box_exists", lambda *a, **k: False)
    monkeypatch.setattr(app_session, "SESSIONS_DIR", repo / ".sessions")
    monkeypatch.setattr(dev_app, "_boot", lambda session: {"app": {}})
    args = parse_args(["dev", "app", "start", "demo", "--team", "mine"])
    dev_app._start(args)

    assert calls["team"] == "mine"
    assert app_session.find_session(None).name.startswith("dev-demo-")
    assert calls["root"] == repo
    assert calls["explicit_root"] is False
    session = app_session.find_session(None)
    assert session.module == "pkg.models.app"
    assert session.paths == [".", "pkg/models"]
    assert session.workdir == "pkg/models"
    assert session.ignore == [r"\.bin$"]


def test_child_layout_is_passed_to_the_app_process(tmp_path, monkeypatch):
    from fal.dev.remote import holder as app_holder

    child = app_holder._ChildProcess(tmp_path)
    # No background launch or log pump threads.
    monkeypatch.setattr(
        app_holder.threading, "Thread", lambda **kwargs: SimpleNamespace(start=dict)
    )
    with pytest.raises(ValueError, match="refusing"):
        child.start("m", "C", layout={"paths": ["../outside"], "workdir": "."})

    launched = []
    monkeypatch.setattr(child, "_install_sdk_deps", lambda generation: None)
    monkeypatch.setattr(child, "_wait_ready", lambda proc: None)
    monkeypatch.setattr(
        app_holder.subprocess,
        "Popen",
        lambda command, **kwargs: launched.append((command, kwargs)),
    )
    child.module, child.class_name = "pkg.models.app", "Demo"
    child.paths, child.workdir = [".", "pkg/models"], "pkg/models"
    child._launch_once(child.generation)
    command, _ = launched[0]
    assert "--cwd=pkg/models" in command
    assert [arg for arg in command if arg.startswith("--path=")] == [
        "--path=.",
        "--path=pkg/models",
    ]


def test_class_app_files_with_a_pyproject_context_folder(repo, monkeypatch):
    """The class's app_files win over pyproject.toml's, but the folder they are
    relative to still comes from pyproject.toml: as on a deploy."""
    from fal.dev.app.session import deploy_box, resolve_app

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
        app_name="dev-demo",
        session="dev-demo",
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
    from fal.dev.app.session import deploy_box

    app = repo / "pkg" / "models" / "app.py"
    app.write_text(APP_SOURCE + "\nExposed = Demo\ndel Demo\n")
    _fake_deploy(monkeypatch)
    deployed = deploy_box(
        str(app),
        "Exposed",
        root=repo,
        app_name="dev-demo",
        session="dev-demo",
        token="t",
        keep_alive=300,
        machine_type=None,
        team=None,
        host=None,
        environment=None,
    )
    assert deployed.class_name == "Exposed"


def test_holder_refuses_a_bad_layout_without_changing_state(tmp_path):
    from fal.dev.remote import holder as app_holder

    child = app_holder._ChildProcess(tmp_path)
    with pytest.raises(ValueError, match="refusing"):
        child.start("m", "C", layout={"paths": ["."], "workdir": "/etc"})
    assert (child.module, child.paths, child.workdir) == ("", ["."], ".")


def test_deps_install_explains_an_index_that_refuses_the_box(monkeypatch):
    from fal.dev.cli import app as dev_app
    from fal.dev.app.session import DevAppError

    monkeypatch.setattr(dev_app, "_session", lambda args: None)
    monkeypatch.setattr(dev_app, "_ensure_booted", lambda session: None)
    errors = iter(
        [
            "deps failed (409): pip: 401 Unauthorized for https://pkgs.example.com",
            "deps failed (409): no matching distribution for nope",
            "deps failed (403): Forbidden",
        ]
    )

    def deps_call(session, payload, **kwargs):
        raise DevAppError(next(errors))

    monkeypatch.setattr(dev_app, "_deps_call", deps_call)
    args = parse_args(["dev", "app", "deps", "install", "private-pkg"])
    with pytest.raises(DevAppError, match="builds an app's environment") as info:
        dev_app._deps_install(args)
    assert "401 Unauthorized" in str(info.value)
    # Other failures, and a refused CLI key, are passed through as they are.
    for _ in range(2):
        with pytest.raises(DevAppError) as info:
            dev_app._deps_install(args)
        assert "builds an app's environment" not in str(info.value)


def test_app_process_gets_the_session_name(tmp_path, monkeypatch):
    from fal.dev.remote import holder as app_holder

    child = app_holder._ChildProcess(tmp_path / "dev-tiny-image-me")
    assert child._env()["FAL_DEV_SESSION"] == "dev-tiny-image-me"

    # Apps that read another name for it get that one too.
    monkeypatch.setattr(
        app_holder.threading, "Thread", lambda **kwargs: SimpleNamespace(start=dict)
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
        with pytest.raises(ValueError, match="cannot set"):
            child.start("m", "C", session_env=[bad])
    assert child.session_env == ["MY_CACHE_NAMESPACE", "FAL_MY_CACHE_NS"]


def test_start_saves_session_env(repo, monkeypatch):
    from fal.dev.cli import app as dev_app
    from fal.dev.app import session as app_session
    from fal.dev.app.session import DeployedBox, Layout

    monkeypatch.setattr(
        app_session,
        "deploy_box",
        lambda *a, **k: DeployedBox("Demo", "u", Layout(repo, ["."], ".", ".", [])),
    )
    monkeypatch.setattr(app_session, "box_exists", lambda *a, **k: False)
    monkeypatch.setattr(app_session, "SESSIONS_DIR", repo / ".sessions")
    monkeypatch.setattr(dev_app, "_boot", lambda session: {"app": {}})
    args = parse_args(
        ["dev", "app", "start", "demo", "--team", "t", "--session-env", "MY_NS"]
    )
    dev_app._start(args)
    assert app_session.find_session(None).session_env == ["MY_NS"]

    # A name the box would refuse stops the start before anything deploys.
    deployed = []
    monkeypatch.setattr(app_session, "deploy_box", lambda *a, **k: deployed.append(1))
    args = parse_args(
        ["dev", "app", "start", "demo", "--team", "t", "--session-env", "LD_PRELOAD"]
    )
    with pytest.raises(ValueError, match="cannot set"):
        dev_app._start(args)
    assert not deployed


@pytest.fixture
def roomy(monkeypatch):
    """The store's free-space floor, off: test disks are small."""
    from fal.dev.remote import objects as app_objects

    monkeypatch.setattr(app_objects, "MIN_FREE_BYTES", 0)


def test_object_store_keeps_checked_copies(tmp_path, roomy, monkeypatch):
    import hashlib

    from fal.dev.remote import objects as app_objects
    from fal.dev.remote.objects import Store

    assert Store.open(tmp_path / "missing") is None
    store = Store.open(tmp_path)
    assert store is not None
    source = tmp_path / "a.py"
    source.write_text("print(1)\n")
    digest = store.put(source)
    assert digest == hashlib.sha256(b"print(1)\n").hexdigest()
    assert store.missing([digest, "0" * 64]) == ["0" * 64]

    target = tmp_path / "src" / "pkg" / "a.py"
    assert store.get(digest, target, executable=True)
    assert target.read_text() == "print(1)\n"
    assert target.stat().st_mode & 0o111

    # A damaged object counts as missing, is never copied out, and is dropped
    # so the next put repairs it.
    store.path(digest).write_text("tampered")
    assert not store.get(digest, tmp_path / "other.py")
    assert not (tmp_path / "other.py").exists()
    assert store.missing([digest]) == [digest]
    assert store.put(source) == digest
    assert store.get(digest, tmp_path / "other.py")

    # Links planted in the store are never followed.
    planted = tmp_path / "planted.py"
    planted.write_text("print(2)\n")
    planted_digest = hashlib.sha256(b"print(2)\n").hexdigest()
    store.path(planted_digest).parent.mkdir(exist_ok=True)
    store.path(planted_digest).symlink_to(planted)
    assert store.missing([planted_digest]) == [planted_digest]
    assert not store.get(planted_digest, tmp_path / "x.py")

    # A FIFO planted in the store neither hangs nor gets copied.
    fifo_digest = "f" * 64
    store.path(fifo_digest).parent.mkdir(exist_ok=True)
    os.mkfifo(store.path(fifo_digest))
    assert not store.get(fifo_digest, tmp_path / "fifo.py")

    # Nothing bigger than the sync's file limit is copied out.
    big = b"x" * (app_objects.MAX_OBJECT_BYTES + 1)
    big_digest = hashlib.sha256(big).hexdigest()
    store.path(big_digest).parent.mkdir(exist_ok=True)
    store.path(big_digest).write_bytes(big)
    assert not store.get(big_digest, tmp_path / "big.bin")
    assert not (tmp_path / "big.bin").exists()

    # A nearly full volume is left alone.
    monkeypatch.setattr(app_objects, "MIN_FREE_BYTES", 1 << 62)
    assert store.put(planted) is None
    with pytest.raises(ValueError):
        store.path("../../etc/passwd")


def test_holder_places_stored_files_and_stores_synced_ones(tmp_path, roomy):
    from fal.dev.remote import holder as app_holder
    from fal.dev.app import sync as app_sync
    from fal.dev.remote.objects import Store

    (tmp_path / "data").mkdir()
    store = Store.open(tmp_path / "data")
    assert store is not None
    child = app_holder._ChildProcess(tmp_path / "session")
    child.store = store
    local = tmp_path / "local"
    (local / "pkg").mkdir(parents=True)
    (local / "pkg" / "a.py").write_text("A = 1\n")
    files = app_sync.collect(local)
    manifest = app_sync.manifest(files)

    # Nothing stored yet: everything is missing, nothing placed.
    found = app_holder._objects(
        child, {"action": "missing", "digests": list(manifest.values())}
    )
    assert found == {"enabled": True, "missing": sorted(manifest.values())}

    # A synced file is queued for the store, and the worker keeps it...
    archive = app_sync.pack(files, ["pkg/a.py"])
    child.remember(app_holder._extract(archive, child.src))
    store.put(child.keep.get_nowait())
    assert child.keep.empty()
    # ...so a fresh runner can copy it into place.
    fresh = app_holder._ChildProcess(tmp_path / "fresh")
    fresh.store = store
    placed = app_holder._objects(
        fresh,
        {"action": "place", "files": {"pkg/a.py": [manifest["pkg/a.py"], False]}},
    )
    assert placed["placed"] == ["pkg/a.py"]
    assert (fresh.src / "pkg" / "a.py").read_text() == "A = 1\n"
    assert "pkg/a.py" in fresh.synced
    with pytest.raises(ValueError, match="refusing"):
        app_holder._objects(
            fresh, {"action": "place", "files": {"../x": [manifest["pkg/a.py"], 0]}}
        )

    child.store = None
    assert app_holder._objects(child, {"action": "missing"}) == {"enabled": False}


def test_boot_uploads_only_what_the_store_lacks(tmp_path, monkeypatch):
    from fal.dev.cli import app as dev_app
    from fal.dev.app import session as app_session
    from fal.dev.app.session import Session

    root = tmp_path / "tree"
    root.mkdir()
    for name in ("a.py", "b.py", "c.py"):
        (root / name).write_text(name)
    session = Session(
        name="dev-x",
        app_name="dev-x",
        file_path=str(root / "a.py"),
        root=str(root),
        module="a",
        class_name="A",
        run_url="https://fal.run/me/dev-x",
        token="t",
    )
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path / "sessions")
    calls = []

    def call(session, action, payload=None, *, timeout=60):
        calls.append((action, payload))
        if action == "objects" and payload["action"] == "missing":
            from fal.dev.app import sync as app_sync

            digests = app_sync.manifest(app_sync.collect(root))
            return {"enabled": True, "missing": [digests["c.py"]]}
        if action == "objects":
            return {"enabled": True, "placed": sorted(payload["files"]), "missing": []}
        return {}

    monkeypatch.setattr(app_session, "call", call)
    result = dev_app._sync(session, full=True, reload=False, from_store=True)
    assert result["placed"] == 2
    assert result["changed"] == ["c.py"]
    files_call = [p for action, p in calls if action == "files"][0]
    assert files_call["full"] is True
    assert files_call["all_files"] == ["a.py", "b.py", "c.py"]
    assert session.manifest.keys() == {"a.py", "b.py", "c.py"}

    # A box without the store (or an older one) gets the whole tree.
    calls.clear()

    def old_box(session, action, payload=None, *, timeout=60):
        if action == "objects":
            raise app_session.DevAppError("objects failed (404): Not Found")
        return {}

    monkeypatch.setattr(app_session, "call", old_box)
    result = dev_app._sync(session, full=True, reload=False, from_store=True)
    assert result["changed"] == ["a.py", "b.py", "c.py"]


def test_boot_falls_back_to_uploading(tmp_path, monkeypatch):
    """No store, a failed place, or a damaged object: those files are sent."""
    from fal.dev.cli import app as dev_app
    from fal.dev.app import session as app_session
    from fal.dev.app import sync as app_sync
    from fal.dev.app.session import DevAppError, Session

    root = tmp_path / "tree"
    root.mkdir()
    for name in ("a.py", "b.py", "c.py"):
        (root / name).write_text(name)
    session = Session(
        "dev-x", "dev-x", str(root / "a.py"), str(root), "a", "A", "u", "t"
    )
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(dev_app, "PLACE_BATCH", 1)
    places = []

    def call(session, action, payload=None, *, timeout=60):
        if action == "objects" and payload["action"] == "missing":
            return {"enabled": True, "missing": []}
        if action == "objects":
            places.append(sorted(payload["files"]))
            (rel,) = payload["files"]
            if rel == "b.py":
                raise DevAppError("objects failed (400): disk error")
            if rel == "c.py":
                return {"enabled": True, "placed": [], "missing": ["c.py"]}
            return {"enabled": True, "placed": [rel], "missing": []}
        return {}

    monkeypatch.setattr(app_session, "call", call)
    result = dev_app._sync(session, full=True, reload=False, from_store=True)
    # One file per place request, as PLACE_BATCH says.
    assert places == [["a.py"], ["b.py"], ["c.py"]]
    assert result["placed"] == 1
    assert result["changed"] == ["b.py", "c.py"]

    def no_store(session, action, payload=None, *, timeout=60):
        if action == "objects":
            return {"enabled": False}
        return {}

    monkeypatch.setattr(app_session, "call", no_store)
    result = dev_app._sync(session, full=True, reload=False, from_store=True)
    assert result["placed"] is None
    assert result["changed"] == sorted(app_sync.collect(root))


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
    from fal.dev.remote.holder import check_session_env

    names = set()
    for path in Path(fal.__file__).parent.rglob("*.py"):
        for line in path.read_text(errors="ignore").splitlines():
            if "environ" in line or "getenv" in line:
                names.update(re.findall(r"[\"'](FAL_[A-Z0-9_]+)[\"']", line))
    assert names
    for name in sorted(names):
        with pytest.raises(ValueError, match="cannot set"):
            check_session_env([name])
