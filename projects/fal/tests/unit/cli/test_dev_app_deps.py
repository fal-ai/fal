import json
import sys
from pathlib import Path

import pytest

from fal.dev import app_deps


def _dist(target: Path, name: str, version: str) -> None:
    info = target / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    )


class FakeInstaller:
    """Records installer commands; --target installs write dist-info folders."""

    def __init__(self, plan: list[dict], pip: bool = True) -> None:
        self.plan = plan
        self.pip = pip
        self.commands: list[list[str]] = []
        self.envs: list[dict] = []

    def installer(self) -> list[str]:
        if self.pip:
            return [sys.executable, "-m", "pip", "install"]
        return ["uv", "pip", "install", "--python", sys.executable]

    def run(self, overlay, command: list[str], log: list[str]) -> str:
        self.commands.append(command)
        self.envs.append(overlay._env())
        if "--report" in command:
            report = Path(command[command.index("--report") + 1])
            report.write_text(json.dumps({"install": self.plan}))
            return ""
        if "--dry-run" in command:
            return "\n".join(
                f" + {item['metadata']['name']}=={item['metadata']['version']}"
                for item in self.plan
            )
        target = Path(command[command.index("--target") + 1])
        for pin in command[command.index("--no-deps") + 1 :]:
            if "==" in pin:
                name, version = pin.split("==")
                _dist(target, name, version)
        return ""


def _overlay(tmp_path, monkeypatch, plan, pip=True):
    fake = FakeInstaller(
        [{"metadata": {"name": n, "version": v}} for n, v in plan], pip=pip
    )
    monkeypatch.setattr(app_deps.Overlay, "_installer", lambda self: fake.installer())
    monkeypatch.setattr(
        app_deps.Overlay,
        "_run",
        lambda self, command, log: fake.run(self, command, log),
    )
    return app_deps.Overlay(tmp_path / "overlay", [tmp_path / "sdk-deps"]), fake


@pytest.mark.parametrize("pip", [True, False])
def test_install_plans_against_the_app_env_then_pins_into_a_layer(
    tmp_path, monkeypatch, pip
):
    overlay, fake = _overlay(
        tmp_path, monkeypatch, [("diffusers", "0.31.0"), ("safetensors", "0.5.0")], pip
    )
    result = overlay.install(["diffusers==0.31.0"], index_url="https://pypi.example")

    plan, install = fake.commands
    assert "--dry-run" in plan
    assert plan[-1] == "diffusers==0.31.0"
    assert install[install.index("--no-deps") + 1 :][-2:] == [
        "diffusers==0.31.0",
        "safetensors==0.5.0",
    ]
    assert "--index-url" in install
    layer = result["layer"]
    assert [p["name"] for p in layer["installed"]] == ["diffusers", "safetensors"]
    assert layer["packages"] == ["diffusers==0.31.0", "safetensors==0.5.0"]
    assert overlay.paths() == [str(overlay.layers_dir / layer["id"])]


def test_install_with_no_deps_skips_the_plan(tmp_path, monkeypatch):
    overlay, fake = _overlay(tmp_path, monkeypatch, [])
    overlay.install(["diffusers==0.31.0"], no_deps=True)
    (install,) = fake.commands
    assert "--dry-run" not in install
    assert install[-1] == "diffusers==0.31.0"


def test_install_that_is_already_satisfied_adds_no_layer(tmp_path, monkeypatch):
    overlay, _ = _overlay(tmp_path, monkeypatch, [])
    result = overlay.install(["httpx"])
    assert result["layer"] is None
    assert overlay.layers() == []
    assert overlay.version == 0


def test_install_refuses_control_plane_changes_unless_forced(tmp_path, monkeypatch):
    overlay, _ = _overlay(tmp_path, monkeypatch, [("pydantic", "1.10.0")])
    with pytest.raises(app_deps.DepsError, match="pydantic"):
        overlay.install(["pydantic<2"])
    assert overlay.install(["pydantic<2"], force=True)["layer"] is not None


def test_install_refuses_installer_flags(tmp_path, monkeypatch):
    overlay, _ = _overlay(tmp_path, monkeypatch, [])
    with pytest.raises(app_deps.DepsError, match="flag"):
        overlay.install(["--target=/"])


def test_installer_env_drops_fal_credentials_and_sees_layers(tmp_path, monkeypatch):
    monkeypatch.setenv("FAL_KEY", "secret")
    overlay, fake = _overlay(tmp_path, monkeypatch, [("six", "1.16.0")])
    first = overlay.install(["six==1.16.0"])["layer"]
    overlay.install(["six==1.16.0"])
    env = fake.envs[-1]
    assert "FAL_KEY" not in env
    paths = env["PYTHONPATH"].split(":")
    # The second plan sees the first layer, ahead of the SDK's dependencies.
    assert paths == [str(overlay.layers_dir / first["id"]), str(tmp_path / "sdk-deps")]


def test_layers_stack_newest_first_and_uninstall_drops_newer_layers_too(
    tmp_path, monkeypatch
):
    overlay, fake = _overlay(tmp_path, monkeypatch, [("six", "1.16.0")])
    old = overlay.install(["six==1.16.0"])["layer"]
    fake.plan = [{"metadata": {"name": "rich", "version": "13.0.0"}}]
    new = overlay.install(["rich==13.0.0"])["layer"]
    assert overlay.paths() == [
        str(overlay.layers_dir / new["id"]),
        str(overlay.layers_dir / old["id"]),
    ]
    assert overlay.status()[0]["id"] == new["id"]

    # rich was planned with six in place, so its layer goes too.
    removed = overlay.uninstall(["Six"])
    assert [layer["id"] for layer in removed] == [old["id"], new["id"]]
    assert not (overlay.layers_dir / old["id"]).exists()
    assert overlay.paths() == []
    with pytest.raises(app_deps.DepsError):
        overlay.uninstall(["six"])

    overlay.install(["rich==13.0.0"])
    version = overlay.version
    assert overlay.reset() == 1
    assert overlay.paths() == []
    assert not overlay.layers_dir.exists()
    # Resetting an empty overlay changes nothing (and restarts nothing).
    assert overlay.reset() == 0
    assert overlay.version == version + 1


def test_uninstall_keeps_older_layers(tmp_path, monkeypatch):
    overlay, fake = _overlay(tmp_path, monkeypatch, [("six", "1.16.0")])
    old = overlay.install(["six==1.16.0"])["layer"]
    fake.plan = [{"metadata": {"name": "rich", "version": "13.0.0"}}]
    overlay.install(["rich==13.0.0"])
    overlay.uninstall(["rich"])
    assert overlay.paths() == [str(overlay.layers_dir / old["id"])]


def test_uv_plan_keeps_url_pins_and_skips_exact_repeats(tmp_path, monkeypatch):
    overlay, fake = _overlay(tmp_path, monkeypatch, [("six", "1.16.0")], pip=False)
    overlay.install(["six==1.16.0"])
    output = (
        " + six==1.16.0\n"
        " + pkg @ https://tok@example.com/pkg-1.0-py3-none-any.whl\n"
        " - six==1.17.0\n"
    )
    monkeypatch.setattr(app_deps.Overlay, "_run", lambda self, command, log: output)
    pins = overlay._plan(fake.installer(), [], ["six==1.16.0", "pkg"], [])
    assert pins == ["pkg @ https://tok@example.com/pkg-1.0-py3-none-any.whl"]


def test_status_redacts_credentials_but_replay_keeps_them(tmp_path, monkeypatch):
    overlay, _ = _overlay(tmp_path, monkeypatch, [("six", "1.16.0")])
    url = "https://user:secret@pypi.example/simple"
    overlay.install(["six==1.16.0"], index_url=url, extra_index_urls=[url])
    (layer,) = overlay.status()
    assert layer["index_url"] == "https://***@pypi.example/simple"
    assert layer["extra_index_urls"] == ["https://***@pypi.example/simple"]
    assert "secret" not in json.dumps(overlay.status())
    (replay,) = overlay.replay()
    assert replay["index_url"] == url


def test_status_reports_the_shadowed_version(tmp_path, monkeypatch):
    overlay, _ = _overlay(tmp_path, monkeypatch, [("pytest", "7.0.0")])
    overlay.install(["pytest==7.0.0"], force=True)
    (layer,) = overlay.status()
    (package,) = layer["installed"]
    assert package["version"] == "7.0.0"
    assert package["base_version"] == pytest.__version__


def test_child_path_puts_layers_ahead_of_the_app_env(tmp_path, monkeypatch):
    from fal.dev.app_holder import _ChildProcess

    child = _ChildProcess(tmp_path)
    fake = FakeInstaller([{"metadata": {"name": "six", "version": "1.16.0"}}])
    monkeypatch.setattr(app_deps.Overlay, "_installer", lambda self: fake.installer())
    monkeypatch.setattr(
        app_deps.Overlay,
        "_run",
        lambda self, command, log: fake.run(self, command, log),
    )
    layer = child.overlay.install(["six==1.16.0"])["layer"]
    paths = child._pythonpath().split(":")
    layer_path = str(child.overlay.layers_dir / layer["id"])
    assert paths[0] == str(child.sdk)
    assert paths[1] == layer_path
    assert child._env()["FAL_DEV_OVERLAY"] == layer_path


def test_holder_deps_restarts_the_app_only_when_asked_and_changed(
    tmp_path, monkeypatch
):
    from fal.dev import app_holder

    child = app_holder._ChildProcess(tmp_path)
    child.phase = "ready"
    starts = []

    def start() -> bool:
        starts.append(1)
        return True

    monkeypatch.setattr(child, "start", start)
    fake = FakeInstaller([{"metadata": {"name": "six", "version": "1.16.0"}}])
    monkeypatch.setattr(app_deps.Overlay, "_installer", lambda self: fake.installer())
    monkeypatch.setattr(
        app_deps.Overlay,
        "_run",
        lambda self, command, log: fake.run(self, command, log),
    )
    monkeypatch.setattr(
        app_holder, "_child_call", lambda *a, **k: (200, {"six": {"version": "1"}})
    )

    result = app_holder._deps(
        child, {"action": "install", "packages": ["six==1.16.0"], "restart": True}
    )
    assert result["restarted"] and starts == [1]
    assert result["layers"][0]["installed"][0]["name"] == "six"

    fake.plan = []
    result = app_holder._deps(
        child, {"action": "install", "packages": ["six==1.16.0"], "restart": True}
    )
    assert not result["restarted"] and starts == [1]

    status = app_holder._deps(child, {"action": "status"})
    assert status["app_versions"] == {"six": {"version": "1"}}
    app_holder._deps(child, {"action": "reset"})
    asked = []

    def child_call(action, payload, timeout):
        asked.append(payload)
        return 200, {}

    monkeypatch.setattr(app_holder, "_child_call", child_call)
    app_holder._deps(child, {"action": "status", "names": ["diffusers"]})
    assert asked == [{"names": ["diffusers"]}]
    with pytest.raises(ValueError):
        app_holder._deps(child, {"action": "explode"})


def test_child_reports_versions_and_processes_pth_files(tmp_path, monkeypatch):
    from fal.dev import app_child

    _dist(tmp_path, "overlaypkg", "9.9.9")
    (tmp_path / "extra.pth").write_text(str(tmp_path / "extra") + "\n")
    (tmp_path / "extra").mkdir()
    monkeypatch.setenv("FAL_DEV_OVERLAY", str(tmp_path))
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    app_child._activate_overlay()
    assert str(tmp_path / "extra") in sys.path
    found = app_child._versions(["overlaypkg", "not-a-package-xyz"])
    assert found["overlaypkg"]["version"] == "9.9.9"
    assert found["not-a-package-xyz"] is None


class FakeBox:
    """The holder's deps route, as far as the CLI sees it."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.replay: list[dict] = []
        self.sent: list[tuple[str, dict]] = []
        self.fail_on = fail_on

    def call(self, _session, action, payload: dict | None = None, **_):
        from fal.dev.app_session import DevAppError

        payload = payload or {}
        self.sent.append((action, payload))
        if action == "status":
            return {"phase": "ready", "module": "demo_app"}
        assert action == "deps"
        layer = None
        if payload["action"] == "install":
            if self.fail_on and self.fail_on in payload["packages"][0]:
                raise DevAppError("deps failed (409): index unreachable")
            packages = payload["packages"]
            if not payload.get("no_deps"):
                packages = [*packages, "safetensors==0.5.0"]
            layer = {
                "id": f"layer-{len(self.replay)}",
                "packages": packages,
                "pre": False,
                "index_url": payload.get("index_url"),
                "extra_index_urls": [],
            }
            self.replay.append(layer)
        elif payload["action"] == "reset":
            self.replay = []
        layers = [
            {
                **entry,
                "requested": entry["packages"],
                "installed": [
                    {"name": pin.split("==")[0], "version": pin.split("==")[1]}
                    for pin in entry["packages"]
                ],
            }
            for entry in reversed(self.replay)
        ]
        return {
            "layer": layer,
            "layers": layers,
            "replay": list(self.replay),
            "removed": 0,
            "restarted": False,
        }


def _cli_session(tmp_path, monkeypatch, box):
    from fal.dev import app_session

    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(app_session, "call", box.call)
    app_session.Session(
        name="dev-demo-me",
        app_name="dev-demo-me",
        file_path="/src/demo_app.py",
        root="/src",
        module="demo_app",
        class_name="Demo",
        run_url="https://fal.run/me/dev-demo-me",
        token="t",
    ).save()
    return app_session


def test_cli_remembers_the_overlay_and_replays_it_on_a_new_runner(
    tmp_path, monkeypatch
):
    from fal.cli import dev_app
    from fal.cli.main import parse_args
    from fal.dev import app_bench

    box = FakeBox()
    app_session = _cli_session(tmp_path, monkeypatch, box)
    args = parse_args(
        ["dev", "app", "deps", "install", "diffusers==0.31.0", "--no-restart"]
        + ["--index-url", "https://u:secret@pypi.example/simple"]
    )
    args.func(args)
    install = next(p for a, p in box.sent if a == "deps")
    assert install["packages"] == ["diffusers==0.31.0"]
    assert install["restart"] is False

    session = app_session.find_session(None)
    assert [layer["packages"] for layer in session.overlay] == [
        ["diffusers==0.31.0", "safetensors==0.5.0"]
    ]
    run = app_bench.record(
        name="x",
        session=session,
        path="/",
        body={},
        result={"runs": [{"elapsed_seconds": 1.0}], "warmup_runs": []},
        warmups=0,
    )
    assert run["overlay"] == ["diffusers==0.31.0", "safetensors==0.5.0"]

    # A new runner: the box has no layers; replay installs them, pinned.
    box.replay = []
    box.sent.clear()
    dev_app._replay_overlay(session)
    installs = [p for a, p in box.sent if a == "deps" and p["action"] == "install"]
    assert len(installs) == 1
    assert installs[0]["no_deps"] is True
    assert installs[0]["packages"] == ["diffusers==0.31.0", "safetensors==0.5.0"]
    assert installs[0]["index_url"] == "https://u:secret@pypi.example/simple"

    # Same runner, overlay already there: nothing is installed again.
    box.sent.clear()
    dev_app._replay_overlay(session)
    assert [p["action"] for a, p in box.sent if a == "deps"] == ["status"]


def test_replay_drops_a_layer_that_no_longer_installs(tmp_path, monkeypatch, capsys):
    from fal.cli import dev_app

    box = FakeBox(fail_on="broken")
    app_session = _cli_session(tmp_path, monkeypatch, box)
    session = app_session.find_session(None)
    session.overlay = [
        {"id": "a", "packages": ["broken @ https://u:tok@x/b.whl"], "index_url": None},
        {"id": "b", "packages": ["six==1.16.0"], "index_url": None},
    ]
    dev_app._replay_overlay(session)
    out = capsys.readouterr().out
    assert "Dropped an overlay layer" in out
    assert "tok" not in out
    assert [layer["packages"] for layer in session.overlay] == [["six==1.16.0"]]


def test_status_json_hides_replay(tmp_path, monkeypatch, capsys):
    from fal.cli.main import parse_args

    box = FakeBox()
    _cli_session(tmp_path, monkeypatch, box)
    box.replay = [
        {"id": "a", "packages": ["six==1.16.0"], "index_url": "https://u:s@x/simple"}
    ]
    args = parse_args(["dev", "app", "deps", "status", "--json"])
    args.func(args)
    out = capsys.readouterr().out
    assert "replay" not in json.loads(out)
