import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fal.cli.main import parse_args
from fal.dev.app import client
from fal.dev.app import session as app_session
from fal.dev.errors import DevError
from fal.dev.remote import deps as app_deps
from fal.dev.remote.holder import DevAppHolder
from fal.dev.remote.process import ChildProcess, LogBuffer


def _dist(target: Path, name: str, version: str) -> None:
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
                _dist(target, name, version)
        return ""


def _overlay(tmp_path, monkeypatch, plan):
    fake = FakeInstaller(plan)
    monkeypatch.setattr(
        app_deps.Overlay, "_installer", lambda self: ["uv", "pip", "install"]
    )
    monkeypatch.setattr(app_deps.Overlay, "_run", lambda self, c: fake.run(c))
    return app_deps.Overlay(tmp_path / "overlay"), fake


def test_install_plans_against_the_app_env_then_pins_into_the_overlay(
    tmp_path, monkeypatch
):
    overlay, fake = _overlay(
        tmp_path, monkeypatch, ["diffusers==0.31.0", "safetensors==0.5.0"]
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
    assert result["pins"] == ["diffusers==0.31.0", "safetensors==0.5.0"]
    assert [p["name"] for p in overlay.packages()] == ["diffusers", "safetensors"]
    assert overlay.state()["packages"] == result["pins"]


def test_install_with_no_deps_skips_the_plan(tmp_path, monkeypatch):
    overlay, fake = _overlay(tmp_path, monkeypatch, [])
    overlay.install(["diffusers==0.31.0"], no_deps=True)
    (install,) = fake.commands
    assert "--dry-run" not in install
    assert install[-1] == "diffusers==0.31.0"


def test_install_that_is_already_satisfied_changes_nothing(tmp_path, monkeypatch):
    overlay, _ = _overlay(tmp_path, monkeypatch, [])
    assert overlay.install(["httpx"])["pins"] == []
    assert overlay.state() == {"packages": []}
    assert overlay.version == 0

    overlay, fake = _overlay(tmp_path, monkeypatch, ["six==1.16.0"])
    overlay.install(["six"])
    fake.commands.clear()
    # The same pin again is not installed again.
    assert overlay.install(["six"])["pins"] == []
    assert len(fake.commands) == 1


def test_install_refuses_control_plane_changes_unless_forced(tmp_path, monkeypatch):
    overlay, _ = _overlay(tmp_path, monkeypatch, ["pydantic==1.10.0"])
    with pytest.raises(DevError, match="pydantic"):
        overlay.install(["pydantic<2"])
    assert overlay.install(["pydantic<2"], force=True)["pins"] == ["pydantic==1.10.0"]


def test_install_refuses_installer_flags(tmp_path, monkeypatch):
    overlay, _ = _overlay(tmp_path, monkeypatch, [])
    with pytest.raises(DevError, match="flag"):
        overlay.install(["--target=/"])


def test_installer_env_drops_fal_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("FAL_KEY", "secret")
    monkeypatch.setattr(
        app_deps.Overlay, "_installer", lambda self: ["uv", "pip", "install"]
    )
    envs = []

    def run(command, **kwargs):
        envs.append(kwargs["env"])
        if "--target" in command:
            Path(command[command.index("--target") + 1]).mkdir()
        return SimpleNamespace(returncode=0, stdout=" + six==1.16.0", stderr="")

    monkeypatch.setattr(app_deps.subprocess, "run", run)
    app_deps.Overlay(tmp_path / "overlay").install(["six"])
    assert envs and all("FAL_KEY" not in env for env in envs)


def test_installer_errors_hide_index_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(
        app_deps.Overlay, "_installer", lambda self: ["uv", "pip", "install"]
    )
    monkeypatch.setattr(
        app_deps.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="401 Unauthorized for https://user:secret@pypi.example/simple",
        ),
    )
    with pytest.raises(DevError) as info:
        app_deps.Overlay(tmp_path / "overlay").install(["six"])
    assert "401 Unauthorized" in str(info.value)
    assert "secret" not in str(info.value)


def test_installs_accumulate_and_a_newer_pin_replaces_an_older_one(
    tmp_path, monkeypatch
):
    overlay, fake = _overlay(tmp_path, monkeypatch, ["six==1.16.0"])
    overlay.install(["six"], index_url="https://one.example", pre=True)
    fake.plan = ["rich==13.0.0", "six==1.17.0"]
    overlay.install(["rich"], extra_index_urls=["https://two.example"])

    state = overlay.state()
    assert state["packages"] == ["rich==13.0.0", "six==1.17.0"]
    assert state["pre"] is True
    assert state["index_url"] == "https://one.example"
    assert state["extra_index_urls"] == ["https://two.example"]
    # The overlay is rebuilt from every pin, so it holds exactly these.
    install = fake.commands[-1]
    assert install[install.index("--no-deps") + 1 :][-2:] == [
        "rich==13.0.0",
        "six==1.17.0",
    ]
    assert [(p["name"], p["version"]) for p in overlay.packages()] == [
        ("rich", "13.0.0"),
        ("six", "1.17.0"),
    ]


def test_uninstall_and_reset(tmp_path, monkeypatch):
    overlay, fake = _overlay(tmp_path, monkeypatch, ["six==1.16.0", "rich==13.0.0"])
    overlay.install(["rich"])
    assert overlay.uninstall(["Six"]) == ["six==1.16.0"]
    assert [p["name"] for p in overlay.packages()] == ["rich"]
    with pytest.raises(DevError, match="six"):
        overlay.uninstall(["six"])

    version = overlay.version
    assert overlay.reset() == 1
    assert overlay.packages() == []
    assert not overlay.site.exists()
    # Resetting an empty overlay changes nothing (and restarts nothing).
    assert overlay.reset() == 0
    assert overlay.version == version + 1


def test_uv_plan_keeps_url_pins(tmp_path, monkeypatch):
    output = (
        " + six==1.16.0\n"
        " + pkg @ https://tok@example.com/pkg-1.0-py3-none-any.whl\n"
        " - six==1.17.0\n"
    )
    overlay, fake = _overlay(tmp_path, monkeypatch, [])
    monkeypatch.setattr(
        app_deps.Overlay,
        "_run",
        lambda self, command: output if "--dry-run" in command else fake.run(command),
    )
    pins = overlay.install(["six==1.16.0", "pkg"])["pins"]
    assert pins == [
        "six==1.16.0",
        "pkg @ https://tok@example.com/pkg-1.0-py3-none-any.whl",
    ]


def test_packages_report_the_shadowed_version(tmp_path, monkeypatch):
    overlay, _ = _overlay(tmp_path, monkeypatch, ["pytest==7.0.0"])
    overlay.install(["pytest==7.0.0"], force=True)
    (package,) = overlay.packages()
    assert package["version"] == "7.0.0"
    assert package["base_version"] == pytest.__version__


def test_app_process_path_puts_the_overlay_ahead_of_the_app_env(tmp_path):
    child = ChildProcess(tmp_path, LogBuffer())
    paths = child._env()["PYTHONPATH"].split(":")
    assert paths[:2] == [str(child.sdk), str(child.overlay.site)]
    assert child._env()["FAL_DEV_OVERLAY"] == str(child.overlay.site)


def _holder(tmp_path, monkeypatch):
    holder = object.__new__(DevAppHolder)
    holder._fal_dev_token = "t"
    holder._child = ChildProcess(tmp_path, LogBuffer())
    holder._child.phase = "ready"
    fake = FakeInstaller(["six==1.16.0"])
    monkeypatch.setattr(
        app_deps.Overlay, "_installer", lambda self: ["uv", "pip", "install"]
    )
    monkeypatch.setattr(app_deps.Overlay, "_run", lambda self, c: fake.run(c))

    def deps(**payload):
        return asyncio.run(holder.dev_deps({"token": "t", **payload}))

    return holder, fake, deps


def test_deps_route_restarts_the_app_only_when_asked_and_changed(tmp_path, monkeypatch):
    from fal.dev.remote import holder as holder_module

    holder, fake, deps = _holder(tmp_path, monkeypatch)
    starts = []
    monkeypatch.setattr(holder._child, "start", lambda: starts.append(1) or True)
    monkeypatch.setattr(
        holder_module, "child_call", lambda *a, **k: (200, {"six": {"version": "1"}})
    )

    result = deps(action="install", packages=["six==1.16.0"], restart=True)
    assert result["restarted"] and starts == [1]
    assert result["packages"][0]["name"] == "six"
    assert result["replay"]["packages"] == ["six==1.16.0"]

    fake.plan = []
    result = deps(action="install", packages=["six==1.16.0"], restart=True)
    assert not result["restarted"] and starts == [1]

    status = deps(action="status")
    assert status["app_versions"] == {"six": {"version": "1"}}
    assert deps(action="reset")["removed"] == 1
    with pytest.raises(Exception, match="unknown deps action"):
        deps(action="explode")


def test_child_reports_versions_and_processes_pth_files(tmp_path, monkeypatch):
    from fal.dev.remote import child as app_child

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


def _cli_session(tmp_path, monkeypatch, box):
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
    ).save()


def test_cli_remembers_the_overlay_and_replays_it_on_a_new_runner(
    tmp_path, monkeypatch, capsys
):
    from fal.dev.app import bench as app_bench

    box = FakeBox()
    _cli_session(tmp_path, monkeypatch, box)
    args = parse_args(
        ["dev", "app", "deps", "install", "diffusers==0.31.0", "--no-restart"]
        + ["--index-url", "https://u:secret@pypi.example/simple"]
    )
    args.func(args)
    install = next(p for a, p in box.sent if a == "deps")
    assert install["packages"] == ["diffusers==0.31.0"]
    assert install["restart"] is False
    assert "Installed diffusers==0.31.0, safetensors==0.5.0" in capsys.readouterr().out

    session = app_session.find_session(None)
    assert session.overlay["packages"] == ["diffusers==0.31.0", "safetensors==0.5.0"]
    run = app_bench.record(
        name="x",
        session=session,
        path="/",
        body={},
        result={"runs": [{"elapsed_seconds": 1.0}], "warmup_runs": []},
        warmups=0,
    )
    assert run["overlay"] == ["diffusers==0.31.0", "safetensors==0.5.0"]

    # A new runner: the box has no overlay; replay installs it, pinned.
    box.state = {"packages": []}
    box.sent.clear()
    client.replay_overlay(session)
    installs = [p for a, p in box.sent if a == "deps" and p["action"] == "install"]
    assert len(installs) == 1
    assert installs[0]["no_deps"] is True
    assert installs[0]["packages"] == ["diffusers==0.31.0", "safetensors==0.5.0"]
    assert installs[0]["index_url"] == "https://u:secret@pypi.example/simple"

    # Same runner, overlay already there: nothing is installed again.
    box.sent.clear()
    client.replay_overlay(session)
    assert [p["action"] for a, p in box.sent if a == "deps"] == ["status"]


def test_replay_drops_an_overlay_that_no_longer_installs(tmp_path, monkeypatch, capsys):
    box = FakeBox(fail_on="broken")
    _cli_session(tmp_path, monkeypatch, box)
    session = app_session.find_session(None)
    session.overlay = {"packages": ["broken @ https://u:tok@x/b.whl"]}
    client.replay_overlay(session)
    out = capsys.readouterr().out
    assert "Could not install the dependency overlay" in out
    assert "tok" not in out
    # Kept, so the next boot tries again.
    assert session.overlay == {"packages": ["broken @ https://u:tok@x/b.whl"]}


def test_replay_failure_from_a_refusing_index_explains_the_grant(
    tmp_path, monkeypatch, capsys
):
    refusal = "deps failed (400): HTTP status client error (401 Unauthorized)"
    box = FakeBox(fail_on="private", error=refusal)
    _cli_session(tmp_path, monkeypatch, box)
    session = app_session.find_session(None)
    session.overlay = {"packages": ["private-pkg==1"]}
    client.replay_overlay(session)
    out = " ".join(capsys.readouterr().out.split())
    assert "fal_wheels_access" in out and "--team" in out


def test_status_json_hides_replay(tmp_path, monkeypatch, capsys):
    box = FakeBox()
    _cli_session(tmp_path, monkeypatch, box)
    box.state = {"packages": ["six==1.16.0"], "index_url": "https://u:s@x/simple"}
    args = parse_args(["dev", "app", "deps", "status", "--json"])
    args.func(args)
    out = capsys.readouterr().out
    assert "replay" not in json.loads(out)
    assert "u:s@" not in out


def test_deps_lines_are_plain_data(tmp_path, monkeypatch, capsys):
    box = FakeBox()
    _cli_session(tmp_path, monkeypatch, box)
    box.state = {"packages": ["six==1.16.0"]}
    args = parse_args(["dev", "app", "deps", "status"])
    args.func(args)
    assert "  six==1 (new)\n" in capsys.readouterr().out


def test_redact_hides_credentials_and_keeps_refs():
    from fal.dev.remote.protocol import redact

    for url in (
        "https://user:ab/cd@host.example/simple",
        "https://user:p@ss@host.example/simple",
        "https://tok@host.example/simple",
    ):
        assert redact(f"failed for {url} twice") == (
            "failed for https://***@host.example/simple twice"
        )
    assert redact("http://u:tok@pypi:8080/simple") == "http://***@pypi:8080/simple"
    assert redact("see http://tok@pypi/simple.") == "see http://***@pypi/simple."
    assert redact("git+https://github.com/org/repo.git@v1.2.3") == (
        "git+https://github.com/org/repo.git@v1.2.3"
    )
    assert redact("http://pypi:8080/o.git@v1") == "http://pypi:8080/o.git@v1"
    # Only the userinfo goes; a git ref stays readable.
    assert redact("git+https://tok@github.com/o/r.git@v1") == (
        "git+https://***@github.com/o/r.git@v1"
    )
    assert redact("git+https://github.com/o/r.git@v1") == (
        "git+https://github.com/o/r.git@v1"
    )
    assert redact("https://h.example/x?a=1&token=abc&Password=p%20q") == (
        "https://h.example/x?a=1&token=***&Password=***"
    )


def test_overlay_record_is_private(tmp_path, monkeypatch):
    overlay, _ = _overlay(tmp_path, monkeypatch, ["six==1.16.0"])
    overlay.install(["six"], index_url="https://u:secret@pypi.example/simple")
    assert overlay.state_file.stat().st_mode & 0o777 == 0o600


def test_a_failed_replay_survives_status_reads_and_is_retried(tmp_path, monkeypatch):
    box = FakeBox(fail_on="broken")
    _cli_session(tmp_path, monkeypatch, box)
    session = app_session.find_session(None)
    session.overlay = {"packages": ["broken==1"]}
    session.save()
    client.replay_overlay(session)
    args = parse_args(["dev", "app", "deps", "status"])
    args.func(args)
    assert app_session.find_session(None).overlay == {"packages": ["broken==1"]}

    # The box recovers: the next boot installs the original pins.
    box.fail_on = None
    box.sent.clear()
    client.replay_overlay(app_session.find_session(None))
    (install,) = (p for a, p in box.sent if a == "deps" and p["action"] == "install")
    assert install["packages"] == ["broken==1"]
