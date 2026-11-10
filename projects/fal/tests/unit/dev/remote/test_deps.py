from pathlib import Path
from types import SimpleNamespace

import pytest

from dev.remote.fakes import FakeInstaller
from fal.dev.errors import DevError
from fal.dev.remote import deps as app_deps
from fal.dev.remote.process import ChildProcess, LogBuffer

pytestmark = pytest.mark.usefixtures("dev_account")


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


def test_overlay_record_is_private(tmp_path, monkeypatch):
    overlay, _ = _overlay(tmp_path, monkeypatch, ["six==1.16.0"])
    overlay.install(["six"], index_url="https://u:secret@pypi.example/simple")
    assert overlay.state_file.stat().st_mode & 0o777 == 0o600


def test_a_runner_without_uv_gets_one_from_pip(tmp_path, monkeypatch):
    monkeypatch.setattr(app_deps.shutil, "which", lambda name: None)
    monkeypatch.setenv("PIP_INDEX_URL", "https://elsewhere.example/simple")
    monkeypatch.setenv("DEVBOX_ADMIN_KEY", "admin")
    commands, envs = [], []

    def run(command, **kwargs):
        commands.append(command)
        if "pip" in command and "uv" in command:
            uv = Path(command[command.index("--target") + 1]) / "bin" / "uv"
            envs.append(kwargs["env"])
            uv.parent.mkdir(parents=True)
            uv.write_text("")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(app_deps.subprocess, "run", run)
    overlay = app_deps.Overlay(tmp_path / "overlay")
    overlay._deadline = app_deps.time.monotonic() + 60

    installer = overlay._installer()
    assert installer[0] == str(tmp_path / "overlay" / "uv" / "bin" / "uv")
    assert commands[0][1:3] == ["-m", "pip"] and "--isolated" in commands[0]
    assert "PIP_INDEX_URL" not in envs[0] and "DEVBOX_ADMIN_KEY" not in envs[0]
    # Kept for later installs: pip is not run again.
    overlay._installer()
    assert len(commands) == 1


def test_a_failed_uv_bootstrap_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(app_deps.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        app_deps.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    overlay = app_deps.Overlay(tmp_path / "overlay")
    overlay._deadline = app_deps.time.monotonic() + 60

    with pytest.raises(DevError, match="could not install uv"):
        overlay._installer()
