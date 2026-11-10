"""devbox-hot.sh run for real, with a fake `uv` and a fake server on PATH."""

import os
import stat
import subprocess
import threading
import time
from pathlib import Path

import pytest

from fal.dev.box import REMOTE

SCRIPT = (REMOTE / "scripts" / "devbox-hot.sh").read_text()


@pytest.fixture
def rig(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "uv-calls.txt"
    # `uv venv` makes a .venv with a python; `uv pip install` records its arguments
    # and the constraints file it was given.
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/bin/bash\n"
        f'echo "$@" >> {calls}\n'
        'if [ "$1" = venv ]; then\n'
        "  mkdir -p .venv/bin\n"
        '  printf "#!/bin/bash\\necho venv-python\\n" > .venv/bin/python\n'
        "  chmod +x .venv/bin/python\n"
        'elif [ "$1" = pip ]; then\n'
        "  [ -e fail-uv ] && exit 1\n"
        '  while [ $# -gt 0 ]; do [ "$1" = -c ] && cat "$2" >> '
        f"{tmp_path}/constraints-seen.txt; shift; done\n"
        "fi\n"
    )
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    # The base interpreter reports the base torch stack; the script's constraint
    # generator runs under `python3`.
    server = tmp_path / "server.sh"
    server.write_text("#!/bin/bash\nexit 0\n")
    script = tmp_path / "devbox-hot"
    script.write_text(
        SCRIPT.replace("$SERVER", str(server))
        .replace("$PORT", "1")
        .replace("$RESTART_CODE", "3")
    )
    project = tmp_path / "project"
    project.mkdir()
    # Installed distributions the constraint generator sees, via PYTHONPATH.
    site = tmp_path / "site"
    for name, version in [
        ("torch", "2.8.0+cu128"),
        ("Torchvision", "0.23.0+cu128"),
        ("nvidia_cudnn_cu12", "9.10.2.21"),
        ("unrelated", "1.0"),
    ]:
        info = site / f"{name}-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        )
    constraints = tmp_path / "constraints.txt"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PYTHONPATH": str(site),
        "DEVBOX_HOT_CONSTRAINTS": str(constraints),
    }

    def run():
        # The server exits 0, which ends the script's loop.
        return subprocess.run(
            ["bash", str(script), str(project)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

    return project, calls, constraints, run


def test_requirements_install_is_constrained_to_the_base_torch_stack(rig):
    project, calls, seen, run = rig
    (project / "requirements.txt").write_text("diffusers\n")

    done = run()

    assert done.returncode == 0, done.stderr
    install = calls.read_text().splitlines()[-1]
    assert f"-c {seen} -r requirements.txt" in install
    pins = seen.read_text().split()
    # Public versions only (the +cu128 build tag is dropped); other packages are free.
    assert {
        "torch==2.8.0",
        "torchvision==0.23.0",
        "nvidia-cudnn-cu12==9.10.2.21",
    } <= set(pins)
    assert not any(pin.startswith("unrelated") for pin in pins)


def test_a_changed_requirements_file_rebuilds_the_venv(rig):
    project, calls, _, run = rig
    (project / "requirements.txt").write_text("diffusers\n")
    run()
    (project / ".venv" / "stale-marker").write_text("x")

    (project / "requirements.txt").write_text("diffusers\npeft\n")
    done = run()

    assert "requirements.txt changed; rebuilding .venv" in done.stdout
    assert not (project / ".venv" / "stale-marker").exists()
    assert calls.read_text().count("venv -q") == 2


def test_an_unchanged_requirements_file_keeps_the_venv(rig):
    project, calls, _, run = rig
    (project / "requirements.txt").write_text("diffusers\n")
    run()
    (project / ".venv" / "keep").write_text("x")

    done = run()

    assert "rebuilding" not in done.stdout
    assert (project / ".venv" / "keep").exists()


@pytest.mark.parametrize("emptied", ["", "# nothing yet\n\n"])
def test_an_emptied_requirements_file_drops_the_venv(rig, emptied):
    project, calls, _, run = rig
    (project / "requirements.txt").write_text("diffusers\n")
    run()
    assert (project / ".venv").exists()

    (project / "requirements.txt").write_text(emptied)
    done = run()

    assert "rebuilding .venv" in done.stdout
    assert not (project / ".venv").exists()


def test_a_venv_the_script_did_not_make_is_left_alone(rig):
    project, calls, _, run = rig
    (project / ".venv" / "bin").mkdir(parents=True)
    (project / ".venv" / "bin" / "python").write_text("#!/bin/bash\n")
    (project / ".venv" / "bin" / "python").chmod(0o755)
    (project / ".venv" / "mine").write_text("x")
    (project / "requirements.txt").write_text("diffusers\n")

    run()
    (project / "requirements.txt").write_text("diffusers\npeft\n")
    done = run()

    assert "rebuilding" not in done.stdout
    assert (project / ".venv" / "mine").exists()
    assert not (project / ".venv" / ".devbox-requirements").exists()


def test_a_failed_install_removes_the_venv_and_waits_for_an_edit(rig):
    project, calls, _, run = rig
    (project / "requirements.txt").write_text("broken\n")
    (project / "fail-uv").write_text("")

    def edit():
        time.sleep(2.5)
        (project / "requirements.txt").write_text("")

    threading.Thread(target=edit).start()
    done = run()

    assert "installing requirements.txt failed" in done.stdout
    assert not (project / ".venv").exists()
    assert calls.read_text().count("pip install") == 1
