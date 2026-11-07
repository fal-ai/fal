import re
import runpy

import pytest

from fal.dev import config
from fal.dev.box import BOX_APP, REMOTE


def test_box_app_loads_like_a_deploy(monkeypatch):
    """The deploy runs box_app.py as a file; its settings come from the environment."""
    monkeypatch.setenv(config.ENV_NAME, "demo")
    monkeypatch.setenv(config.ENV_MACHINE_TYPE, "GPU-B200")
    monkeypatch.setenv(config.ENV_PYTHON_VERSION, "3.12")
    monkeypatch.setenv(config.ENV_FAL_SPEC, "fal==1.84.0")
    module = runpy.run_path(str(BOX_APP))

    app = module["DevBox"]
    assert module["APP_NAME"] == "devbox-demo"
    assert app.machine_type == "GPU-B200"
    assert app.min_concurrency == 0
    assert app.image.build_args == {"PYTHON_VERSION": "3.12", "FAL_SPEC": "fal==1.84.0"}
    assert "FROM nvidia/cuda" in app.image.dockerfile_str
    assert "def main()" in module["HOT_SERVER_SOURCE"]
    assert "devbox-save" in module["SAVE_SCRIPT"]
    assert sorted(module["PROBE_FILES"]) == ["config.py", "probe.py", "signals.py"]


def test_box_app_hot_dir_stays_under_work():
    module = runpy.run_path(str(BOX_APP))

    assert str(module["_hot_dir"]("/work/demo")) == "/work/demo"
    with pytest.raises(ValueError):
        module["_hot_dir"]("/work/../etc")
    with pytest.raises(ValueError):
        module["_hot_dir"]("/tmp")


@pytest.mark.parametrize("script", ["devbox-save.sh", "devbox-hot.sh", "bashrc.sh"])
def test_box_scripts_are_valid_bash_once_filled_in(script, tmp_path):
    import subprocess

    module = runpy.run_path(str(BOX_APP))
    text = module["_fill"](
        (REMOTE / "scripts" / script).read_text(),
        WORK="/work",
        WORK_PARENT="/",
        WORK_NAME="work",
        SNAPSHOT="/data/w.tar.zst",
        LEASE="/data/w.lease",
        STAMP="/tmp/stamp",
        INTERVAL=15,
        SERVER="/usr/local/lib/devbox-hot/hot_server.py",
        PORT=7860,
        RESTART_CODE=3,
        WORKSPACE="/data/devboxes/demo",
        CLAUDE_DIR="/data/c",
        WARNING_FILE="/tmp/w",
    )
    path = tmp_path / script
    path.write_text(text)

    done = subprocess.run(
        ["bash", "-n", str(path)], capture_output=True, text=True, check=False
    )

    assert done.returncode == 0, done.stderr
    # Bash's own variables survive; the placeholders do not.
    assert ("$stamp" in text) == (script == "devbox-save.sh")
    assert "$WORK" not in text
    assert "$PORT" not in text


def test_sshd_config_carries_the_hosts_environment(monkeypatch, tmp_path):
    module = runpy.run_path(str(BOX_APP))
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda")
    monkeypatch.setenv(config.ADMIN_KEY_SECRET, "secret")

    text = module["_sshd_config"](tmp_path / "host_key")

    assert f"HostKey {tmp_path / 'host_key'}" in text
    assert '"CUDA_HOME=/usr/local/cuda"' in text
    assert "secret" not in text
    assert "$" not in text


def test_box_app_write_files_skips_unchanged(tmp_path):
    module = runpy.run_path(str(BOX_APP))
    module["_hot_dir"].__globals__["LOCAL_WORK"] = tmp_path
    project = tmp_path / "demo"
    project.mkdir()
    (project / "load.py").write_text("same")
    before = (project / "load.py").stat().st_mtime_ns

    files = {"load.py": "same", "infer.py": "new", "pkg/util.py": "x"}
    result = module["_hot_write_files"]({"dir": str(project), "files": files})

    assert sorted(result["written"]) == ["infer.py", "pkg/util.py"]
    assert (project / "load.py").stat().st_mtime_ns == before
    with pytest.raises(ValueError, match="outside"):
        module["_hot_write_files"]({"dir": str(project), "files": {"../x": "y"}})


def test_hold_script_never_shortens_a_longer_hold(tmp_path):
    import subprocess

    script = (REMOTE / "scripts" / "hold.sh").read_text()
    hold = tmp_path / "a hold file"

    def run(until):
        return subprocess.run(
            [
                "sh",
                "-c",
                script,
                "sh",
                str(hold),
                str(until),
                str(config.HOLD_KEPT_EXIT),
            ],
            capture_output=True,
            check=False,
        ).returncode

    assert run(100) == 0
    assert hold.read_text().strip() == "100"
    assert run(50) == config.HOLD_KEPT_EXIT
    assert hold.read_text().strip() == "100"
    assert run(200) == 0
    assert hold.read_text().strip() == "200"


def test_box_app_mounts_its_volume_at_data(monkeypatch):
    monkeypatch.setenv(config.ENV_VOLUME, "devbox-demo")
    app = runpy.run_path(str(BOX_APP))["DevBox"]
    assert app.host_kwargs["volumes"] == {"/data": "devbox-demo"}

    monkeypatch.setenv(config.ENV_VOLUME, "")
    app = runpy.run_path(str(BOX_APP))["DevBox"]
    assert "volumes" not in app.host_kwargs


def test_a_failed_gpu_query_never_lets_a_busy_box_look_idle():
    from fal.dev.remote import signals

    module = runpy.run_path(str(BOX_APP))
    gpu_activity = module["_gpu_activity"]
    busy = {"util_pct": 90}
    idle = {"util_pct": 0}
    patches = {
        "busy": ([busy], None),
        "idle": ([idle], None),
        "failed": ([], "nvidia-smi timed out"),
    }
    original = signals.gpus
    try:
        results = {}
        for name, answer in patches.items():
            signals.gpus = lambda answer=answer: answer
            results[name] = gpu_activity()
    finally:
        signals.gpus = original

    assert results == {
        "busy": "GPU busy",
        "idle": None,
        "failed": "GPU status unknown",
    }


def test_a_gpu_that_cannot_be_read_is_warned_about_once_then_cleared(tmp_path):
    from fal.dev.remote import signals

    module = runpy.run_path(str(BOX_APP))
    warning = tmp_path / "warning"
    module["_warn"].__globals__["WARNING_FILE"] = warning
    original = signals.gpus
    try:
        signals.gpus = lambda: ([], "timed out")
        for _ in range(5):
            assert module["_gpu_activity"]() == "GPU status unknown"
        assert warning.read_text().count("GPU status is unknown") == 1
        signals.gpus = lambda: ([{"util_pct": 0}], None)
        assert module["_gpu_activity"]() is None
    finally:
        signals.gpus = original

    assert warning.read_text() == ""


def test_persisted_dotfile_links_do_not_dangle(monkeypatch, tmp_path):
    from pathlib import Path

    module = runpy.run_path(str(BOX_APP))
    home, workspace = tmp_path / "home", tmp_path / "workspace"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    prepare = module["_prepare_home"]
    # The function's own globals, not the copy runpy returns.
    monkeypatch.setitem(prepare.__globals__, "WORKSPACE", workspace)
    monkeypatch.setitem(
        prepare.__globals__, "CLAUDE_CONFIG_DIR", workspace / ".home" / ".claude"
    )
    monkeypatch.setitem(
        prepare.__globals__, "BASHRC_BLOCK", "# >>> devbox >>>\n# <<< devbox <<<\n"
    )

    prepare()

    link = home / ".gitconfig"
    assert link.is_symlink() and link.resolve().exists()


def _bashrc(**values):
    module = runpy.run_path(str(BOX_APP))
    return module["_fill"](
        (REMOTE / "scripts" / "bashrc.sh").read_text(),
        WORKSPACE="/data/devboxes/demo",
        WORK="/work",
        CLAUDE_DIR="/data/c",
        WARNING_FILE="/tmp/w",
        BOX_NAME="demo",
        **values,
    )


def test_bashrc_default_prompt_shows_box_gpu_path_and_branch(tmp_path):
    import subprocess

    text = _bashrc()
    assert "PS1=" in text
    # The prompt is set before .devboxrc, so a user's PS1 there wins.
    assert text.index("PS1=") < text.index('.devboxrc"')
    # Starship is only used when setup() installed it with --starship.
    assert "command -v starship" in text

    nvidia = tmp_path / "nvidia-smi"
    nvidia.write_text(
        "#!/bin/sh\necho 'NVIDIA RTX PRO 6000 Blackwell Server Edition'\n"
    )
    nvidia.chmod(0o755)
    repo = tmp_path / "repo"
    repo.mkdir()
    rc = tmp_path / "rc"
    rc.write_text(
        text.replace("/tmp/devbox-gpu-short", str(tmp_path / "gpu"))
        .replace('cd "/work"', "")
        .replace('cd "$WORK"', "")
    )
    script = (
        f'export PATH="{tmp_path}:$PATH" HOME={tmp_path}; PS1="x"; . {rc}; '
        f"cd {repo}; git init -q -b main . ; "
        'printf "%s\\n" "$(PROMPT_COMMAND= ; echo "${PS1@P}")"'
    )
    done = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=False
    )
    plain = re.sub(r"\x1b\[[0-9;]*m", "", done.stdout)
    assert "demo RTXPRO6000" in plain, done.stderr
    assert "(main)" in plain


def test_bashrc_prompt_skips_gpu_without_nvidia_smi(tmp_path):
    import subprocess

    rc = tmp_path / "rc"
    rc.write_text(_bashrc().replace("/tmp/devbox-gpu-short", str(tmp_path / "gpu")))
    done = subprocess.run(
        ["bash", "-c", f'PATH=/usr/bin:/bin; PS1=x; . {rc}; echo "${{PS1@P}}"'],
        capture_output=True,
        text=True,
        check=False,
    )
    plain = re.sub(r"\x1b\[[0-9;]*m", "", done.stdout)
    assert plain.startswith("demo ")
    assert "RTX" not in plain and "H100" not in plain


def test_bashrc_warns_about_environments_on_the_data_volume(tmp_path):
    import subprocess

    fake_uv = tmp_path / "uv"
    fake_uv.write_text('#!/bin/sh\necho "ran uv $*"\nexit 3\n')
    fake_uv.chmod(0o755)
    rc = tmp_path / "rc"
    rc.write_text(_bashrc().replace("/tmp/devbox-gpu-short", str(tmp_path / "gpu")))
    script = (
        f'PATH="{tmp_path}:/usr/bin:/bin"; PS1=x; . {rc}; '
        "PWD=/data/voice uv venv; echo $?; PWD=/data/voice uv run x; "
        "PWD=/work/voice uv venv"
    )
    done = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=False
    )
    # The real uv runs unchanged and its exit status comes back.
    assert done.stdout.splitlines() == [
        "ran uv venv",
        "3",
        "ran uv run x",
        "ran uv venv",
    ]
    # Only the environment-building command under /data gets the note.
    assert done.stderr.count("network volume") == 1
    assert "Keep them in /work (local disk" in done.stderr


def test_starship_off_by_default_and_failed_install_keeps_default(
    monkeypatch, tmp_path
):
    module = runpy.run_path(str(BOX_APP))
    assert module["STARSHIP_TOML"] is None

    warnings = []
    module["_setup_starship"].__globals__["_warn"] = warnings.append

    def boom(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(module["subprocess"], "run", boom)
    module["_setup_starship"](tmp_path, "format = '$all'")

    assert not (tmp_path / ".config" / "starship.toml").exists()
    assert "default prompt" in warnings[0]


def test_starship_installs_then_writes_only_the_config(monkeypatch, tmp_path):
    module = runpy.run_path(str(BOX_APP))
    calls = []
    monkeypatch.setenv(config.ADMIN_KEY_SECRET, "secret")
    monkeypatch.setattr(
        module["subprocess"], "run", lambda cmd, **kw: calls.append((cmd, kw))
    )

    module["_setup_starship"](tmp_path, "format = '$all'")

    assert "starship.rs/install.sh" in calls[0][0][-1]
    assert config.ADMIN_KEY_SECRET not in calls[0][1]["env"]
    assert (tmp_path / ".config" / "starship.toml").read_text() == "format = '$all'"


def test_local_starship_config_reads_env_or_default(monkeypatch, tmp_path):
    from fal.dev.box import local_starship_config
    from fal.dev.errors import DevError

    cfg = tmp_path / "s.toml"
    cfg.write_text("add_newline = false\n")
    monkeypatch.setenv("STARSHIP_CONFIG", str(cfg))
    assert local_starship_config() == "add_newline = false\n"

    monkeypatch.setenv("STARSHIP_CONFIG", str(tmp_path / "missing.toml"))
    with pytest.raises(DevError):
        local_starship_config()


def test_bashrc_uses_starship_only_when_installed_with_its_config(tmp_path):
    import subprocess

    rc = tmp_path / "rc"
    rc.write_text(_bashrc().replace("/tmp/devbox-gpu-short", str(tmp_path / "gpu")))
    fake = tmp_path / "bin"
    fake.mkdir()
    starship = fake / "starship"
    starship.write_text("#!/bin/sh\necho 'PS1=STARSHIP '\n")
    starship.chmod(0o755)

    def prompt(home, path):
        done = subprocess.run(
            ["bash", "-c", f'HOME={home} PATH={path}; PS1=x; . {rc}; echo "$PS1"'],
            capture_output=True,
            text=True,
            check=False,
        )
        return done.stdout

    path = f"{fake}:/usr/bin:/bin"
    home = tmp_path / "home"
    (home / ".config").mkdir(parents=True)
    assert "demo" in prompt(home, path)
    (home / ".config" / "starship.toml").write_text("")
    assert prompt(home, path).strip() == "STARSHIP"
    # Config copied but the install failed: the default prompt stays.
    assert "demo" in prompt(home, "/usr/bin:/bin")
