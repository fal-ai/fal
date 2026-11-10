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


def _env_block(**values):
    module = runpy.run_path(str(BOX_APP))
    return module["_fill"](
        (REMOTE / "scripts" / "bashrc_env.sh").read_text(),
        WORKSPACE="/data/devboxes/demo",
        WORK="/work",
        CLAUDE_DIR="/data/c",
        **values,
    )


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
    # The login notice comes first; the prompt is the last line.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", done.stdout).splitlines()[-1]
    assert plain.startswith("demo ")
    assert "RTX" not in plain and "H100" not in plain


def _uv_notes(tmp_path, cases):
    """Run each `(cwd, command)` with fake uv and python; the stderr notes it got."""
    import subprocess

    for tool in ("uv", "python3", "python"):
        fake = tmp_path / tool
        fake.write_text(f'#!/bin/sh\necho "ran {tool} $*"\nexit 3\n')
        fake.chmod(0o755)
    rc = tmp_path / "rc"
    rc.write_text(_env_block())
    results = []
    for cwd, command in cases:
        done = subprocess.run(
            [
                "bash",
                "-c",
                f'PATH="{tmp_path}:/usr/bin:/bin"; unset VIRTUAL_ENV; . {rc}; PWD={cwd}; {command}',
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        results.append(
            (done.returncode, done.stdout.strip(), "network volume" in done.stderr)
        )
    return results


def test_bashrc_warns_about_environments_on_the_data_volume(tmp_path):
    cases = [
        ("/data/voice", "uv venv"),
        ("/data/voice", "uv run x"),
        ("/data/voice", "uv sync"),
        ("/data/voice", "uv pip install x"),
        ("/work/voice", "uv venv"),
        ("/work/voice", "uv venv /data/x"),
        ("/data/voice", "uv venv /work/x"),
        ("/data/voice", "uv venv --python 3.12"),
        ("/work", "UV_PROJECT_ENVIRONMENT=/data/e uv sync"),
        ("/work", "uv venv /data/x --python 3.12"),
        ("/work", "uv venv -p 3.12 /data/x"),
        ("/work", "python3 -m venv /data/x"),
        ("/data/voice", "python -m venv v"),
        ("/work", "python3 -m venv /work/x"),
    ]
    noted = [r[2] for r in _uv_notes(tmp_path, cases)]
    assert noted == [
        True,
        True,
        True,
        True,
        False,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]


def test_bashrc_data_warning_skips_read_only_and_help_commands(tmp_path):
    cases = [
        ("/data/v", "uv pip list"),
        ("/data/v", "uv pip show x"),
        ("/data/v", "uv pip freeze"),
        ("/data/v", "uv sync --help"),
        ("/data/v", "uv venv -h"),
        ("/data/v", "python3 -m venv --help"),
        ("/data/v", "python3 script.py"),
    ]
    results = _uv_notes(tmp_path, cases)
    assert [r[2] for r in results] == [False] * 7
    # The real command still runs and its exit status comes back.
    assert results[0][:2] == (3, "ran uv pip list")


def test_box_environment_reaches_non_interactive_shells(tmp_path):
    import subprocess

    text = _env_block()
    # Variables and functions only: output would break scp and rsync.
    assert "PS1" not in text and "cat " not in text
    rc = tmp_path / "rc"
    rc.write_text(text)
    done = subprocess.run(
        [
            "bash",
            "-c",
            f". {rc}; echo $HISTFILE $DEVBOX_WORKSPACE $DEVBOX_WORK $CLAUDE_CONFIG_DIR; "
            "type -t uv",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.stdout.splitlines() == [
        "/data/devboxes/demo/.bash_history /data/devboxes/demo /work /data/c",
        "function",
    ]
    assert done.stderr == ""
    # The interactive block no longer carries them; it sits after the env block.
    assert "HISTFILE" not in _bashrc()


def test_login_notice_points_at_devboxrc_and_warns_about_cold_data():
    text = _bashrc()
    assert "~/.devboxrc" in text
    assert "cold after a restart" in text


def test_prepare_home_links_devboxrc_and_puts_env_first(monkeypatch, tmp_path):
    module = runpy.run_path(str(BOX_APP))
    glb = module["_prepare_home"].__globals__
    home = tmp_path / "home"
    home.mkdir()
    (home / ".bashrc").write_text('# stock\n[ -z "$PS1" ] && return\n')
    monkeypatch.setattr(module["Path"], "home", staticmethod(lambda: home))
    for name, value in (
        ("WORKSPACE", tmp_path / "ws"),
        ("CLAUDE_CONFIG_DIR", tmp_path / "ws" / "c"),
        ("STARSHIP_TOML", None),
    ):
        monkeypatch.setitem(glb, name, value)

    module["_prepare_home"]()
    module["_prepare_home"]()

    text = (home / ".bashrc").read_text()
    assert text.count("# >>> devbox-env >>>") == 1
    assert text.count("# >>> devbox >>>") == 1
    # The early return of a stock .bashrc must not hide the env from ssh commands.
    assert text.index("# >>> devbox-env >>>") < text.index("return")
    assert (home / ".devboxrc").readlink() == tmp_path / "ws" / ".devboxrc"
    assert (home / ".devboxrc").exists()


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
    assert prompt(home, path).strip().splitlines()[-1] == "STARSHIP"
    # Config copied but the install failed: the default prompt stays.
    assert "demo" in prompt(home, "/usr/bin:/bin")


def _user_env_module(monkeypatch, tmp_path, secrets="HF_TOKEN,DEVBOX_ADMIN_KEY"):
    """box_app loaded with secrets and variables, writing its files under tmp_path."""
    import os

    monkeypatch.setenv(config.ENV_SECRETS, secrets)
    monkeypatch.setenv(config.ENV_USER_ENV, '{"FOO": "it\'s \\"bar\\"\\nbaz"}')
    monkeypatch.setattr(config, "USER_ENV_FILE", str(tmp_path / "env"))
    monkeypatch.setattr(config, "USER_ENV_NAMES_FILE", str(tmp_path / "names.json"))
    # What the runtime gives the app: every listed secret, and the admin key.
    monkeypatch.setattr(
        os,
        "environ",
        {**os.environ, "HF_TOKEN": "hf-value", config.ADMIN_KEY_SECRET: "admin-value"},
    )
    return runpy.run_path(str(BOX_APP))


def test_box_app_lists_the_secrets_you_chose_after_the_admin_key(monkeypatch, tmp_path):
    module = _user_env_module(monkeypatch, tmp_path)

    # The admin key is listed once, whatever --secret said.
    assert module["DevBox"].secrets == [config.ADMIN_KEY_SECRET, "HF_TOKEN"]
    assert module["USER_ENV"] == {"FOO": 'it\'s "bar"\nbaz'}


def test_box_app_without_user_env_keeps_only_the_admin_key(monkeypatch):
    monkeypatch.delenv(config.ENV_SECRETS, raising=False)
    monkeypatch.delenv(config.ENV_USER_ENV, raising=False)

    assert runpy.run_path(str(BOX_APP))["DevBox"].secrets == [config.ADMIN_KEY_SECRET]


def test_user_processes_get_your_secrets_but_never_the_admin_key(monkeypatch, tmp_path):
    module = _user_env_module(monkeypatch, tmp_path)

    module["_prepare_user_env"]()
    env = module["_user_env"]()

    assert env["HF_TOKEN"] == "hf-value"
    assert env["FOO"] == 'it\'s "bar"\nbaz'
    assert config.ADMIN_KEY_SECRET not in env
    # The file that shells, exec and ssh read has no admin key either, even
    # though the runtime gave it to this process.
    text = (tmp_path / "env").read_text()
    assert "admin-value" not in text and config.ADMIN_KEY_SECRET not in text
    assert "hf-value" in text
    assert (tmp_path / "env").stat().st_mode & 0o777 == 0o600


def test_the_names_file_has_names_and_no_values(monkeypatch, tmp_path):
    import json

    module = _user_env_module(monkeypatch, tmp_path)

    module["_prepare_user_env"]()

    text = (tmp_path / "names.json").read_text()
    assert json.loads(text) == {"secrets": ["HF_TOKEN"], "env": ["FOO"]}
    assert "hf-value" not in text and "baz" not in text
    assert (tmp_path / "names.json").stat().st_mode & 0o777 == 0o600


def test_a_secret_missing_from_the_environment_is_not_reported(monkeypatch, tmp_path):
    import json

    module = _user_env_module(monkeypatch, tmp_path, secrets="HF_TOKEN,NOPE")

    module["_prepare_user_env"]()

    assert json.loads((tmp_path / "names.json").read_text())["secrets"] == ["HF_TOKEN"]


def test_sshd_config_never_carries_the_admin_key_with_user_env(monkeypatch, tmp_path):
    module = _user_env_module(monkeypatch, tmp_path)
    module["_prepare_user_env"]()

    text = module["_sshd_config"](tmp_path / "host_key")

    assert '"HF_TOKEN=hf-value"' in text
    assert "admin-value" not in text
    assert f'"{config.ADMIN_KEY_SECRET}=' not in text


def _filled_env_block(monkeypatch, tmp_path):
    module = _user_env_module(monkeypatch, tmp_path)
    module["_prepare_user_env"]()
    text = module["_fill"](
        (REMOTE / "scripts" / "bashrc_env.sh").read_text(),
        WORKSPACE="/data/devboxes/demo",
        WORK="/work",
        CLAUDE_DIR="/data/c",
        USER_ENV_FILE=config.USER_ENV_FILE,
        ADMIN_KEY=config.ADMIN_KEY_SECRET,
    )
    block = tmp_path / "bashrc"
    block.write_text(text)
    return block


def test_new_shells_have_your_environment_and_not_the_admin_key(monkeypatch, tmp_path):
    import os
    import subprocess

    block = _filled_env_block(monkeypatch, tmp_path)
    # A runner shell may carry the admin key from the app's environment.
    env = {"PATH": os.environ["PATH"], config.ADMIN_KEY_SECRET: "admin-value"}

    done = subprocess.run(
        [
            "bash",
            "-c",
            f". {block}; "
            f'printf "%s|%s|%s" "$HF_TOKEN" "$FOO" "${config.ADMIN_KEY_SECRET}"',
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert done.stdout == 'hf-value|it\'s "bar"\nbaz|'
    assert done.stderr == ""


def test_exec_runs_with_your_environment_and_without_the_admin_key(
    monkeypatch, tmp_path
):
    import os
    import subprocess

    from fal.dev import execute

    _filled_env_block(monkeypatch, tmp_path)
    monkeypatch.setattr(execute, "LINGER_SECONDS", 0)
    command, marker = execute.wrap(
        [
            "bash",
            "-c",
            f'echo "$FOO"; test -n "$HF_TOKEN" && echo has-secret; '
            f'test -z "${config.ADMIN_KEY_SECRET}" && echo admin-hidden',
        ]
    )
    env = {"PATH": os.environ["PATH"], config.ADMIN_KEY_SECRET: "admin-value"}

    done = subprocess.run(command, env=env, capture_output=True, text=True, check=True)

    assert done.stdout.split(marker.decode())[0] == (
        'it\'s "bar"\nbaz\nhas-secret\nadmin-hidden\n'
    )
