import builtins
import importlib
import os
import shutil

fal_logging = importlib.import_module("fal.logging")
installer = importlib.import_module("fal._installer")


def _fake_prefix(monkeypatch, tmp_path, marker=None, name="env"):
    prefix = tmp_path / name
    prefix.mkdir()
    if marker:
        (prefix / marker).write_text("{}")
    monkeypatch.setattr(installer.sys, "prefix", str(prefix))
    monkeypatch.setattr(installer.sys, "base_prefix", str(tmp_path / "base"))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)
    return prefix


def _fake_installer_metadata(monkeypatch, value):
    monkeypatch.setattr(installer, "_read_installer", lambda _package: value)


def _fake_interpreter(monkeypatch, executable, on_path=None):
    """Run as `executable`, with `on_path` as what its bare name resolves to."""
    monkeypatch.setattr(installer.sys, "executable", executable)
    monkeypatch.setattr(shutil, "which", lambda _name: on_path)


def test_pipx_install_suggests_pipx_upgrade(monkeypatch, tmp_path) -> None:
    # pipx shells out to uv these days, so INSTALLER says "uv" even here.
    _fake_prefix(monkeypatch, tmp_path, marker="pipx_metadata.json")
    _fake_installer_metadata(monkeypatch, "uv")

    assert installer.get_upgrade_command() == "pipx upgrade fal"


def test_uv_tool_install_suggests_uv_tool_upgrade(monkeypatch, tmp_path) -> None:
    _fake_prefix(monkeypatch, tmp_path, marker="uv-receipt.toml")
    _fake_installer_metadata(monkeypatch, "uv")

    assert installer.get_upgrade_command() == "uv tool upgrade fal"


def test_uv_venv_suggests_uv_pip(monkeypatch, tmp_path) -> None:
    prefix = _fake_prefix(monkeypatch, tmp_path)
    monkeypatch.setenv("VIRTUAL_ENV", str(prefix))
    _fake_installer_metadata(monkeypatch, "uv")

    assert installer.get_upgrade_command() == "uv pip install --upgrade fal"


def test_uv_pip_names_the_interpreter_when_it_is_not_the_activated_venv(
    monkeypatch, tmp_path
) -> None:
    # `uv pip` finds its target by discovery (VIRTUAL_ENV, then a nearby
    # `.venv`), so without --python it would upgrade the activated venv A while
    # fal keeps running from B.
    _fake_prefix(monkeypatch, tmp_path)
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "some-other-venv"))
    _fake_interpreter(monkeypatch, "/usr/bin/python3")
    _fake_installer_metadata(monkeypatch, "uv")

    assert installer.get_upgrade_command() == (
        "uv pip install --python /usr/bin/python3 --upgrade fal"
    )


def test_uv_project_suggests_sync_with_upgrade_package(monkeypatch, tmp_path) -> None:
    prefix = _fake_prefix(monkeypatch, tmp_path, name=".venv")
    (prefix.parent / "uv.lock").write_text("")
    _fake_installer_metadata(monkeypatch, "uv")

    assert installer.get_upgrade_command() == "uv sync --upgrade-package fal"


def test_side_venv_in_a_uv_project_is_not_the_project_environment(
    monkeypatch, tmp_path
) -> None:
    # `uv venv side-env` in a project root sits next to the lockfile but is not
    # what `uv sync` would touch, so syncing would leave this install stale.
    prefix = _fake_prefix(monkeypatch, tmp_path, name="side-env")
    (prefix.parent / "uv.lock").write_text("")
    monkeypatch.setenv("VIRTUAL_ENV", str(prefix))
    _fake_installer_metadata(monkeypatch, "uv")

    assert installer.get_upgrade_command() == "uv pip install --upgrade fal"


def test_uv_project_environment_override_is_recognised(monkeypatch, tmp_path) -> None:
    # UV_PROJECT_ENVIRONMENT moves the managed environment, so it need not be
    # named `.venv` nor sit beside the lockfile.
    prefix = _fake_prefix(monkeypatch, tmp_path, name="managed-env")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(prefix))
    _fake_installer_metadata(monkeypatch, "uv")

    assert installer.get_upgrade_command() == "uv sync --upgrade-package fal"


def test_poetry_install_suggests_poetry_update(monkeypatch, tmp_path) -> None:
    # A real `poetry add` writes its version too, e.g. "Poetry 2.4.1"; only
    # `poetry install` of the root project writes a bare "poetry".
    _fake_prefix(monkeypatch, tmp_path)
    _fake_installer_metadata(monkeypatch, "poetry 2.4.1")

    assert installer.get_upgrade_command() == "poetry update fal"


def test_bare_poetry_installer_value_still_matches(monkeypatch, tmp_path) -> None:
    _fake_prefix(monkeypatch, tmp_path)
    _fake_installer_metadata(monkeypatch, "poetry")

    assert installer.get_upgrade_command() == "poetry update fal"


def test_pdm_install_suggests_pdm_update(monkeypatch, tmp_path) -> None:
    _fake_prefix(monkeypatch, tmp_path)
    _fake_installer_metadata(monkeypatch, "pdm")

    assert installer.get_upgrade_command() == "pdm update fal"


def test_activated_venv_suggests_bare_pip(monkeypatch, tmp_path) -> None:
    prefix = _fake_prefix(monkeypatch, tmp_path)
    monkeypatch.setenv("VIRTUAL_ENV", str(prefix))
    _fake_installer_metadata(monkeypatch, "pip")

    assert installer.get_upgrade_command() == "pip install --upgrade fal"


def test_activated_venv_accepts_an_equivalent_path(monkeypatch, tmp_path) -> None:
    # A trailing slash (or a symlinked path) still names the same venv.
    prefix = _fake_prefix(monkeypatch, tmp_path)
    monkeypatch.setenv("VIRTUAL_ENV", str(prefix) + os.sep)
    _fake_installer_metadata(monkeypatch, "pip")

    assert installer.get_upgrade_command() == "pip install --upgrade fal"


def test_a_different_activated_venv_spells_out_the_interpreter(
    monkeypatch, tmp_path
) -> None:
    # fal runs from venv B while venv A is the activated one, so bare `pip`
    # would resolve through PATH to A's pip and upgrade the wrong environment.
    _fake_prefix(monkeypatch, tmp_path)
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "some-other-venv"))
    _fake_interpreter(monkeypatch, "/usr/bin/python3")
    _fake_installer_metadata(monkeypatch, "pip")

    assert (
        installer.get_upgrade_command()
        == "/usr/bin/python3 -m pip install --upgrade fal"
    )


def test_global_install_spells_out_the_interpreter(monkeypatch, tmp_path) -> None:
    _fake_prefix(monkeypatch, tmp_path)
    _fake_interpreter(monkeypatch, "/usr/bin/python3")
    _fake_installer_metadata(monkeypatch, "pip")

    assert (
        installer.get_upgrade_command()
        == "/usr/bin/python3 -m pip install --upgrade fal"
    )


def test_interpreter_on_path_is_named_without_its_directory(
    monkeypatch, tmp_path
) -> None:
    # When PATH resolves the bare name to this very interpreter, the short
    # spelling is equivalent — and avoids quoting entirely.
    executable = tmp_path / "python3.14"
    executable.write_text("")
    _fake_prefix(monkeypatch, tmp_path)
    _fake_interpreter(monkeypatch, str(executable), on_path=str(executable))
    _fake_installer_metadata(monkeypatch, "pip")

    assert installer.get_upgrade_command() == "python3.14 -m pip install --upgrade fal"


def test_another_venv_symlinked_to_the_same_interpreter_is_not_equivalent(
    monkeypatch, tmp_path
) -> None:
    # A venv's `bin/python` is a symlink to the interpreter it was built from,
    # so following symlinks makes venv A's python look like venv B's. Running
    # the bare name would then upgrade A while fal keeps running from B.
    base = tmp_path / "base-python"
    base.write_text("")

    def venv_python(prefix):
        binary = prefix / "bin" / "python3.12"
        binary.parent.mkdir(parents=True)
        binary.symlink_to(base)
        return binary

    ours = venv_python(_fake_prefix(monkeypatch, tmp_path, name="venv-b"))
    theirs = venv_python(tmp_path / "venv-a")
    _fake_interpreter(monkeypatch, str(ours), on_path=str(theirs))
    _fake_installer_metadata(monkeypatch, "pip")

    assert installer.get_upgrade_command() == f"{ours} -m pip install --upgrade fal"


def test_interpreter_path_with_spaces_is_quoted(monkeypatch, tmp_path) -> None:
    _fake_prefix(monkeypatch, tmp_path)
    # A different `python` leads PATH, so the full path is needed.
    _fake_interpreter(
        monkeypatch, "/opt/my python/bin/python", on_path="/usr/bin/python"
    )
    _fake_installer_metadata(monkeypatch, "pip")

    assert installer.get_upgrade_command() == (
        '"/opt/my python/bin/python" -m pip install --upgrade fal'
    )


def _write_dist_info(tmp_path, contents, *, directory="projects [client]"):
    """Lay out a real `fal-*.dist-info/INSTALLER` and point the module at it."""
    site_packages = tmp_path / directory / "site-packages"
    dist_info = site_packages / "fal-1.2.3.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "INSTALLER").write_text(contents)
    return site_packages / "fal" / "_installer.py"


def test_installer_is_read_from_the_dist_info_beside_the_package(
    monkeypatch, tmp_path
) -> None:
    # Bracketed path components are legal everywhere but are a glob character
    # class, so an unescaped pattern silently matches nothing here. The real
    # capitalisation is exercised too: only this layer lowercases.
    module = _write_dist_info(tmp_path, "Poetry 2.4.1\n")
    monkeypatch.setattr(installer, "__file__", str(module))

    assert installer._installer_from_disk("fal") == "poetry 2.4.1"

    _fake_prefix(monkeypatch, tmp_path)
    assert installer.get_upgrade_command() == "poetry update fal"


def test_an_undecodable_installer_file_is_ignored(monkeypatch, tmp_path) -> None:
    # A corrupted INSTALLER used to escape as UnicodeDecodeError, which only
    # the catch-all in `get_upgrade_command` stopped — silently, and after
    # losing the detection. Raised from `open` rather than written as bytes so
    # the test does not depend on the locale's encoding.
    module = _write_dist_info(tmp_path, "uv\n", directory="plain")
    monkeypatch.setattr(installer, "__file__", str(module))

    def explode(*_args, **_kwargs):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(builtins, "open", explode)

    assert installer._installer_from_disk("fal") is None


def test_missing_metadata_falls_back_to_pip(monkeypatch, tmp_path) -> None:
    prefix = _fake_prefix(monkeypatch, tmp_path)
    monkeypatch.setenv("VIRTUAL_ENV", str(prefix))
    _fake_installer_metadata(monkeypatch, None)

    assert installer.get_upgrade_command() == "pip install --upgrade fal"


def test_detection_errors_fall_back_to_pip_and_are_logged(
    monkeypatch, tmp_path
) -> None:
    prefix = _fake_prefix(monkeypatch, tmp_path)
    monkeypatch.setenv("VIRTUAL_ENV", str(prefix))

    def boom(_package):
        raise RuntimeError("no metadata for you")

    monkeypatch.setattr(installer, "_read_installer", boom)

    warnings = []

    class _Logger:
        def warning(self, message, *args, **kwargs):
            warnings.append((message % args, kwargs))

    monkeypatch.setattr(fal_logging, "get_logger", lambda *_args: _Logger())

    assert installer.get_upgrade_command() == "pip install --upgrade fal"

    assert warnings == [("Failed to detect how fal was installed", {"exc_info": True})]
