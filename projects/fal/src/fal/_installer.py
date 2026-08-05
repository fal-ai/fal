"""Detect how fal was installed, so upgrade hints suggest a command that works.

`pip install --upgrade fal` is wrong (or a no-op) for the common non-pip
installs: pipx and `uv tool` put fal in a managed virtualenv of their own, and
uv projects revert anything installed behind the lockfile's back.
"""

import os
import sys
from typing import Optional

_PACKAGE = "fal"

# `pipx` and `uv tool` both drop a receipt in the root of the virtualenv they
# manage, which is the only reliable way to tell them apart from a plain venv:
# pipx installs with uv under the hood these days, so the dist-info INSTALLER
# says "uv" for both.
_PIPX_MARKER = "pipx_metadata.json"
_UV_TOOL_MARKER = "uv-receipt.toml"


def _installer_from_disk(package: str) -> Optional[str]:
    """Read INSTALLER from the dist-info sitting next to this package.

    `importlib.metadata` resolves by name against `sys.path`, so a stale
    `fal.egg-info` in the working directory can shadow the real install. The
    dist-info next to the module we are actually running cannot.
    """
    import glob

    site_packages = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # `glob.escape` the directory: a bracketed path component (legal on every
    # platform) would otherwise be read as a character class and match nothing.
    pattern = os.path.join(
        glob.escape(site_packages), f"{package}-*.dist-info", "INSTALLER"
    )
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as fobj:
                installer = fobj.read().strip().lower()
        except (OSError, UnicodeDecodeError):
            continue

        if installer:
            return installer

    return None


def _installer_from_metadata(package: str) -> Optional[str]:
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        installer = distribution(package).read_text("INSTALLER")
    except (PackageNotFoundError, OSError, ValueError):
        return None

    if not installer:
        return None

    return installer.strip().lower()


def _read_installer(package: str) -> Optional[str]:
    return _installer_from_disk(package) or _installer_from_metadata(package)


def _same_path(left: Optional[str], right: Optional[str]) -> bool:
    if not left or not right:
        return False

    try:
        return os.path.samefile(left, right)
    except OSError:
        # One of them does not exist; compare them textually instead.
        return os.path.normcase(os.path.normpath(left)) == os.path.normcase(
            os.path.normpath(right)
        )


def _uv_project_root(prefix: str) -> Optional[str]:
    """The nearest ancestor of this venv holding a `uv.lock`, if any."""
    current = os.path.dirname(os.path.abspath(prefix))
    while True:
        if os.path.exists(os.path.join(current, "uv.lock")):
            return current

        parent = os.path.dirname(current)
        if parent == current:
            return None

        current = parent


def _is_uv_project(prefix: str) -> bool:
    """Is this venv the environment uv manages for a project with a lockfile?

    Sitting next to a `uv.lock` is not enough: `uv venv side-env` in a project
    root creates one too, and `uv sync` there would upgrade `.venv` instead,
    leaving the running install untouched. uv's project environment is `.venv`
    unless `UV_PROJECT_ENVIRONMENT` moves it, in which case the lockfile need
    not be adjacent at all — and, just as importantly, an adjacent `.venv` is
    then merely an ordinary venv that `uv sync` will not touch.
    """
    override = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if override:
        # The override *is* the project environment, so this is the whole
        # question — a `.venv` beside the lockfile no longer qualifies.
        if not os.path.isabs(override):
            # uv resolves a relative override against the project root, not the
            # cwd (verified against uv 0.11.29). Without a lockfile to locate
            # that root we cannot say, so decline rather than guess: the caller
            # then falls back to a `uv pip` command, which upgrades the right
            # environment even though a later `uv sync` would revert it.
            root = _uv_project_root(prefix)
            if root is None:
                return False

            override = os.path.join(root, override)

        return _same_path(override, prefix)

    if os.path.basename(prefix) != ".venv":
        return False

    return os.path.exists(os.path.join(os.path.dirname(prefix), "uv.lock"))


def _in_activated_venv() -> bool:
    """Is fal running from the virtualenv that is currently activated?

    Only then does the activated venv's `bin/` — which leads PATH — belong to
    the install we want upgraded. If a *different* venv is activated, a bare
    `pip` or `uv pip` would resolve to that one and upgrade the wrong
    environment, so callers have to spell the interpreter out instead.
    """
    return sys.prefix != sys.base_prefix and _same_path(
        os.environ.get("VIRTUAL_ENV"), sys.prefix
    )


def _quote(path: str) -> str:
    if " " not in path:
        return path

    # Valid in POSIX shells and in cmd.exe. PowerShell parses a leading quoted
    # string as an expression and needs its `&` call operator to run it, which
    # cmd.exe in turn rejects, so no single spelling satisfies all three.
    return f'"{path}"'


def _same_binary(left: str, right: str) -> bool:
    """Are these the same file, *without* following symlinks?

    A venv's `bin/python` is normally a symlink to the interpreter it was built
    from, so following symlinks would call a different venv's `python` — or the
    base interpreter itself — equivalent, and the bare name would then upgrade
    the wrong environment. pip's own hint compares `lstat`s for this reason.
    """
    try:
        return os.path.samestat(os.lstat(left), os.lstat(right))
    except OSError:
        return False


def _python_invocation() -> str:
    """The shortest spelling of the running interpreter, as pip's own hint does.

    Preferring the bare name when PATH resolves it to this very interpreter
    keeps the hint short and sidesteps quoting a path with spaces entirely.
    """
    import shutil

    executable = sys.executable or "python"
    name = os.path.basename(executable)
    found = shutil.which(name)
    if found and _same_binary(found, executable):
        return name

    return _quote(executable)


def _pip_command() -> str:
    if _in_activated_venv():
        return "pip"

    return f"{_python_invocation()} -m pip"


def _uv_pip_install_command(package: str) -> str:
    if _in_activated_venv():
        return f"uv pip install --upgrade {package}"

    # `uv pip` picks its target environment by discovery (VIRTUAL_ENV, then a
    # nearby `.venv`), not from the interpreter running fal — name it.
    python = _quote(sys.executable or "python")
    return f"uv pip install --python {python} --upgrade {package}"


def detect_install_manager(package: str = _PACKAGE) -> str:
    prefix = sys.prefix

    if os.path.exists(os.path.join(prefix, _PIPX_MARKER)):
        return "pipx"

    if os.path.exists(os.path.join(prefix, _UV_TOOL_MARKER)):
        return "uv-tool"

    # Only the leading token names the installer: Poetry writes its version
    # too ("Poetry 2.4.1"), while pip, uv and pdm write a bare name.
    installer = (_read_installer(package) or "").split()
    name = installer[0] if installer else None

    if name == "uv":
        return "uv-project" if _is_uv_project(prefix) else "uv"

    if name in ("poetry", "pdm"):
        return name

    return "pip"


def get_upgrade_command(package: str = _PACKAGE) -> str:
    try:
        manager = detect_install_manager(package)
    except Exception:
        from fal.logging import get_logger

        # Never fatal — a version nudge must not break the CLI — but log it, or
        # a wrong hint for the very installs this exists to serve leaves no
        # trace at all.
        get_logger(__name__).warning(
            "Failed to detect how %s was installed", package, exc_info=True
        )
        manager = "pip"

    if manager == "pipx":
        return f"pipx upgrade {package}"
    if manager == "uv-tool":
        return f"uv tool upgrade {package}"
    if manager == "uv-project":
        # Relocks and syncs in one command; `uv lock && uv sync` would need a
        # `&&`, which Windows PowerShell 5.1 rejects.
        return f"uv sync --upgrade-package {package}"
    if manager == "uv":
        return _uv_pip_install_command(package)
    if manager == "poetry":
        return f"poetry update {package}"
    if manager == "pdm":
        return f"pdm update {package}"

    return f"{_pip_command()} install --upgrade {package}"
