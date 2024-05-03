from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import portalocker

_DEFAULT_HOME_DIR = str(Path.home() / ".fal")
_FAL_HOME_DIR = os.getenv("FAL_HOME_DIR", _DEFAULT_HOME_DIR)
_TOKEN_FILE = "auth0_token"
_LOCK_FILE = ".portalock"


def _check_dir_exist():
    """
    Checks if a specific directory exists, creates if not.
    In case the user didn't set a custom dir, will turn to the default home
    """
    dir = Path(_FAL_HOME_DIR).expanduser()

    if not dir.exists():
        dir.mkdir(parents=True)

    return dir


def _read_token_file(path: Path) -> list[str] | None:
    if path.exists():
        return path.read_text().splitlines()


def _write_file(path: Path, contents: list[str]):
    path.write_text("\n".join(contents))


def load_token() -> tuple[str | None, str | None]:
    lines = _read_token_file(_check_dir_exist() / _TOKEN_FILE)
    if not lines:
        return None, None

    refresh_token = lines[0]
    access_token = None
    if len(lines) > 1:
        access_token = lines[1]

    return refresh_token, access_token


def save_token(refresh_token: str, access_token: str | None = None) -> None:
    tokens = [refresh_token]
    if access_token:
        tokens.append(access_token)
    return _write_file(_check_dir_exist() / _TOKEN_FILE, tokens)


def delete_token() -> None:
    path = _check_dir_exist() / _TOKEN_FILE
    path.unlink()


@contextmanager
def lock_token():
    """
    Lock the access to the token file to avoid race conditions when running multiple
    scripts at the same time.
    """
    lock_file = _check_dir_exist() / _LOCK_FILE
    with portalocker.utils.TemporaryFileLock(
        str(lock_file),
        fail_when_locked=False,
        timeout=20,
    ):
        yield
