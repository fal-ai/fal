"""Which project files may leave this machine.

Used by the commands that upload a project folder to a box. Secrets stay
local because uploaded files also land in the team's object store. The secret
check is a denylist, so it catches the usual names only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Collection, Iterator

SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
    }
)
_ENV_PREFIX = ".env"
_SECRET_PREFIXES = ("id_rsa", "id_ed25519", "id_ecdsa", "id_dsa")
_SECRET_DIRS = {".ssh", ".aws", ".gnupg", ".docker", ".kube", ".azure", "secrets"}
_SECRET_NAMES = {
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".pgpass",
    ".htpasswd",
    ".git-credentials",
    "credentials.json",
    "service-account.json",
    "token.json",
    "kubeconfig",
    ".dockercfg",
    ".boto",
    "application_default_credentials.json",
}
_SECRET_SUFFIXES = (
    ".pem",
    ".key",
    ".p7b",
    ".p8",
    ".p12",
    ".pfx",
    ".jks",
    ".keystore",
    ".gpg",
    ".kdbx",
    ".env",
    ".tfvars",
    ".tfstate",
)
_SERVICE_ACCOUNT = ("service_account", "service-account", "serviceaccount")
_SECRET_JSON_SUFFIXES = (".tfvars.json", "-key.json", "_key.json")
_SECRET_JSON_PREFIXES = ("client_secret", "client-secret")
_SECRET_STEMS = ("secrets.", "secret.")


def secret_like(parts: tuple[str, ...]) -> bool:
    """Whether a path (split into its parts) is an env, key or credential file."""
    lowered = tuple(part.lower() for part in parts)
    name = lowered[-1] if lowered else ""
    return (
        any(part.startswith(_ENV_PREFIX) for part in lowered)
        or bool(_SECRET_DIRS.intersection(lowered))
        or name.startswith(_SECRET_PREFIXES)
        # secrets.yaml, secrets.json...; a secrets.py module is code.
        or (name.startswith(_SECRET_STEMS) and not name.endswith(".py"))
        or name in _SECRET_NAMES
        or (name.startswith(_SERVICE_ACCOUNT) and name.endswith(".json"))
        or name == "credentials"
        or name.startswith("credentials.")
        or (name.startswith(_SECRET_JSON_PREFIXES) and name.endswith(".json"))
        or ".tfstate." in name  # backups
        or name.endswith(_SECRET_JSON_SUFFIXES)
        or ("adminsdk" in name and name.endswith(".json"))
        or name.endswith(_SECRET_SUFFIXES)
    )


def walk(
    root: Path, *, skip_dirs: Collection[str] = SKIP_DIRS, hidden: bool = True
) -> Iterator[Path]:
    """The files under root that may be uploaded, as paths relative to it.

    With hidden=False, dot files and dot folders are left out too. Symlinks are
    never followed, so a link cannot pull in a file from outside the folder.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in skip_dirs
            and (hidden or not d.startswith("."))
            and not os.path.islink(os.path.join(dirpath, d))
        ]
        for filename in filenames:
            path = (Path(dirpath) / filename).relative_to(root)
            if (
                (hidden or not filename.startswith("."))
                and not secret_like(path.parts)
                and not (root / path).is_symlink()
            ):
                yield path
