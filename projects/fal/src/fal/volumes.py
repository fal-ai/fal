import re
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Union

from fal.exceptions.volumes import VolumeInvalidMountPathError, VolumeInvalidNameError

# Mirrors the controller's rules so a bad name or mount path fails before any
# network call. The server stays authoritative (it also rejects paths that
# shadow platform mounts), and reports those through the same typed errors.
VOLUME_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}")
MOUNT_PATH_PATTERN = re.compile(r"/(?!\.{1,2}$)[A-Za-z0-9._-]+")
MAX_VOLUMES = 5


def canonical_volume_name(name: str) -> str:
    """Validate a volume name and return its canonical (lowercase) form."""
    if not isinstance(name, str) or not VOLUME_NAME_PATTERN.fullmatch(name):
        raise VolumeInvalidNameError(
            f"Invalid volume name {name!r}: volume names start with a letter or "
            "digit, contain only letters, digits, '.', '_' and '-', and are at "
            "most 32 characters long",
            volume_name=name if isinstance(name, str) else None,
        )
    return name.lower()


def validate_mount_path(path: str) -> str:
    if not isinstance(path, str) or not MOUNT_PATH_PATTERN.fullmatch(path):
        raise VolumeInvalidMountPathError(
            f"Invalid volume mount path {path!r}: a mount path is a single "
            "top-level directory such as '/models', containing only letters, "
            "digits, '.', '_' and '-'",
            mount_path=path if isinstance(path, str) else None,
        )
    return path


@dataclass(frozen=True)
class Volume:
    """A reference to an existing fal volume by name.

    This does not create storage. Create the volume first with
    ``fal.create_volume`` or ``fal volumes create <name>``. The name is
    stored in its canonical (lowercase) form.
    """

    name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", canonical_volume_name(self.name))


def normalize_volumes(volumes: Mapping[str, Union[Volume, str]]) -> Dict[str, str]:
    """Validate a ``{mount_path: volume}`` mapping and return it as
    ``{mount_path: canonical_name}``, the form sent to the server."""
    if not isinstance(volumes, Mapping):
        raise ValueError("volumes must be a mapping of mount path to Volume.")
    if len(volumes) > MAX_VOLUMES:
        raise VolumeInvalidMountPathError(
            f"At most {MAX_VOLUMES} volumes can be mounted, got {len(volumes)}"
        )
    normalized: Dict[str, str] = {}
    for mount_path, value in volumes.items():
        validate_mount_path(mount_path)
        volume = Volume(value) if isinstance(value, str) else value
        if not isinstance(volume, Volume):
            raise ValueError(
                f"volumes[{mount_path!r}] must be a fal.Volume or a volume name."
            )
        normalized[mount_path] = volume.name
    return normalized


def create_volume(
    name: str,
    *,
    host: Optional[str] = None,
    team: Optional[str] = None,
    api_key: Optional[str] = None,
    profile: Optional[str] = None,
) -> str:
    """Create a volume and return its canonical (lowercase) name.

    Raises a typed ``fal.exceptions.VolumeException`` subclass when the server
    rejects the request with a known reason.
    """
    from fal.api.client import SyncServerlessClient  # noqa: PLC0415

    client = SyncServerlessClient(
        host=host, api_key=api_key, profile=profile, team=team
    )
    return client.volumes.create(name)
