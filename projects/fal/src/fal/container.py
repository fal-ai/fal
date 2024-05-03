from dataclasses import dataclass
from os import PathLike
from typing import Any


@dataclass
class ContainerImage:
    """ContainerImage represents a Docker image that can be built
    from a Dockerfile.
    """

    _dockerfile: str | PathLike
    build_environment: dict[str, Any] | None = None
    build_arguments: dict[str, str] | None = None

    @property
    def dockerfile(self) -> str:
        if isinstance(self._dockerfile, PathLike):
            with open(self._dockerfile) as f:
                return f.read()
        return self._dockerfile

    @dockerfile.setter
    def dockerfile(self, value: str | PathLike) -> None:
        self._dockerfile = value
