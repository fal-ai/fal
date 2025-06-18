from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

Builder = Literal["depot", "service", "worker"]
BUILDERS = {"depot", "service", "worker"}
DEFAULT_COMPRESSION: str = "gzip"
DEFAULT_FORCE_COMPRESSION: bool = False


@dataclass
class ContainerImage:
    """ContainerImage represents a Docker image that can be built
    from a Dockerfile.
    """

    dockerfile_str: str
    build_args: Dict[str, str] = field(default_factory=dict)
    registries: Dict[str, Dict[str, str]] = field(default_factory=dict)
    builder: Optional[Builder] = field(default=None)
    compression: str = DEFAULT_COMPRESSION
    force_compression: bool = DEFAULT_FORCE_COMPRESSION
    secrets: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.registries:
            for registry in self.registries.values():
                keys = registry.keys()
                if "username" not in keys or "password" not in keys:
                    raise ValueError(
                        "Username and password are required for each registry"
                    )

        if self.builder and self.builder not in BUILDERS:
            raise ValueError(
                f"Invalid builder: {self.builder}, must be one of {BUILDERS}"
            )

    @classmethod
    def from_dockerfile_str(cls, text: str, **kwargs) -> "ContainerImage":
        return cls(dockerfile_str=text, **kwargs)

    @classmethod
    def from_dockerfile(cls, path: str, **kwargs) -> "ContainerImage":
        with open(path) as fobj:
            return cls.from_dockerfile_str(fobj.read(), **kwargs)

    def to_dict(self) -> dict:
        return {
            "dockerfile_str": self.dockerfile_str,
            "build_args": self.build_args,
            "registries": self.registries,
            "builder": self.builder,
            "compression": self.compression,
            "force_compression": self.force_compression,
            "secrets": self.secrets,
        }
