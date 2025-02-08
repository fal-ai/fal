from dataclasses import dataclass, field
from typing import Dict, Literal

Builder = Literal["depot", "service", "worker"]
BUILDERS = {"depot", "service", "worker"}
DEFAULT_BUILDER: Builder = "depot"


@dataclass
class ContainerImage:
    """ContainerImage represents a Docker image that can be built
    from a Dockerfile.
    """

    dockerfile_str: str
    build_args: Dict[str, str] = field(default_factory=dict)
    registries: Dict[str, Dict[str, str]] = field(default_factory=dict)
    builder: Builder = field(default=DEFAULT_BUILDER)

    def __post_init__(self) -> None:
        if self.registries:
            for registry in self.registries.values():
                keys = registry.keys()
                if "username" not in keys or "password" not in keys:
                    raise ValueError(
                        "Username and password are required for each registry"
                    )

        if self.builder not in BUILDERS:
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
        }
