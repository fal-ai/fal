import json
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

Builder = Literal["depot", "service", "worker"]
BUILDERS = {"depot", "service", "worker"}
DEFAULT_COMPRESSION: str = "gzip"
DEFAULT_FORCE_COMPRESSION: bool = False

# Default patterns to ignore
DEFAULT_DOCKERIGNORE_PATTERNS = [
    ".git",
    ".gitignore",
    ".dockerignore",
    "Dockerfile",
    "Dockerfile.*",
    "*.md",
    "__pycache__",
    "*.pyc",
    ".DS_Store",
    ".venv",
    "venv",
    ".env",
    "node_modules",
]


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
    # Build context directory
    context_dir: os.PathLike = field(default=Path.cwd())
    dockerignore: Optional[List[str]] = field(default=None)
    dockerignore_path: Optional[os.PathLike] = field(default=None)

    def __post_init__(self) -> None:
        # Validate dockerfile first
        if not self.dockerfile_str or not self.dockerfile_str.strip():
            raise ValueError(
                "Invalid dockerfile: Dockerfile is required.\n"
                "Either use ContainerImage.from_dockerfile_str() or "
                "ContainerImage.from_dockerfile() to create a container image."
            )

        # Initialize dockerignore
        self._dockerignore = DockerignoreHandler(
            context_dir=self.context_dir,
            dockerignore=self.dockerignore,
            dockerignore_path=self.dockerignore_path,
        ).get_regex_patterns()

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
            "docker_context_dir": str(self.context_dir),
            "docker_files_list": self.get_copy_add_sources(),
            "docker_ignore": self._dockerignore,
        }

    def get_copy_add_sources(self) -> List[str]:
        """
        Get list of src paths/patterns from COPY/ADD commands. This method only
        parses the Dockerfile - it doesn't access the filesystem.

        Returns:
            List of src paths (e.g., ["src/", "requirements.txt", "*.py"]) that can be
            passed to FileSync.sync_files(). Returns empty list if no COPY/ADD commands
            found.
        """
        return DockerfileParser(content=self.dockerfile_str).parse_copy_add_sources()

    def add_dockerignore(
        self,
        patterns: Optional[List[str]] = None,
        path: Optional[os.PathLike] = None,
    ) -> None:
        """Add or update dockerignore patterns.

        Sets the internal dockerignore patterns using gitignore-style matching.
        You can provide either a list of patterns or a path to a .dockerignore file.

        Args:
            patterns: List of gitignore-style patterns
            path: Path to a .dockerignore file

        Raises:
            ValueError: If both patterns and path are provided, or neither
        """
        if patterns is not None and path is not None:
            raise ValueError(
                "Cannot specify both 'patterns' and 'path'. Please provide only one."
            )

        if patterns is None and path is None:
            raise ValueError(
                "Must specify either 'patterns' or 'path'. Please provide one of them."
            )

        if path is not None:
            # Load patterns from file
            handler = DockerignoreHandler(dockerignore_path=path)
        else:
            # patterns is guaranteed to be not None here due to validation above
            handler = DockerignoreHandler(dockerignore=patterns)

        # Convert gitignore patterns to regex
        self._dockerignore = handler.get_regex_patterns()


@dataclass
class DockerignoreHandler:
    context_dir: Optional[os.PathLike] = None
    dockerignore: Optional[List[str]] = None
    dockerignore_path: Optional[os.PathLike] = None

    def _read_dockerignore_file(self, path: Path) -> List[str]:
        with open(path) as f:
            lines = f.read().splitlines()

        # Filter out empty lines and comments
        return [
            stripped
            for line in lines
            if (stripped := line.strip()) and not stripped.startswith("#")
        ]

    def get_patterns(self) -> List[str]:
        """
        Get list of ignore patterns.

        Priority (highest to lowest):
        1. Explicit dockerignore list
        2. Explicit path to the .dockerignore file
        3. .dockerignore file in the context directory
        4. Default ignore patterns

        Returns:
            List of ignore patterns
        """
        # Explicit ignore list
        if self.dockerignore is not None:
            return self.dockerignore

        # Explicit .dockerignore path
        if self.dockerignore_path is not None:
            dockerignore_path = Path(self.dockerignore_path)
            if dockerignore_path.is_file():
                return self._read_dockerignore_file(dockerignore_path)
            else:
                raise FileNotFoundError(
                    f"Specified .dockerignore file not found: {self.dockerignore_path}"
                )

        # .dockerignore in context_dir
        if self.context_dir is not None:
            context_path = Path(self.context_dir)
            dockerignore_path = context_path / ".dockerignore"
            if dockerignore_path.is_file():
                return self._read_dockerignore_file(dockerignore_path)

        # Fallback to defaults
        return DEFAULT_DOCKERIGNORE_PATTERNS

    def get_regex_patterns(self) -> List[str]:
        from pathspec.patterns.gitwildmatch import GitWildMatchPattern

        patterns = self.get_patterns()
        regex_patterns = []
        for pattern in patterns:
            # Convert ignore patterns to regex, this way we can use `re` at runtime.
            regex, _ = GitWildMatchPattern.pattern_to_regex(pattern)
            if regex:
                regex_patterns.append(regex)
        return regex_patterns


@dataclass
class DockerfileParser:
    content: str
    normalized_content: str = field(init=False)

    def __post_init__(self) -> None:
        self.normalized_content = re.sub(r"\\\n\s*", " ", self.content)

    def parse_copy_add_sources(self) -> List[str]:
        """
        Parse COPY and ADD commands to extract source paths.
            - Skips COPY --from=... (multi-stage builds)
            - Skips ADD with URLs (http://, https://)
            - Normalizes absolute paths by stripping leading slash (Docker treats
              them as relative to the build context)
            - Handles both shell form and JSON form

        Returns:
            List of source paths/patterns referenced in COPY/ADD commands.
        """
        sources: List[str] = []

        # Regex to match COPY or ADD instructions
        instruction_pattern = re.compile(
            r"^(?P<instruction>COPY|ADD)\s+(?P<rest>.+?)(?:\s*\\)?$",
            re.MULTILINE | re.IGNORECASE,
        )

        for match in instruction_pattern.finditer(self.normalized_content):
            instruction = match.group("instruction").upper()
            rest = match.group("rest").strip()

            if not self._is_valid_source(rest, instruction):
                continue

            src_paths = self._parse_instruction_args(rest, instruction)
            sources.extend(src_paths)

        return sources

    def _is_valid_source(self, args: str, instruction: str) -> bool:
        # Skip COPY --from=... (multi-stage builds)
        if instruction == "COPY" and re.match(r"--from=", args, re.IGNORECASE):
            return False

        # Skip ADD with URLs (e.g., ADD https://example.com/file.tar.gz)
        # Use re.search to handle cases where flags precede the URL
        if instruction == "ADD" and re.search(r"https?://", args, re.IGNORECASE):
            return False

        return True

    def _parse_instruction_args(self, args_str: str, instruction: str) -> List[str]:
        """
        Parse arguments from COPY/ADD instruction.

        Args:
            args_str: The arguments string after the instruction
            instruction: The instruction type (COPY or ADD)

        Returns:
            List of valid source paths
        """
        src_paths: List[str] = []

        try:
            # Remove flags from the args string first
            # Matches: --flag or --flag=value or --flag="value"
            r = r'--\w+(?:=[^\s\]]+|="[^"]*")?'
            args_str_clean = re.sub(r, "", args_str).strip()

            if not args_str_clean:
                return []

            # Try JSON form first: ["src1", "src2", "dest"]
            if args_str_clean.endswith("]"):
                args = json.loads(args_str_clean)
            else:
                args = shlex.split(args_str_clean)

            if len(args) >= 2:
                # All but last are sources
                src_paths = args[:-1]

            # We normalize to relative paths (strip leading "/") because:
            # - They're semantically identical in Docker
            # - FileSync requires relative paths to context_dir
            return [src_path.lstrip("/") for src_path in src_paths]

        except (AttributeError, json.JSONDecodeError, TypeError, ValueError):
            from fal.console import console

            console.print(
                f"[yellow][WARNING][/yellow] Failed to parse instruction arguments: "
                f"{args_str}. Skipping this instruction. Please check the Dockerfile "
                "syntax and try again."
            )
            return []
