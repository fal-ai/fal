import json
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

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


class DockerfileParser:
    """Parses Dockerfile content to extract information."""

    def __init__(self, content: str):
        """Initialize parser with Dockerfile content.

        Args:
            content: The Dockerfile content as a string
        """
        self.content = content
        self._normalized_content: Optional[str] = None

    @property
    def normalized_content(self) -> str:
        """Get Dockerfile content with line continuations resolved."""
        if self._normalized_content is None:
            # Handle line continuations (backslash)
            self._normalized_content = re.sub(r"\\\n\s*", " ", self.content)
        return self._normalized_content

    def parse_copy_add_sources(self) -> List[str]:
        """Parse COPY and ADD commands to extract source paths.
            - Skips COPY --from=... (multi-stage builds)
            - Skips ADD with URLs (http://, https://)
            - Skips absolute paths (they reference paths inside the image)
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

            # Skip COPY --from=... (multi-stage builds)
            if instruction == "COPY" and re.match(r"--from=", rest, re.IGNORECASE):
                continue

            src_paths = self._parse_instruction_args(rest, instruction)
            sources.extend(src_paths)

        return sources

    def _parse_instruction_args(self, args_str: str, instruction: str) -> List[str]:
        """Parse arguments from COPY/ADD instruction.

        Args:
            args_str: The arguments string after the instruction
            instruction: The instruction type (COPY or ADD)

        Returns:
            List of valid source paths
        """
        src_paths: List[str] = []

        try:
            # Try JSON form first: ["src1", "src2", "dest"]
            if args_str.startswith("["):
                args = json.loads(args_str)
                if len(args) >= 2:
                    # All but last are sources
                    src_paths = args[:-1]
            else:
                # Shell form: handle flags like --chown, --chmod, --link
                # Remove flags (--flag=value or --flag value patterns)
                args_without_flags = re.sub(
                    r"--\w+(?:=\S+|\s+\S+)?", "", args_str
                ).strip()
                args = shlex.split(args_without_flags)
                if len(args) >= 2:
                    # All but last are sources
                    src_paths = args[:-1]

            # Filter out invalid paths
            return [
                src
                for src in src_paths
                if self._is_valid_local_source(src, instruction)
            ]

        except (json.JSONDecodeError, ValueError):
            # If parsing fails, skip this instruction
            return []

    @staticmethod
    def _is_valid_local_source(src: str, instruction: str) -> bool:
        """Check if source path is a valid local file reference.

        Args:
            src: The source path
            instruction: The instruction type (COPY or ADD)

        Returns:
            True if it's a valid local source path
        """
        # Skip URLs for ADD
        if instruction == "ADD" and re.match(r"https?://", src, re.IGNORECASE):
            return False
        # Skip absolute paths (they reference paths inside the image)
        if src.startswith("/"):
            from fal.console import console

            console.print(
                f"[yellow]WARNING: Skipping absolute source path '{src}' in "
                f"{instruction} command. Use relative paths instead (e.g., '.{src}')."
                "[/yellow]"
            )
            return False
        return True

    def get_workdir(self) -> Optional[str]:
        """Get the effective WORKDIR from the Dockerfile.

        Parses all WORKDIR directives and resolves relative paths according to
        Docker's behavior:
        - Absolute paths (starting with /) set the workdir directly
        - Relative paths are resolved against the current workdir
        - Paths with variables ($VAR or ${VAR}) are returned as-is

        Returns:
            The effective WORKDIR path, or None if no WORKDIR is specified.
        """
        from pathlib import PurePosixPath

        workdir = None

        workdir_pattern = re.compile(
            r"^\s*WORKDIR\s+(?P<path>.+?)\s*$",
            re.MULTILINE | re.IGNORECASE,
        )

        for match in workdir_pattern.finditer(self.normalized_content):
            path = match.group("path").strip()
            # Remove quotes if present
            if (path.startswith('"') and path.endswith('"')) or (
                path.startswith("'") and path.endswith("'")
            ):
                path = path[1:-1]

            # Check if path contains variables - if so, return as-is
            if "$" in path:
                # Path contains environment/build variables
                # We can't resolve these, so just set it directly
                workdir = path
            elif path.startswith("/"):
                # Absolute path - set directly
                workdir = path
            else:
                # Relative path - resolve against current workdir
                if workdir is None:
                    # First WORKDIR is relative, base it from root
                    workdir = "/"
                # Resolve the relative path and normalize (.., ., etc.)
                resolved = PurePosixPath(workdir) / path
                # Normalize by processing parts to handle .. and .
                parts: list[str] = []
                for part in resolved.parts:
                    if part == "/":
                        continue
                    elif part == "..":
                        # Go up one directory if possible
                        if parts:
                            parts.pop()
                    elif part != ".":
                        # Add normal parts, skip "."
                        parts.append(part)
                workdir = "/" + "/".join(parts) if parts else "/"

        return workdir

    def get_all_copy_destinations(self) -> List[str]:
        """Get all unique root destination directories from COPY/ADD commands.

        Parses all COPY/ADD commands that copy local files and extracts
        all absolute destination paths. Filters out subdirectories when a
        parent directory is already included (e.g., if /app and /app/src
        are both destinations, only /app is returned).

        Returns:
            List of unique root-level absolute destination directory paths.
            Empty list if no absolute destinations found.

        Example:
            >>> dockerfile = '''
            ... FROM python:3.11
            ... COPY config.yaml /etc/myapp/
            ... COPY app.py /app/
            ... COPY src/ /app/src/
            ... COPY data/ /data/
            ... '''
            >>> parser = DockerfileParser(dockerfile)
            >>> parser.get_all_copy_destinations()
            ['/etc/myapp', '/app', '/data']  # /app/src excluded (parent /app included)
        """
        destinations = []

        # Regex to match COPY or ADD instructions
        instruction_pattern = re.compile(
            r"^(?P<instruction>COPY|ADD)\s+(?P<rest>.+?)\s*$",
            re.MULTILINE | re.IGNORECASE,
        )

        for match in instruction_pattern.finditer(self.normalized_content):
            instruction = match.group("instruction").upper()
            rest = match.group("rest").strip()

            # Skip COPY --from=... (multi-stage builds)
            if instruction == "COPY" and re.match(r"--from=", rest, re.IGNORECASE):
                continue

            # Skip ADD with URLs
            if instruction == "ADD" and re.match(r"https?://", rest, re.IGNORECASE):
                continue

            # Parse the destination (last argument)
            dest = self._parse_copy_destination(rest)
            if dest and dest.startswith("/"):
                # Normalize destination (strip trailing filename if present)
                # e.g., /app/ -> /app, /app/file.txt -> /app
                if dest.endswith("/"):
                    normalized = dest.rstrip("/") or "/"
                else:
                    # If it looks like a file, use parent directory
                    parent = str(Path(dest).parent)
                    normalized = parent if parent != "." else "/"

                # Add to list if not already present
                if normalized not in destinations:
                    destinations.append(normalized)

        # Filter out subdirectories when parent is already in list
        # Sort by length (shorter paths first) to check parents first
        destinations.sort(key=len)
        filtered: list[str] = []
        for dest in destinations:
            # Check if any existing path is a parent of this dest
            is_subdir = False
            for existing in filtered:
                # Check if dest is a subdirectory of existing
                # e.g., /app/src is subdir of /app
                if dest.startswith(existing + "/") or dest == existing:
                    is_subdir = True
                    break
            if not is_subdir:
                filtered.append(dest)

        return filtered

    def _parse_copy_destination(self, args_str: str) -> Optional[str]:
        """Parse the destination from COPY/ADD arguments.

        Args:
            args_str: The arguments string after COPY/ADD

        Returns:
            The destination path, or None if parsing fails.
        """
        # Remove flags like --chown, --chmod, --link
        args_str = re.sub(r"--\w+(?:=\S+|\s+\S+)?\s*", "", args_str).strip()

        try:
            # Check for JSON form: ["src", "dest"]
            if args_str.startswith("["):
                args = json.loads(args_str)
                if len(args) >= 2:
                    return args[-1]  # Last element is destination
            else:
                # Shell form: src1 src2 ... dest
                parts = shlex.split(args_str)
                if parts:
                    return parts[-1]  # Last element is destination
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    def get_effective_workdir(self) -> Union[str, List[str], None]:
        """Get the effective working directory where files will be placed.

        Priority:
        1. WORKDIR directive (single canonical location)
        2. All absolute COPY destinations (when no WORKDIR)
        3. None if no WORKDIR and no absolute destinations

        Returns:
            - str: Single path if WORKDIR exists or only one absolute COPY destination
            - list[str]: Multiple paths if no WORKDIR and multiple absolute COPY dest
            - None: No WORKDIR and no absolute COPY destinations
        """
        # Priority 1: WORKDIR directive (single canonical location)
        if workdir := self.get_workdir():
            return workdir

        # Priority 2: All absolute COPY destinations (when no WORKDIR)
        copy_dests = self.get_all_copy_destinations()
        if copy_dests:
            # Return list if multiple, single string if one
            return copy_dests if len(copy_dests) > 1 else copy_dests[0]

        # No WORKDIR, no absolute destinations
        return None


class DockerignoreHandler:
    """Handles .dockerignore file loading and provides ignore patterns."""

    def __init__(
        self,
        context_dir: Optional[os.PathLike] = None,
        dockerignore: Optional[List[str]] = None,
        dockerignore_path: Optional[os.PathLike] = None,
    ):
        """Initialize handler with optional context directory.

        Priority for ignore patterns (highest to lowest):
        1. docker_ignore - explicit list of patterns
        2. docker_ignore_path - explicit path to .dockerignore file
        3. context_dir/.dockerignore - check for .dockerignore in context
        4. DEFAULT_DOCKERIGNORE_PATTERNS - fallback defaults

        Args:
            context_dir: Path to the build context directory
            docker_ignore: Explicit list of ignore patterns (highest priority)
            docker_ignore_path: Explicit path to .dockerignore file
        """
        self.context_dir = context_dir
        self.docker_ignore = dockerignore
        self.docker_ignore_path = dockerignore_path
        self._pathspec = None  # Cache for pathspec matcher

    @staticmethod
    def _read_dockerignore_file(path: Path) -> List[str]:
        """Read and parse .dockerignore file.

        Args:
            path: Path to .dockerignore file

        Returns:
            List of patterns (filtered, no comments or empty lines)
        """
        with open(path) as f:
            lines = f.read().splitlines()

        # Filter out empty lines and comments
        return [
            line.strip() for line in lines if line.strip() and not line.startswith("#")
        ]

    def get_patterns(self) -> List[str]:
        """Get list of ignore patterns following priority order.

        Priority (highest to lowest):
        1. Explicit docker_ignore list
        2. Explicit docker_ignore_path file
        3. .dockerignore in context_dir
        4. DEFAULT_DOCKERIGNORE_PATTERNS

        Returns:
            List of gitignore-style patterns

        Example:
            >>> # Explicit patterns
            >>> handler = DockerignoreHandler(docker_ignore=["*.log", "tmp/"])
            >>> patterns = handler.get_patterns()  # ["*.log", "tmp/"]
            >>>
            >>> # Explicit .dockerignore path
            >>> handler = DockerignoreHandler(docker_ignore_path=".dockerignore")
            >>> patterns = handler.get_patterns()  # reads from .dockerignore
            >>>
            >>> # context_dir/.dockerignore
            >>> handler = DockerignoreHandler(context_dir=".")
            >>> patterns = handler.get_patterns()  # checks ./.dockerignore
            >>>
            >>> # Defaults
            >>> handler = DockerignoreHandler()
            >>> patterns = handler.get_patterns()  # DEFAULT_DOCKERIGNORE_PATTERNS
        """
        # Explicit ignore list
        if self.docker_ignore is not None:
            return self.docker_ignore

        # Explicit .dockerignore path
        if self.docker_ignore_path is not None:
            dockerignore_path = Path(self.docker_ignore_path)
            if dockerignore_path.exists():
                return self._read_dockerignore_file(dockerignore_path)
            else:
                raise FileNotFoundError(
                    f"Specified .dockerignore file not found: {self.docker_ignore_path}"
                )

        # .dockerignore in context_dir
        if self.context_dir is not None:
            context_path = Path(self.context_dir)
            dockerignore_path = context_path / ".dockerignore"
            if dockerignore_path.exists():
                return self._read_dockerignore_file(dockerignore_path)

        # Fallback to defaults
        return DEFAULT_DOCKERIGNORE_PATTERNS

    def get_regex_patterns(self) -> List[str]:
        """Convert gitignore patterns to regex patterns.

        Uses pathspec's GitWildMatchPattern.pattern_to_regex() to convert
        gitignore-style patterns to regex. This allows using standard re.compile()
        for matching without requiring pathspec at runtime.

        Returns:
            List of regex pattern strings compatible with re.compile()
        """
        from pathspec.patterns.gitwildmatch import GitWildMatchPattern

        patterns = self.get_patterns()
        regex_patterns = []
        for pattern in patterns:
            regex, _ = GitWildMatchPattern.pattern_to_regex(pattern)
            if regex:
                regex_patterns.append(regex)
        return regex_patterns


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
    docker_context_dir: Optional[os.PathLike] = field(default=None)
    dockerignore: Optional[List[str]] = field(default=None)

    def __post_init__(self) -> None:
        # Validate dockerfile first
        if not self.dockerfile_str or not self.dockerfile_str.strip():
            raise ValueError(
                "Invalid dockerfile: Dockerfile is required.\n"
                "Either use ContainerImage.from_dockerfile_str() or "
                "ContainerImage.from_dockerfile() to create a container image."
            )

        # Set docker_context_dir to cwd if not provided
        if self.docker_context_dir is None:
            self.docker_context_dir = Path.cwd()

        # Initialize dockerignore patterns (converted to regex)
        # Priority: explicit list > .dockerignore file in context > defaults
        if self.dockerignore is None:
            # Check for .dockerignore file in context_dir, or use defaults
            handler = DockerignoreHandler(context_dir=self.docker_context_dir)
        else:
            # Use provided patterns
            handler = DockerignoreHandler(dockerignore=self.dockerignore)

        # Convert gitignore patterns to regex for use with re.compile()
        # This avoids requiring pathspec on workers
        self._dockerignore = handler.get_regex_patterns()

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

    @property
    def workdir(self) -> Union[str, List[str], None]:
        """
        Get the effective working directory where files are placed in the container.w

        This is determined by parsing the Dockerfile:
        1. First checks for WORKDIR directive
        2. Then checks for absolute COPY destinations (e.g., COPY . /app/)
        3. Falls back to "/" if neither is specified

        Returns:
            The directory path where copied files reside in the container.
        """
        # Parse the effective workdir from Dockerfile
        parser = DockerfileParser(self.dockerfile_str)
        return parser.get_effective_workdir()

    @classmethod
    def from_dockerfile_str(cls, text: str, **kwargs) -> "ContainerImage":
        """
        Create from Dockerfile string.

        Args:
            text: Dockerfile string
        """
        return cls(dockerfile_str=text, **kwargs)

    @classmethod
    def from_dockerfile(cls, path: str, **kwargs) -> "ContainerImage":
        """
        Create from Dockerfile path.

        Args:
            path: Path to the Dockerfile
        """
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
            "docker_context_dir": str(self.docker_context_dir)
            if self.docker_context_dir
            else None,
            "docker_files_list": self.get_copy_sources(),
            "docker_ignore": self._dockerignore,
            "workdir": self.workdir,
        }

    def get_copy_sources(self) -> List[str]:
        """
        Get list of src paths/patterns from COPY/ADD commands. This method only
        parses the Dockerfile - it doesn't access the filesystem.

        Returns:
            List of src paths (e.g., ["src/", "requirements.txt", "*.py"]) that can be
            passed to FileSync.sync_files(). Returns empty list if no COPY/ADD commands
            found.
        """
        parser = DockerfileParser(self.dockerfile_str)
        return parser.parse_copy_add_sources()

    def add_dockerignore(
        self,
        patterns: Optional[List[str]] = None,
        path: Optional[os.PathLike] = None,
    ) -> None:
        """Add or update dockerignore patterns.

        Sets the internal dockerignore patterns using gitignore-style matching.
        You can provide either a list of patterns or a path to a .dockerignore file.

        Patterns follow gitignore syntax:
        - * matches any filename (not /)
        - ** matches any path including /
        - / at end matches directories only
        - # for comments (in files)

        Args:
            patterns: List of gitignore-style patterns
            path: Path to a .dockerignore file

        Raises:
            ValueError: If both patterns and path are provided, or neither

        Examples:
            >>> # Gitignore-style patterns
            >>> img = ContainerImage.from_dockerfile_str("FROM python:3.11")
            >>> img.add_dockerignore(patterns=["*.log", "tmp/", "node_modules"])
            >>>
            >>> # Complex patterns with **
            >>> img.add_dockerignore(patterns=["**/*.pyc", "**/__pycache__/"])
            >>>
            >>> # Using a .dockerignore file path
            >>> img.add_dockerignore(path=".dockerignore")
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
            assert patterns is not None
            handler = DockerignoreHandler(dockerignore=patterns)

        # Convert gitignore patterns to regex
        self._dockerignore = handler.get_regex_patterns()

    def _matches_ignore_patterns(self, relative_path: str) -> bool:
        """
        Check if a path matches any of the dockerignore patterns.

        Args:
            relative_path: Path relative to docker_context_dir

        Returns:
            True if the path should be ignored
        """
        for pattern in self._dockerignore:
            if re.search(pattern, relative_path):
                return True
        return False

    def _resolve_copy_sources(self) -> List[Path]:
        """Resolve COPY/ADD sources to actual file paths.

        Returns:
            List of resolved file paths relative to docker_context_dir
        """
        if not self.docker_context_dir:
            return []

        context_path = Path(self.docker_context_dir)
        sources = self.get_copy_sources()
        resolved_files: List[Path] = []

        for source in sources:
            source_path = context_path / source

            if source_path.is_file():
                # Direct file reference
                resolved_files.append(Path(source))
            elif source_path.is_dir():
                # Directory - include all files recursively
                for file_path in source_path.rglob("*"):
                    if file_path.is_file():
                        rel_path = file_path.relative_to(context_path)
                        resolved_files.append(rel_path)
            elif "*" in source or "?" in source:
                # Glob pattern
                for file_path in context_path.glob(source):
                    if file_path.is_file():
                        rel_path = file_path.relative_to(context_path)
                        resolved_files.append(rel_path)
                    elif file_path.is_dir():
                        for sub_file in file_path.rglob("*"):
                            if sub_file.is_file():
                                rel_path = sub_file.relative_to(context_path)
                                resolved_files.append(rel_path)

        return resolved_files
