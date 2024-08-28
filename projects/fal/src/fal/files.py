from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import tomli


@lru_cache
def _load_toml(path: Union[Path, str]) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomli.load(f)


@lru_cache
def _cached_resolve(path: Path) -> Path:
    return path.resolve()


@lru_cache
def find_project_root(srcs: Optional[Sequence[str]]) -> Tuple[Path, str]:
    """Return a directory containing .git, or pyproject.toml.

    That directory will be a common parent of all files and directories
    passed in `srcs`.

    If no directory in the tree contains a marker that would specify it's the
    project root, the root of the file system is returned.

    Returns a two-tuple with the first element as the project root path and
    the second element as a string describing the method by which the
    project root was discovered.
    """
    if not srcs:
        srcs = [str(_cached_resolve(Path.cwd()))]

    path_srcs = [_cached_resolve(Path(Path.cwd(), src)) for src in srcs]

    # A list of lists of parents for each 'src'. 'src' is included as a
    # "parent" of itself if it is a directory
    src_parents = [
        list(path.parents) + ([path] if path.is_dir() else []) for path in path_srcs
    ]

    common_base = max(
        set.intersection(*(set(parents) for parents in src_parents)),
        key=lambda path: path.parts,
    )

    for directory in (common_base, *common_base.parents):
        if (directory / ".git").exists():
            return directory, ".git directory"

        if (directory / "pyproject.toml").is_file():
            pyproject_toml = _load_toml(directory / "pyproject.toml")
            if "fal" in pyproject_toml.get("tool", {}):
                return directory, "pyproject.toml"

    return directory, "file system root"


def find_pyproject_toml(
    path_search_start: Optional[Tuple[str, ...]] = None,
) -> Optional[str]:
    """Find the absolute filepath to a pyproject.toml if it exists"""
    path_project_root, _ = find_project_root(path_search_start)
    path_pyproject_toml = path_project_root / "pyproject.toml"

    if path_pyproject_toml.is_file():
        return str(path_pyproject_toml)


def parse_pyproject_toml(path_config: str) -> Dict[str, Any]:
    """Parse a pyproject toml file, pulling out relevant parts for fal.

    If parsing fails, will raise a tomli.TOMLDecodeError.
    """
    pyproject_toml = _load_toml(path_config)
    config: Dict[str, Any] = pyproject_toml.get("tool", {}).get("fal", {})
    config = {k.replace("--", "").replace("-", "_"): v for k, v in config.items()}

    return config
