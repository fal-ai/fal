"""Install fal agent skills into coding agents.

Skills come from the fal skills registry, whose ``index.json`` lists every
file of every skill with its sha256. Files are copied, never symlinked, so
installs behave the same on Windows.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

if TYPE_CHECKING:
    import httpx

REGISTRY_URL = (
    "https://raw.githubusercontent.com/fal-ai-community/skills/refs/heads/main/skills"
)
REGISTRY_URL_ENV = "FAL_SKILLS_URL"

DEFAULT_SKILLS = ("fal-serverless", "fal-serverless-operate")

# Most agents read the shared ``.agents/skills``, in a project and in $HOME.
UNIVERSAL_DIR = ".agents/skills"

# Marks a skill directory written by fal; ``update`` only touches these.
MARKER = ".fal-skill"

_SKILL_NAME = re.compile(r"[a-z0-9]([a-z0-9._-]*[a-z0-9])?")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_WINDOWS_INVALID = set('<>:"/\\|?*') | {chr(i) for i in range(32)}


@dataclass(frozen=True)
class Agent:
    name: str
    project_dir: str
    global_dir: str
    # Home-relative path whose presence means the agent is installed.
    marker: str


AGENTS: Dict[str, Agent] = {
    agent.name: agent
    for agent in [
        Agent("universal", UNIVERSAL_DIR, UNIVERSAL_DIR, ""),
        Agent("claude-code", ".claude/skills", ".claude/skills", ".claude"),
        Agent("codex", UNIVERSAL_DIR, UNIVERSAL_DIR, ".codex"),
        Agent("cursor", UNIVERSAL_DIR, ".cursor/skills", ".cursor"),
        Agent("gemini-cli", UNIVERSAL_DIR, ".gemini/skills", ".gemini"),
        Agent("github-copilot", UNIVERSAL_DIR, ".copilot/skills", ".copilot"),
        Agent(
            "opencode",
            UNIVERSAL_DIR,
            ".config/opencode/skills",
            ".config/opencode",
        ),
        Agent("windsurf", UNIVERSAL_DIR, UNIVERSAL_DIR, ".codeium/windsurf"),
    ]
}


@dataclass(frozen=True)
class SkillFile:
    path: PurePosixPath
    sha256: str


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    files: List[SkillFile]


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _matches(content: bytes, sha256: str) -> bool:
    # Git's autocrlf rewrites line endings of skills committed to a repo.
    return sha256 in (_sha256(content), _sha256(content.replace(b"\r\n", b"\n")))


def _is_safe_component(part: str) -> bool:
    """Whether ``part`` names one entry inside a directory on every OS."""
    return (
        part not in ("", ".", "..")
        and not _WINDOWS_INVALID.intersection(part)
        and part == part.rstrip(". ")
        and part.split(".")[0].upper() not in _WINDOWS_RESERVED
    )


def _safe_relative_path(path: str) -> PurePosixPath:
    if (
        PurePosixPath(path).is_absolute()
        or PureWindowsPath(path).anchor
        or not all(_is_safe_component(part) for part in path.split("/"))
    ):
        raise ValueError(f"Refusing unsafe skill path: {path!r}")
    return PurePosixPath(path)


def _check_skill_name(name: str) -> str:
    if not _SKILL_NAME.fullmatch(name) or not _is_safe_component(name):
        raise ValueError(f"Refusing unsafe skill name: {name!r}")
    return name


def _skill_dir(skills_dir: Path, name: str) -> Path:
    path = skills_dir / _check_skill_name(name)
    # Lexical, so a symlinked skill is still removed as a link.
    if Path(os.path.abspath(path)).parent != Path(os.path.abspath(skills_dir)):
        raise ValueError(f"Refusing unsafe skill name: {name!r}")
    return path


class Registry:
    def __init__(self) -> None:
        self.url = (os.environ.get(REGISTRY_URL_ENV) or REGISTRY_URL).rstrip("/")
        self._skills: Optional[Dict[str, Skill]] = None
        self._files: Dict[str, Dict[PurePosixPath, bytes]] = {}
        self._client: Optional[httpx.Client] = None

    def __enter__(self) -> Registry:
        return self

    def __exit__(self, *exc_info) -> None:
        if self._client is not None:
            self._client.close()

    def _get(self, path: str) -> bytes:
        import httpx

        if self._client is None:
            from fal._user_agent import USER_AGENT

            self._client = httpx.Client(
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
                timeout=30,
            )
        url = f"{self.url}/{path}"
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Could not reach the fal skills registry at {url}: {exc}"
            ) from None
        if response.status_code != 200:
            raise RuntimeError(f"GET {url} returned {response.status_code}")
        return response.content

    @property
    def skills(self) -> Dict[str, Skill]:
        if self._skills is None:
            try:
                index = json.loads(self._get("index.json"))
                skills = [
                    Skill(
                        name=_check_skill_name(entry["name"]),
                        description=str(entry.get("description") or ""),
                        files=[
                            SkillFile(_safe_relative_path(f["path"]), f["sha256"])
                            for f in entry["files"]
                        ],
                    )
                    for entry in index["skills"]
                ]
            except (KeyError, TypeError, ValueError) as exc:
                reason = f"missing {exc}" if isinstance(exc, KeyError) else exc
                raise RuntimeError(
                    f"Malformed skills index at {self.url}/index.json: {reason}"
                ) from None
            self._skills = {skill.name: skill for skill in skills}
        return self._skills

    def resolve(self, names: Iterable[str]) -> List[Skill]:
        names = list(dict.fromkeys(names))
        unknown = [name for name in names if name not in self.skills]
        if unknown:
            raise ValueError(
                f"Unknown skill(s): {', '.join(unknown)}. "
                "Run 'fal skills list' to see available skills."
            )
        return [self.skills[name] for name in names]

    def download(self, skill: Skill) -> Dict[PurePosixPath, bytes]:
        if skill.name not in self._files:
            files = {}
            for file in skill.files:
                content = self._get(f"{skill.name}/{file.path}")
                if _sha256(content) != file.sha256:
                    raise RuntimeError(
                        f"Checksum mismatch for {skill.name}/{file.path}. The "
                        "skills registry may have just been updated; try again "
                        "in a few minutes."
                    )
                files[file.path] = content
            self._files[skill.name] = files
        return self._files[skill.name]


def detect_agents(home: Path) -> List[str]:
    detected = [
        name
        for name, agent in AGENTS.items()
        if agent.marker and (home / agent.marker).is_dir()
    ]
    return ["universal", *detected]


def target_dirs(
    agents: Iterable[str], *, is_global: bool, root: Path, home: Path
) -> Dict[Path, List[str]]:
    """Map each skills directory to the agents that read it."""
    targets: Dict[Path, List[str]] = {}
    for name in dict.fromkeys(agents):
        agent = AGENTS[name]
        path = home / agent.global_dir if is_global else root / agent.project_dir
        targets.setdefault(path, []).append(name)
    return targets


def skill_status(skill: Skill, skills_dir: Path) -> str:
    installed = skills_dir / skill.name
    if not installed.is_dir():
        return "missing"
    if not (installed / MARKER).is_file():
        return "unmanaged"
    on_disk = {
        PurePosixPath(path.relative_to(installed).as_posix())
        for path in installed.rglob("*")
        if path.is_file()
    }
    if on_disk != {file.path for file in skill.files} | {PurePosixPath(MARKER)}:
        return "outdated"
    for file in skill.files:
        if not _matches((installed / file.path).read_bytes(), file.sha256):
            return "outdated"
    return "current"


def _is_link(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    # A Windows junction is a directory link that ``is_symlink`` misses.
    junction = getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", None)
    return stat.S_ISLNK(info.st_mode) or (
        junction is not None and getattr(info, "st_reparse_tag", 0) == junction
    )


def _clear_readonly(func, path, _exc) -> None:
    # Windows refuses to delete read-only files.
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _remove(path: Path) -> None:
    if _is_link(path) or path.is_file():
        path.unlink()
    elif path.is_dir():
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=_clear_readonly)  # type: ignore[call-arg]
        else:
            shutil.rmtree(path, onerror=_clear_readonly)


def write_skill(
    skill: Skill, files: Dict[PurePosixPath, bytes], skills_dir: Path
) -> None:
    destination = _skill_dir(skills_dir, skill.name)
    # Build the new copy beside the old one and swap them, so a failed write
    # never leaves a partial install.
    staging = skills_dir / f".{skill.name}.{os.getpid()}.tmp"
    backup = skills_dir / f".{skill.name}.{os.getpid()}.old"
    _remove(staging)
    staging.mkdir(parents=True)
    try:
        for relative, content in files.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (staging / MARKER).write_text(
            "Installed by `fal skills`, which replaces this directory on update.\n"
        )
        if destination.exists() or _is_link(destination):
            os.replace(destination, backup)
            try:
                os.replace(staging, destination)
            except BaseException:
                os.replace(backup, destination)
                raise
            _remove(backup)
        else:
            os.replace(staging, destination)
    finally:
        _remove(staging)


def _sync(
    registry: Registry,
    skills: List[Skill],
    targets: Dict[Path, List[str]],
    *,
    managed_only: bool = False,
) -> List[dict]:
    plan: List[Tuple[Skill, Path, List[str], str]] = []
    for skills_dir, agents in targets.items():
        for skill in skills:
            status = skill_status(skill, skills_dir)
            if managed_only and status in ("missing", "unmanaged"):
                continue
            plan.append((skill, skills_dir, agents, status))

    # Download everything first so a registry error changes nothing on disk.
    for skill, _, _, status in plan:
        if status not in ("current", "unmanaged"):
            registry.download(skill)

    changes = []
    for skill, skills_dir, agents, status in plan:
        if status == "current":
            action = "unchanged"
        elif status == "unmanaged":
            # Never replace a skill that fal did not install.
            action = "skipped (not installed by fal)"
        else:
            write_skill(skill, registry.download(skill), skills_dir)
            action = "installed" if status == "missing" else "updated"
        changes.append(
            {
                "skill": skill.name,
                "path": str(skills_dir / skill.name),
                "agents": agents,
                "action": action,
            }
        )
    return changes


def _resolve_targets(args, default_agents=None) -> Dict[Path, List[str]]:
    home = Path.home()
    agents = args.agents or default_agents or detect_agents(home)
    return target_dirs(agents, is_global=args.is_global, root=Path.cwd(), home=home)


def _print_json(args, data: dict) -> None:
    args.console.print(
        json.dumps(data), markup=False, highlight=False, emoji=False, soft_wrap=True
    )


def _plain(args, text: str) -> str:
    from rich.markup import escape

    from fal.console.encoding import make_terminal_safe

    return escape(make_terminal_safe(text, args.console.file))


def _print_changes(args, changes: List[dict], empty_message: str) -> None:
    from fal.console.icons import get_check_icon

    if args.output == "json":
        _print_json(args, {"skills": changes})
        return

    if not changes:
        args.console.print(empty_message)
        return

    icon = get_check_icon(args.console)
    for change in changes:
        args.console.print(
            f"{icon} {change['action']} [bold]{_plain(args, change['skill'])}[/] "
            f"in {_plain(args, change['path'])} "
            f"({_plain(args, ', '.join(change['agents']))})",
            highlight=False,
            emoji=False,
        )


def _install(args):
    with Registry() as registry:
        if args.all:
            names: Iterable[str] = registry.skills
        else:
            names = args.names or DEFAULT_SKILLS
        changes = _sync(registry, registry.resolve(names), _resolve_targets(args))
    _print_changes(args, changes, "No skills to install.")


def _update(args):
    with Registry() as registry:
        changes = _sync(
            registry,
            list(registry.skills.values()),
            _resolve_targets(args, list(AGENTS)),
            managed_only=True,
        )
    _print_changes(args, changes, "No installed registry skills found.")


def _remove_skills(args):
    names = [_check_skill_name(name) for name in dict.fromkeys(args.names)]
    changes = []
    # Removal checks every known agent so a stray copy is never left behind.
    for skills_dir, agents in _resolve_targets(args, list(AGENTS)).items():
        for name in names:
            path = _skill_dir(skills_dir, name)
            if not (path.exists() or _is_link(path)):
                continue
            if (path / MARKER).is_file():
                _remove(path)
                action = "removed"
            else:
                action = "skipped (not installed by fal)"
            changes.append(
                {"skill": name, "path": str(path), "agents": agents, "action": action}
            )
    _print_changes(args, changes, "No matching skills are installed.")


def _list(args):
    targets = _resolve_targets(args)
    rows: List[dict] = []
    with Registry() as registry:
        for skill in registry.skills.values():
            statuses = {
                str(skills_dir): skill_status(skill, skills_dir)
                for skills_dir in targets
            }
            rows.append(
                {
                    "skill": skill.name,
                    "description": skill.description,
                    "default": skill.name in DEFAULT_SKILLS,
                    "installed": {
                        path: status
                        for path, status in statuses.items()
                        if status != "missing"
                    },
                }
            )

    if args.output == "json":
        _print_json(args, {"registry": registry.url, "skills": rows})
        return

    from rich.table import Table
    from rich.text import Text

    from fal.console.encoding import make_terminal_safe

    def cell(text: str) -> Text:
        return Text(make_terminal_safe(text, args.console.file))

    table = Table()
    table.add_column("Skill", no_wrap=True)
    table.add_column("Installed")
    table.add_column("Description")
    for row in rows:
        name = row["skill"] + (" (default)" if row["default"] else "")
        installed = "\n".join(
            f"{status}: {path}" for path, status in row["installed"].items()
        )
        summary = row["description"].split(". ", 1)[0].rstrip(".")
        table.add_row(cell(name), cell(installed or "-"), cell(summary))
    args.console.print(table)


def _add_target_arguments(parser, *, agents_default: str) -> None:
    parser.add_argument(
        "--agent",
        "-a",
        dest="agents",
        action="append",
        choices=list(AGENTS),
        help=f"Agent to target; repeat for several. {agents_default}",
    )
    parser.add_argument(
        "--global",
        dest="is_global",
        action="store_true",
        help="Use the user-level skills directory instead of the current project.",
    )


DETECTED_AGENTS_HELP = (
    "Defaults to the agents found in your home directory, plus the shared "
    ".agents/skills."
)
ALL_AGENTS_HELP = "Defaults to every known agent."


def add_parser(main_subparsers, parents):
    from .parser import FalParser, get_output_parser

    skills_help = "Install fal agent skills for coding agents."
    parser = main_subparsers.add_parser(
        "skills",
        aliases=["skill"],
        parents=parents,
        description=skills_help,
        help=skills_help,
    )
    subparsers = parser.add_subparsers(
        title="Commands",
        metavar="command",
        required=True,
        parser_class=FalParser,
    )
    command_parents = [*parents, get_output_parser()]

    install_help = "Install skills from the fal skills registry; re-run to update."
    install_parser = subparsers.add_parser(
        "install",
        aliases=["add"],
        description=install_help,
        help=install_help,
        parents=command_parents,
        epilog=(
            "Examples:\n"
            "  fal skills install\n"
            "  fal skills install fal-serverless --agent claude-code --global\n"
            "  fal skills install --all"
        ),
    )
    install_parser.add_argument(
        "names",
        metavar="SKILL",
        nargs="*",
        help=f"Skills to install. Defaults to {', '.join(DEFAULT_SKILLS)}.",
    )
    install_parser.add_argument(
        "--all",
        action="store_true",
        help="Install every skill in the registry.",
    )
    _add_target_arguments(install_parser, agents_default=DETECTED_AGENTS_HELP)
    install_parser.set_defaults(func=_install)

    update_help = "Update skills that fal installed, wherever they are."
    update_parser = subparsers.add_parser(
        "update",
        description=update_help,
        help=update_help,
        parents=command_parents,
    )
    _add_target_arguments(update_parser, agents_default=ALL_AGENTS_HELP)
    update_parser.set_defaults(func=_update)

    list_help = "List registry skills and where they are installed."
    list_parser = subparsers.add_parser(
        "list",
        aliases=["ls"],
        description=list_help,
        help=list_help,
        parents=command_parents,
    )
    _add_target_arguments(list_parser, agents_default=DETECTED_AGENTS_HELP)
    list_parser.set_defaults(func=_list)

    remove_help = "Remove installed skills."
    remove_parser = subparsers.add_parser(
        "remove",
        aliases=["rm"],
        description=remove_help,
        help=remove_help,
        parents=command_parents,
    )
    remove_parser.add_argument(
        "names", metavar="SKILL", nargs="+", help="Skills to remove."
    )
    _add_target_arguments(remove_parser, agents_default=ALL_AGENTS_HELP)
    remove_parser.set_defaults(func=_remove_skills)
