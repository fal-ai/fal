"""Install fal agent skills into coding agents.

Skills come from the fal skills registry, whose ``index.json`` lists every
file of every skill with its sha256.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REGISTRY_URL = (
    "https://raw.githubusercontent.com/fal-ai-community/skills/refs/heads/main/skills"
)
REGISTRY_URL_ENV = "FAL_SKILLS_URL"

DEFAULT_SKILLS = ("fal-serverless", "fal-serverless-operate")

UNIVERSAL_DIR = ".agents/skills"

# Maps each file fal wrote to its sha256. A skill directory is fal's only while
# its contents match this record exactly; fal never changes any other.
MARKER = ".fal-skill"

MISSING = "missing"
UNMANAGED = "unmanaged"
MODIFIED = "modified"
CURRENT = "current"
OUTDATED = "outdated"

_SKILL_NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
# No trailing dot, which also rules out "." and "..".
_PATH_PART = re.compile(r"[A-Za-z0-9._-]+(?<!\.)")
_WINDOWS_RESERVED = re.compile(r"(con|prn|aux|nul|com\d|lpt\d)(\..*)?", re.I)


@dataclass(frozen=True)
class Agent:
    name: str
    # Relative to the project, or to $HOME with --global.
    skills_dir: str
    # Home-relative directory whose presence means the agent is installed.
    marker: str


AGENTS: Dict[str, Agent] = {
    agent.name: agent
    for agent in [
        Agent("universal", UNIVERSAL_DIR, ""),
        Agent("claude-code", ".claude/skills", ".claude"),
        Agent("codex", UNIVERSAL_DIR, ".codex"),
        Agent("cursor", UNIVERSAL_DIR, ".cursor"),
        Agent("gemini-cli", UNIVERSAL_DIR, ".gemini"),
        Agent("github-copilot", UNIVERSAL_DIR, ".copilot"),
        Agent("opencode", UNIVERSAL_DIR, ".config/opencode"),
        Agent("windsurf", UNIVERSAL_DIR, ".codeium/windsurf"),
    ]
}


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    # Relative POSIX path -> sha256.
    files: Dict[str, str]


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _check_name(name: str) -> str:
    if (
        len(name) > 64
        or not _SKILL_NAME.fullmatch(name)
        or _WINDOWS_RESERVED.fullmatch(name)
    ):
        raise ValueError(
            f"Invalid skill name {name!r}: skill names use lowercase letters, "
            "digits and single hyphens."
        )
    return name


def _check_path(path: str) -> str:
    parts = path.split("/")
    if path == MARKER or not all(
        _PATH_PART.fullmatch(part) and not _WINDOWS_RESERVED.fullmatch(part)
        for part in parts
    ):
        raise ValueError(f"Refusing unsafe skill file path {path!r}")
    return path


class Registry:
    def __init__(self) -> None:
        import httpx

        from fal._user_agent import USER_AGENT

        self.url = (os.environ.get(REGISTRY_URL_ENV) or REGISTRY_URL).rstrip("/")
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=30
        )
        self._skills: Dict[str, Skill] = {}

    def __enter__(self) -> Registry:
        return self

    def __exit__(self, *exc_info) -> None:
        self._client.close()

    def _get(self, path: str) -> bytes:
        import httpx

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
        if not self._skills:
            try:
                for entry in json.loads(self._get("index.json"))["skills"]:
                    skill = Skill(
                        name=_check_name(entry["name"]),
                        description=str(entry.get("description") or ""),
                        files={
                            _check_path(f["path"]): str(f["sha256"])
                            for f in entry["files"]
                        },
                    )
                    self._skills[skill.name] = skill
            except (KeyError, TypeError, ValueError) as exc:
                self._skills = {}
                reason = f"missing {exc}" if isinstance(exc, KeyError) else exc
                raise RuntimeError(
                    f"Malformed skills index at {self.url}/index.json: {reason}"
                ) from None
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

    def download(self, skill: Skill) -> Dict[str, bytes]:
        files = {}
        for path, sha256 in skill.files.items():
            content = self._get(f"{skill.name}/{path}")
            if _sha256(content) != sha256:
                raise RuntimeError(
                    f"Checksum mismatch for {skill.name}/{path}. The skills "
                    "registry may have just been updated; try again in a few "
                    "minutes."
                )
            files[path] = content
        return files


def _matches(content: bytes, sha256: str) -> bool:
    # A git checkout with autocrlf turns LF into CRLF; that is not a change.
    return sha256 in (_sha256(content), _sha256(content.replace(b"\r\n", b"\n")))


def _unchanged(root: Path, recorded: Dict[str, str]) -> bool:
    found = set()
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames + dirnames:
            path = Path(dirpath, name)
            relative = path.relative_to(root).as_posix()
            if relative == MARKER or (path.is_dir() and not path.is_symlink()):
                continue
            # Links are never followed, so they always count as changes.
            if path.is_symlink() or relative not in recorded:
                return False
            if not _matches(path.read_bytes(), recorded[relative]):
                return False
            found.add(relative)
    return found == set(recorded)


def _recorded(path: Path) -> Optional[Dict[str, str]]:
    try:
        if path.is_symlink() or not path.is_dir():
            return None
        files = json.loads((path / MARKER).read_text())["files"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return files if isinstance(files, dict) else None


def local_status(path: Path) -> str:
    """MISSING, UNMANAGED, MODIFIED, or CURRENT if fal's record still holds."""
    if not os.path.lexists(path):
        return MISSING
    recorded = _recorded(path)
    if recorded is None:
        return UNMANAGED
    try:
        return CURRENT if _unchanged(path, recorded) else MODIFIED
    except OSError:
        return MODIFIED


def skill_status(skill: Skill, path: Path) -> str:
    status = local_status(path)
    if status == CURRENT and _recorded(path) != skill.files:
        return OUTDATED
    return status


def write_skill(skill: Skill, files: Dict[str, bytes], skills_dir: Path) -> None:
    """Replace ``skills_dir / skill.name`` with ``files`` in one rename."""
    destination = skills_dir / skill.name
    skills_dir.mkdir(parents=True, exist_ok=True)
    # Nested one level down, so a leftover never looks like a skill to an agent.
    temp = skills_dir / f".fal-skill-{os.urandom(6).hex()}"
    new, old = temp / "new", temp / "old"
    try:
        for relative, content in files.items():
            path = new.joinpath(*relative.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (new / MARKER).write_text(json.dumps({"files": skill.files}, indent=2))
        if os.path.lexists(destination):
            if _recorded(destination) is None:
                raise FileExistsError(f"{destination} was not installed by fal")
            os.replace(destination, old)
        try:
            os.replace(new, destination)
        except OSError:
            if os.path.lexists(old):
                os.replace(old, destination)
            raise
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def detect_agents(home: Path) -> List[str]:
    return [
        name
        for name, agent in AGENTS.items()
        if not agent.marker or (home / agent.marker).is_dir()
    ]


def _targets(args, default_agents: List[str]) -> Dict[Path, List[str]]:
    """Map each skills directory to the agents to name for it."""
    home = Path.home()
    base = home if args.is_global else Path.cwd()
    shown = args.agents or detect_agents(home)
    groups: Dict[Path, List[str]] = {}
    for name in dict.fromkeys(args.agents or default_agents):
        groups.setdefault(base / AGENTS[name].skills_dir, []).append(name)
    return {
        path: [name for name in names if name in shown] or names
        for path, names in groups.items()
    }


def _shown(args, path: str) -> str:
    base = Path.home() if args.is_global else Path.cwd()
    relative = Path(path).relative_to(base).as_posix()
    return f"~/{relative}" if args.is_global else relative


def _change(skill: str, path: Path, agents: List[str], action: str, **extra) -> dict:
    return {
        "skill": skill,
        "path": str(path),
        "agents": agents,
        "action": action,
        **extra,
    }


def _sync(args, registry: Registry, skills: List[Skill], targets, *, update_only):
    plan = []
    for skills_dir, agents in targets.items():
        for skill in skills:
            status = skill_status(skill, skills_dir / skill.name)
            if update_only and status in (MISSING, UNMANAGED):
                continue
            plan.append((skill, skills_dir, agents, status))

    writes = {MISSING, OUTDATED} | ({MODIFIED} if args.force else set())
    # Download everything first so a registry error changes nothing on disk.
    downloads: Dict[str, Dict[str, bytes]] = {}
    for skill, _, _, status in plan:
        if status in writes and skill.name not in downloads:
            downloads[skill.name] = registry.download(skill)

    changes = []
    for skill, skills_dir, agents, status in plan:
        path = skills_dir / skill.name
        if status == CURRENT:
            changes.append(_change(skill.name, path, agents, "unchanged"))
        elif status not in writes:
            changes.append(_change(skill.name, path, agents, "skipped", reason=status))
        else:
            try:
                write_skill(skill, downloads[skill.name], skills_dir)
            except OSError as exc:
                changes.append(
                    _change(skill.name, path, agents, "failed", error=str(exc))
                )
            else:
                action = "installed" if status == MISSING else "updated"
                changes.append(_change(skill.name, path, agents, action))
    return changes


_SKIP_HINTS = {
    UNMANAGED: "not installed by fal, so it is left as is",
    MODIFIED: "changed since fal installed it; pass --force to discard the changes",
}


def _print_json(args, data: dict) -> None:
    args.console.print(
        json.dumps(data), markup=False, highlight=False, emoji=False, soft_wrap=True
    )


def _print_changes(args, changes: List[dict], empty_message: str) -> int:
    from rich.markup import escape

    from fal.console.encoding import make_terminal_safe
    from fal.console.icons import get_check_icon, get_cross_icon

    failed = int(any(change["action"] == "failed" for change in changes))
    if args.output == "json":
        _print_json(args, {"skills": changes})
        return failed
    if not changes:
        args.console.print(empty_message)
        return 0

    def plain(text: str) -> str:
        return escape(make_terminal_safe(text, args.console.file))

    icons = {"failed": get_cross_icon(args.console), "skipped": "[bold yellow]![/]"}
    for change in changes:
        line = (
            f"{icons.get(change['action'], get_check_icon(args.console))} "
            f"{change['action']} [bold]{change['skill']}[/] in "
            f"{plain(_shown(args, change['path']))} ({', '.join(change['agents'])})"
        )
        if "reason" in change:
            line += f": {_SKIP_HINTS[change['reason']]}"
        elif "error" in change:
            line += f": {plain(change['error'])}"
        args.console.print(line, highlight=False, emoji=False)
    return failed


def _scope(args) -> str:
    return "in your home directory." if args.is_global else "in this project."


def _install(args):
    if args.all and args.names:
        raise ValueError("Pass skill names or --all, not both.")
    targets = _targets(args, detect_agents(Path.home()))
    with Registry() as registry:
        names = registry.skills if args.all else args.names or DEFAULT_SKILLS
        skills = registry.resolve(names)
        changes = _sync(args, registry, skills, targets, update_only=False)
    return _print_changes(args, changes, "No skills to install.")


def _update(args):
    targets = _targets(args, list(AGENTS))
    with Registry() as registry:
        skills = list(registry.skills.values())
        changes = _sync(args, registry, skills, targets, update_only=True)
    return _print_changes(args, changes, f"No skills installed by fal {_scope(args)}")


def _remove(args):
    names = [_check_name(name) for name in dict.fromkeys(args.names)]
    changes = []
    for skills_dir, agents in _targets(args, list(AGENTS)).items():
        for name in names:
            path = skills_dir / name
            status = local_status(path)
            if status == MISSING:
                continue
            if status == UNMANAGED or (status == MODIFIED and not args.force):
                changes.append(_change(name, path, agents, "skipped", reason=status))
                continue
            try:
                shutil.rmtree(path)
            except OSError as exc:
                changes.append(_change(name, path, agents, "failed", error=str(exc)))
            else:
                changes.append(_change(name, path, agents, "removed"))
    return _print_changes(
        args, changes, f"No matching skills are installed {_scope(args)}"
    )


def _list(args):
    targets = _targets(args, list(AGENTS))
    with Registry() as registry:
        rows = []
        for skill in registry.skills.values():
            installed: List[dict] = []
            for skills_dir, agents in targets.items():
                path = skills_dir / skill.name
                status = skill_status(skill, path)
                if status != MISSING:
                    installed.append(
                        {"path": str(path), "agents": agents, "status": status}
                    )
            rows.append((skill, installed))

    if args.output == "json":
        data = {
            "registry": registry.url,
            "skills": [
                {
                    "skill": skill.name,
                    "description": skill.description,
                    "default": skill.name in DEFAULT_SKILLS,
                    "installed": installed,
                }
                for skill, installed in rows
            ],
        }
        _print_json(args, data)
        return

    from rich.table import Table
    from rich.text import Text

    from fal.console.encoding import make_terminal_safe

    def cell(text: str) -> Text:
        return Text(make_terminal_safe(text, args.console.file))

    table = Table()
    table.add_column("Skill", no_wrap=True)
    table.add_column("Installed", no_wrap=True)
    table.add_column("Description")
    for skill, installed in rows:
        name = skill.name + (" (default)" if skill.name in DEFAULT_SKILLS else "")
        where = "\n".join(
            f"{i['status']}: {_shown(args, i['path'])}" for i in installed
        )
        summary = skill.description.split(". ", 1)[0].rstrip(".")
        table.add_row(cell(name), cell(where or "-"), cell(summary))
    args.console.print(table)


def _add_target_arguments(parser, *, agents_default: str, force: str = "") -> None:
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
    if force:
        parser.add_argument("--force", action="store_true", help=force)


DETECTED_AGENTS_HELP = (
    "Defaults to the agents found in your home directory, plus the shared "
    ".agents/skills."
)
ALL_AGENTS_HELP = "Defaults to every known agent."
FORCE_HELP = "Also {} skills fal installed that have local changes."


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
    _add_target_arguments(
        install_parser,
        agents_default=DETECTED_AGENTS_HELP,
        force=FORCE_HELP.format("replace"),
    )
    install_parser.set_defaults(func=_install)

    update_help = (
        "Update the skills fal installed in this project, or in your home "
        "directory with --global."
    )
    update_parser = subparsers.add_parser(
        "update",
        description=update_help,
        help=update_help,
        parents=command_parents,
    )
    _add_target_arguments(
        update_parser,
        agents_default=ALL_AGENTS_HELP,
        force=FORCE_HELP.format("replace"),
    )
    update_parser.set_defaults(func=_update)

    list_help = "List registry skills and where they are installed."
    list_parser = subparsers.add_parser(
        "list",
        aliases=["ls"],
        description=list_help,
        help=list_help,
        parents=command_parents,
    )
    _add_target_arguments(list_parser, agents_default=ALL_AGENTS_HELP)
    list_parser.set_defaults(func=_list)

    remove_help = (
        "Remove skills fal installed from this project, or from your home "
        "directory with --global."
    )
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
    _add_target_arguments(
        remove_parser,
        agents_default=ALL_AGENTS_HELP,
        force=FORCE_HELP.format("remove"),
    )
    remove_parser.set_defaults(func=_remove)
