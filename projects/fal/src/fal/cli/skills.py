"""Install fal agent skills into coding agents.

Skills come from the fal skills registry, whose ``index.json`` lists every
file of every skill with its sha256. Files are copied, never symlinked, so
installs behave the same on Windows.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional

REGISTRY_URL = (
    "https://raw.githubusercontent.com/fal-ai-community/skills/refs/heads/main/skills"
)
REGISTRY_URL_ENV = "FAL_SKILLS_URL"

DEFAULT_SKILLS = ("fal-serverless", "fal-serverless-operate")

# Several agents read the shared ``.agents/skills`` directory in a project.
UNIVERSAL_PROJECT_DIR = ".agents/skills"


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
        Agent("universal", UNIVERSAL_PROJECT_DIR, ".config/agents/skills", ""),
        Agent("claude-code", ".claude/skills", ".claude/skills", ".claude"),
        Agent("codex", UNIVERSAL_PROJECT_DIR, ".codex/skills", ".codex"),
        Agent("cursor", UNIVERSAL_PROJECT_DIR, ".cursor/skills", ".cursor"),
        Agent("gemini-cli", UNIVERSAL_PROJECT_DIR, ".gemini/skills", ".gemini"),
        Agent("github-copilot", UNIVERSAL_PROJECT_DIR, ".copilot/skills", ".copilot"),
        Agent(
            "opencode",
            UNIVERSAL_PROJECT_DIR,
            ".config/opencode/skills",
            ".config/opencode",
        ),
        Agent(
            "windsurf",
            ".windsurf/skills",
            ".codeium/windsurf/skills",
            ".codeium/windsurf",
        ),
    ]
}


@dataclass(frozen=True)
class SkillFile:
    path: str
    sha256: str


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    files: List[SkillFile]


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _safe_relative_path(path: str) -> PurePosixPath:
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Refusing unsafe skill path: {path!r}")
    return relative


class Registry:
    def __init__(self, url: Optional[str] = None):
        self.url = (url or os.environ.get(REGISTRY_URL_ENV) or REGISTRY_URL).rstrip("/")
        self._skills: Optional[Dict[str, Skill]] = None

    def _get(self, path: str) -> bytes:
        import httpx

        from fal._user_agent import USER_AGENT

        url = f"{self.url}/{path}"
        response = httpx.get(
            url,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"GET {url} returned {response.status_code}")
        return response.content

    @property
    def skills(self) -> Dict[str, Skill]:
        if self._skills is None:
            index = json.loads(self._get("index.json"))
            self._skills = {
                entry["name"]: Skill(
                    name=entry["name"],
                    description=entry.get("description", ""),
                    files=[
                        SkillFile(path=f["path"], sha256=f["sha256"])
                        for f in entry["files"]
                    ],
                )
                for entry in index["skills"]
            }
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
        files = {}
        for file in skill.files:
            relative = _safe_relative_path(file.path)
            content = self._get(f"{skill.name}/{relative}")
            if _sha256(content) != file.sha256:
                raise RuntimeError(
                    f"Checksum mismatch for {skill.name}/{relative}; "
                    "the skills registry index may be stale."
                )
            files[relative] = content
        return files


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
    for name in agents:
        agent = AGENTS[name]
        path = home / agent.global_dir if is_global else root / agent.project_dir
        targets.setdefault(path, []).append(name)
    return targets


def skill_status(skill: Skill, skills_dir: Path) -> str:
    installed = skills_dir / skill.name
    if not installed.is_dir():
        return "missing"
    for file in skill.files:
        path = installed / _safe_relative_path(file.path)
        if not path.is_file() or _sha256(path.read_bytes()) != file.sha256:
            return "outdated"
    return "current"


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def write_skill(
    skill: Skill, files: Dict[PurePosixPath, bytes], skills_dir: Path
) -> None:
    destination = skills_dir / skill.name
    _remove(destination)
    for relative, content in files.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _sync(
    registry: Registry, skills: List[Skill], targets: Dict[Path, List[str]]
) -> List[dict]:
    changes = []
    downloaded: Dict[str, Dict[PurePosixPath, bytes]] = {}
    for skills_dir, agents in targets.items():
        for skill in skills:
            status = skill_status(skill, skills_dir)
            if status == "current":
                action = "unchanged"
            else:
                if skill.name not in downloaded:
                    downloaded[skill.name] = registry.download(skill)
                write_skill(skill, downloaded[skill.name], skills_dir)
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


def _print_changes(args, changes: List[dict], empty_message: str) -> None:
    from fal.console.icons import get_check_icon

    if args.output == "json":
        args.console.print(json.dumps({"skills": changes}))
        return

    if not changes:
        args.console.print(empty_message)
        return

    icon = get_check_icon(args.console)
    for change in changes:
        args.console.print(
            f"{icon} {change['action']} [bold]{change['skill']}[/] "
            f"in {change['path']} ({', '.join(change['agents'])})",
            highlight=False,
        )


def _install(args):
    registry = Registry()
    if args.all:
        names: Iterable[str] = registry.skills
    else:
        names = args.names or DEFAULT_SKILLS
    changes = _sync(registry, registry.resolve(names), _resolve_targets(args))
    _print_changes(args, changes, "No skills to install.")


def _update(args):
    registry = Registry()
    targets = target_dirs(
        AGENTS, is_global=args.is_global, root=Path.cwd(), home=Path.home()
    )
    changes = []
    for skills_dir, agents in targets.items():
        installed = [
            skill
            for skill in registry.skills.values()
            if skill_status(skill, skills_dir) != "missing"
        ]
        changes.extend(_sync(registry, installed, {skills_dir: agents}))
    _print_changes(args, changes, "No installed registry skills found.")


def _remove_skills(args):
    changes = []
    # Removal checks every known agent so a stray copy is never left behind.
    for skills_dir, agents in _resolve_targets(args, list(AGENTS)).items():
        for name in dict.fromkeys(args.names):
            path = skills_dir / _safe_relative_path(name)
            if path.exists() or path.is_symlink():
                _remove(path)
                changes.append(
                    {
                        "skill": name,
                        "path": str(path),
                        "agents": agents,
                        "action": "removed",
                    }
                )
    _print_changes(args, changes, "No matching skills are installed.")


def _list(args):
    registry = Registry()
    targets = _resolve_targets(args)
    rows: List[dict] = []
    for skill in registry.skills.values():
        statuses = {
            str(skills_dir): skill_status(skill, skills_dir) for skills_dir in targets
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
        args.console.print(json.dumps({"registry": registry.url, "skills": rows}))
        return

    from rich.table import Table

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
        table.add_row(name, installed or "-", summary)
    args.console.print(table)


def _add_target_arguments(parser, *, with_agents: bool = True) -> None:
    if with_agents:
        parser.add_argument(
            "--agent",
            "-a",
            dest="agents",
            action="append",
            choices=list(AGENTS),
            help=(
                "Agent to target; repeat for several. Defaults to the agents "
                "found in your home directory, plus the shared .agents/skills."
            ),
        )
    parser.add_argument(
        "--global",
        dest="is_global",
        action="store_true",
        help="Use the user-level skills directory instead of the current project.",
    )


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
    _add_target_arguments(install_parser)
    install_parser.set_defaults(func=_install)

    update_help = "Update registry skills wherever they are already installed."
    update_parser = subparsers.add_parser(
        "update",
        description=update_help,
        help=update_help,
        parents=command_parents,
    )
    _add_target_arguments(update_parser, with_agents=False)
    update_parser.set_defaults(func=_update)

    list_help = "List registry skills and where they are installed."
    list_parser = subparsers.add_parser(
        "list",
        aliases=["ls"],
        description=list_help,
        help=list_help,
        parents=command_parents,
    )
    _add_target_arguments(list_parser)
    list_parser.set_defaults(func=_list)

    remove_help = "Remove installed skills."
    remove_parser = subparsers.add_parser(
        "remove",
        aliases=["rm"],
        description=remove_help,
        help=remove_help,
        parents=command_parents,
    )
    remove_parser.add_argument("names", metavar="SKILL", nargs="+")
    _add_target_arguments(remove_parser)
    remove_parser.set_defaults(func=_remove_skills)
