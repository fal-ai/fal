import hashlib
import io
import json
from pathlib import Path

import pytest
from rich.console import Console

from fal.cli import skills as skills_cli
from fal.cli.main import parse_args
from fal.cli.parser import FalParserExit


class FakeRegistry:
    def __init__(self):
        self.skills = {}
        self.requests = []

    def add(self, name, files):
        self.skills[name] = {k: v.encode() for k, v in files.items()}

    def index(self):
        return {
            "version": 1,
            "skills": [
                {
                    "name": name,
                    "description": f"{name} description",
                    "files": [
                        {
                            "path": path,
                            "sha256": hashlib.sha256(content).hexdigest(),
                            "bytes": len(content),
                        }
                        for path, content in files.items()
                    ],
                }
                for name, files in self.skills.items()
            ],
        }

    def get(self, path):
        self.requests.append(path)
        if path == "index.json":
            return json.dumps(self.index()).encode()
        name, rel = path.split("/", 1)
        return self.skills[name][rel]


@pytest.fixture
def registry(monkeypatch):
    fake = FakeRegistry()
    for name in ["fal-serverless", "other-skill"]:
        fake.add(
            name,
            {
                "SKILL.md": f"---\nname: {name}\n---\n",
                "references/more.md": "details\n",
            },
        )
    monkeypatch.setattr(skills_cli.Registry, "_get", lambda _, path: fake.get(path))
    monkeypatch.setattr(skills_cli, "DEFAULT_SKILLS", ("fal-serverless",))
    return fake


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.chdir(project)
    return home, project


def _run(argv, console=None):
    console = console or Console(record=True, width=200, force_terminal=False)
    args = parse_args(argv)
    args.console = console
    args.func(args)
    return console.export_text()


def _run_json(argv):
    return json.loads(_run([*argv, "--json"]))["skills"]


def test_parse_install():
    args = parse_args(
        ["skills", "install", "-a", "claude-code", "-a", "codex", "--global"]
    )
    assert args.func == skills_cli._install
    assert args.agents == ["claude-code", "codex"]
    assert args.is_global is True
    assert args.names == []


def test_parse_rejects_unknown_agent():
    with pytest.raises(FalParserExit):
        parse_args(["skills", "install", "--agent", "nope"])


def test_registry_url_env_override(monkeypatch):
    monkeypatch.setenv(skills_cli.REGISTRY_URL_ENV, "https://example.com/skills/")
    assert skills_cli.Registry().url == "https://example.com/skills"


def test_detect_agents_always_includes_universal(tmp_path):
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".cursor").mkdir()
    assert skills_cli.detect_agents(tmp_path) == [
        "universal",
        "claude-code",
        "cursor",
    ]


def test_agents_sharing_a_directory_are_installed_once(tmp_path):
    targets = skills_cli.target_dirs(
        ["universal", "codex", "cursor", "claude-code"],
        is_global=False,
        root=tmp_path,
        home=tmp_path / "home",
    )
    assert targets == {
        tmp_path / ".agents" / "skills": ["universal", "codex", "cursor"],
        tmp_path / ".claude" / "skills": ["claude-code"],
    }


def test_install_defaults_into_detected_agents(registry, env):
    home, project = env
    (home / ".claude").mkdir()

    changes = _run_json(["skills", "install"])

    assert {(c["path"], c["action"]) for c in changes} == {
        (str(project / d / "fal-serverless"), "installed")
        for d in [".agents/skills", ".claude/skills"]
    }
    for skills_dir in [".agents/skills", ".claude/skills"]:
        installed = project / skills_dir / "fal-serverless" / "references" / "more.md"
        assert installed.read_text() == "details\n"
    assert not (project / ".agents" / "skills" / "other-skill").exists()
    # Downloaded once, written to both directories.
    assert registry.requests.count("fal-serverless/SKILL.md") == 1


def test_install_named_skill_globally(registry, env):
    home, project = env

    _run(["skills", "install", "other-skill", "-a", "codex", "--global"])

    assert (home / ".codex" / "skills" / "other-skill" / "SKILL.md").is_file()
    assert not (home / ".codex" / "skills" / "fal-serverless").exists()
    assert not (project / ".agents").exists()


def test_install_all(registry, env):
    changes = _run_json(["skills", "install", "--all", "-a", "universal"])
    assert sorted(c["skill"] for c in changes) == ["fal-serverless", "other-skill"]


def test_install_unknown_skill(registry, env):
    with pytest.raises(ValueError, match="Unknown skill"):
        _run(["skills", "install", "nope"])


def test_install_rejects_checksum_mismatch(registry, env, monkeypatch):
    _, project = env
    index = registry.index()
    index["skills"][0]["files"][0]["sha256"] = "0" * 64
    original = registry.get

    def tampered(_, path):
        if path == "index.json":
            return json.dumps(index).encode()
        return original(path)

    monkeypatch.setattr(skills_cli.Registry, "_get", tampered)

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        _run(["skills", "install", "-a", "universal"])
    assert not (project / ".agents" / "skills" / "fal-serverless").exists()


@pytest.mark.parametrize("bad_path", ["../escape.md", "/etc/passwd"])
def test_install_rejects_paths_outside_the_skill(registry, env, bad_path):
    registry.add("fal-serverless", {bad_path: "x"})

    with pytest.raises(ValueError, match="unsafe skill path"):
        _run(["skills", "install", "-a", "universal"])


def test_reinstall_replaces_stale_files(registry, env):
    _, project = env
    _run(["skills", "install", "--all", "-a", "universal"])
    installed = project / ".agents" / "skills" / "fal-serverless"
    (installed / "SKILL.md").write_text("edited\n")
    (installed / "references" / "stale.md").write_text("old\n")

    changes = _run_json(["skills", "install", "--all", "-a", "universal"])

    actions = {c["skill"]: c["action"] for c in changes}
    assert actions == {"fal-serverless": "updated", "other-skill": "unchanged"}
    assert (installed / "SKILL.md").read_text().startswith("---\n")
    assert not (installed / "references" / "stale.md").exists()


def test_install_leaves_other_skills_alone(registry, env):
    _, project = env
    other = project / ".agents" / "skills" / "someone-else"
    other.mkdir(parents=True)
    (other / "SKILL.md").write_text("mine\n")

    _run(["skills", "install", "-a", "universal"])

    assert (other / "SKILL.md").read_text() == "mine\n"


def test_update_only_touches_existing_installs(registry, env):
    _, project = env
    _run(["skills", "install", "-a", "claude-code"])
    registry.add("fal-serverless", {"SKILL.md": "v2\n"})

    changes = _run_json(["skills", "update"])

    assert [(c["skill"], c["action"]) for c in changes] == [
        ("fal-serverless", "updated")
    ]
    installed = project / ".claude" / "skills" / "fal-serverless"
    assert (installed / "SKILL.md").read_text() == "v2\n"
    assert not (installed / "references").exists()
    assert not (project / ".agents").exists()


def test_list_reports_status(registry, env):
    _, project = env
    _run(["skills", "install", "-a", "universal"])
    registry.add("fal-serverless", {"SKILL.md": "v2\n"})

    rows = _run_json(["skills", "list", "-a", "universal"])

    assert {r["skill"]: r["installed"] for r in rows} == {
        "fal-serverless": {str(project / ".agents" / "skills"): "outdated"},
        "other-skill": {},
    }
    assert [r["skill"] for r in rows if r["default"]] == ["fal-serverless"]


def test_list_pretty_output(registry, env):
    output = _run(["skills", "list"])
    assert "fal-serverless (default)" in output
    assert "other-skill description" in output


def test_remove_checks_every_agent_by_default(registry, env):
    _, project = env
    _run(["skills", "install", "--all", "-a", "universal", "-a", "windsurf"])

    changes = _run_json(["skills", "remove", "fal-serverless"])

    assert sorted(c["path"] for c in changes) == sorted(
        str(project / d / "fal-serverless")
        for d in [".agents/skills", ".windsurf/skills"]
    )
    assert (project / ".agents" / "skills" / "other-skill").is_dir()


def test_remove_rejects_path_traversal(env):
    with pytest.raises(ValueError, match="unsafe skill path"):
        _run(["skills", "remove", ".."])


class StrictEncodingStream(io.StringIO):
    encoding = "cp1252"

    def write(self, text):
        text.encode(self.encoding)
        return super().write(text)


def test_install_output_renders_on_cp1252(registry, env):
    console = Console(
        file=StrictEncodingStream(), record=True, width=200, force_terminal=False
    )
    _run(["skills", "install", "-a", "universal"], console=console)
    assert "+ installed fal-serverless" in console.file.getvalue()
