import contextlib
import hashlib
import io
import json
import os
import stat
import sys
from pathlib import Path

import httpx
import pytest
from rich.console import Console

from fal.cli import skills as skills_cli
from fal.cli.main import parse_args
from fal.cli.parser import FalParserExit


class FakeRegistry:
    def __init__(self):
        self.skills = {}
        self.descriptions = {}
        self.requests = []

    def add(self, name, files):
        self.skills[name] = {k: v.encode() for k, v in files.items()}

    def index(self):
        return {
            "version": 1,
            "skills": [
                {
                    "name": name,
                    "description": self.descriptions.get(name, f"{name} description"),
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

    _run(["skills", "install", "other-skill", "-a", "claude-code", "--global"])

    assert (home / ".claude" / "skills" / "other-skill" / "SKILL.md").is_file()
    assert not (home / ".claude" / "skills" / "fal-serverless").exists()
    assert not (project / ".claude").exists()


def test_global_universal_dir_is_shared_home_dir(tmp_path):
    targets = skills_cli.target_dirs(
        ["universal", "codex", "windsurf"], is_global=True, root=tmp_path, home=tmp_path
    )
    assert targets == {
        tmp_path / ".agents" / "skills": ["universal", "codex", "windsurf"]
    }


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


@pytest.mark.parametrize(
    "bad_path",
    [
        "../escape.md",
        "/etc/passwd",
        "a/../../b.md",
        "a//b.md",
        "./a.md",
        "..\\escape.md",
        "C:\\x.md",
        "C:/x.md",
        "C:x.md",
        "\\\\server\\share\\x.md",
        "\\evil.md",
        "CON",
        "nul.txt",
        "trailing.",
    ],
)
def test_install_rejects_paths_outside_the_skill(registry, env, bad_path):
    _, project = env
    registry.add("fal-serverless", {bad_path: "x"})

    with pytest.raises(RuntimeError, match="unsafe skill path"):
        _run(["skills", "install", "-a", "universal"])
    assert list(project.iterdir()) == []


BAD_NAMES = ["..", ".", "", "/abs", "a/b", "../x", "a\\b", "C:x", "con", "Upper", "x."]


@pytest.mark.parametrize("bad_name", BAD_NAMES)
def test_install_rejects_unsafe_skill_names(registry, env, bad_name):
    _, project = env
    registry.skills = {bad_name: {"SKILL.md": b"pwned\n"}}
    (project / "keep.txt").write_text("mine\n")

    with pytest.raises(RuntimeError, match="unsafe skill name"):
        _run(["skills", "install", "--all", "-a", "universal"])
    assert [p.name for p in project.iterdir()] == ["keep.txt"]


@pytest.mark.parametrize("bad_name", [*BAD_NAMES, "fal-serverless/references"])
def test_remove_rejects_unsafe_skill_names(registry, env, bad_name):
    _, project = env
    _run(["skills", "install", "-a", "universal"])
    installed = sorted(p for p in project.rglob("*"))

    with pytest.raises(ValueError, match="unsafe skill name"):
        _run(["skills", "remove", bad_name, "-a", "universal"])
    assert sorted(p for p in project.rglob("*")) == installed


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
    _run(["skills", "install", "--all", "-a", "universal", "-a", "claude-code"])

    changes = _run_json(["skills", "remove", "fal-serverless"])

    assert sorted(c["path"] for c in changes) == sorted(
        str(project / d / "fal-serverless")
        for d in [".agents/skills", ".claude/skills"]
    )
    assert (project / ".agents" / "skills" / "other-skill").is_dir()


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


def test_list_table_renders_on_cp1252(registry, env):
    registry.add("fal-serverless", {"SKILL.md": "x\n"})
    registry.descriptions = {"fal-serverless": "Arrows → here. More text."}
    console = Console(
        file=StrictEncodingStream(), record=True, width=200, force_terminal=False
    )
    _run(["skills", "list"], console=console)
    assert "Arrows \\u2192 here" in console.file.getvalue()


def test_update_leaves_a_users_own_same_named_skill_alone(registry, env):
    _, project = env
    own = project / ".claude" / "skills" / "fal-serverless"
    own.mkdir(parents=True)
    (own / "SKILL.md").write_text("mine\n")
    (own / "notes.md").write_text("notes\n")

    assert _run_json(["skills", "update"]) == []
    assert (own / "SKILL.md").read_text() == "mine\n"
    assert (own / "notes.md").read_text() == "notes\n"
    rows = _run_json(["skills", "list", "-a", "claude-code"])
    assert rows[0]["installed"] == {str(own.parent): "unmanaged"}


@pytest.mark.parametrize("command", [["install"], ["install", "--all"], ["remove"]])
def test_a_users_own_same_named_skill_is_never_replaced(registry, env, command):
    _, project = env
    own = project / ".claude" / "skills" / "fal-serverless"
    own.mkdir(parents=True)
    (own / "SKILL.md").write_text("mine\n")
    (own / "notes.md").write_text("notes\n")
    argv = ["skills", *command, "-a", "claude-code"]
    if command == ["remove"]:
        argv.insert(2, "fal-serverless")

    changes = _run_json(argv)

    assert {c["skill"]: c["action"] for c in changes}["fal-serverless"] == (
        "skipped (not installed by fal)"
    )
    assert sorted(p.name for p in own.iterdir()) == ["SKILL.md", "notes.md"]
    assert (own / "SKILL.md").read_text() == "mine\n"


def test_update_downloads_each_skill_once(registry, env):
    _run(["skills", "install", "-a", "universal", "-a", "claude-code"])
    registry.add("fal-serverless", {"SKILL.md": "v2\n"})
    registry.requests.clear()

    changes = _run_json(["skills", "update"])

    assert [c["action"] for c in changes] == ["updated", "updated"]
    assert registry.requests.count("fal-serverless/SKILL.md") == 1


def test_dropped_files_make_a_skill_outdated(registry, env):
    _, project = env
    _run(["skills", "install", "-a", "universal"])
    installed = project / ".agents" / "skills" / "fal-serverless"
    skill_md = (installed / "SKILL.md").read_text()
    registry.add("fal-serverless", {"SKILL.md": skill_md})

    changes = _run_json(["skills", "update"])

    assert [c["action"] for c in changes] == ["updated"]
    assert not (installed / "references").exists()
    assert (installed / "SKILL.md").read_text() == skill_md


def test_crlf_checkout_counts_as_current(registry, env):
    _, project = env
    _run(["skills", "install", "-a", "universal"])
    skill_md = project / ".agents" / "skills" / "fal-serverless" / "SKILL.md"
    skill_md.write_bytes(skill_md.read_bytes().replace(b"\n", b"\r\n"))

    changes = _run_json(["skills", "install", "-a", "universal"])

    assert [c["action"] for c in changes] == ["unchanged"]


def test_failed_write_keeps_the_previous_install(registry, env, monkeypatch):
    _, project = env
    _run(["skills", "install", "-a", "universal"])
    skills_dir = project / ".agents" / "skills"
    before = {
        p.relative_to(skills_dir): p.read_bytes()
        for p in skills_dir.rglob("*")
        if p.is_file()
    }
    registry.add("fal-serverless", {"SKILL.md": "v2\n", "references/more.md": "v2\n"})
    original = Path.write_bytes
    calls = []

    def flaky(self, data):
        calls.append(self)
        if len(calls) == 2:
            raise OSError("disk full")
        return original(self, data)

    monkeypatch.setattr(Path, "write_bytes", flaky)

    with pytest.raises(OSError, match="disk full"):
        _run(["skills", "install", "-a", "universal"])
    after = {
        p.relative_to(skills_dir): p.read_bytes()
        for p in skills_dir.rglob("*")
        if p.is_file()
    }
    assert after == before


def test_checksum_mismatch_writes_nothing(registry, env, monkeypatch):
    _, project = env
    index = registry.index()
    index["skills"][1]["files"][0]["sha256"] = "0" * 64
    original = registry.get

    def tampered(_, path):
        if path == "index.json":
            return json.dumps(index).encode()
        return original(path)

    monkeypatch.setattr(skills_cli.Registry, "_get", tampered)

    with pytest.raises(RuntimeError, match="try again in a few minutes"):
        _run(["skills", "install", "--all", "-a", "universal"])
    assert not (project / ".agents").exists()


def test_registry_markup_is_printed_verbatim(registry, env, tmp_path, monkeypatch):
    description = "Use [optional] args, see [/x] and :fire: [docs](http://x)."
    registry.descriptions = {"fal-serverless": description}
    project = tmp_path / "app[wip]"
    project.mkdir()
    monkeypatch.chdir(project)

    rows = _run_json(["skills", "list", "-a", "universal"])
    assert rows[0]["description"] == description
    changes = _run_json(["skills", "install", "-a", "universal"])
    assert changes[0]["path"] == str(project / ".agents" / "skills" / "fal-serverless")

    table = _run(["skills", "list", "-a", "universal"])
    assert "Use [optional] args, see [/x] and :fire: [docs](http://x)" in table
    output = _run(["skills", "install", "other-skill", "-a", "universal"])
    assert str(project) in output


@pytest.mark.parametrize(
    "body, reason",
    [
        (b"<html>", "Expecting value"),
        (b'{"version": 2}', "missing 'skills'"),
        (b'{"skills": [{"name": "x"}]}', "missing 'files'"),
        (b'{"skills": ["x"]}', "string indices"),
    ],
)
def test_malformed_index_names_the_registry(env, monkeypatch, body, reason):
    monkeypatch.setattr(skills_cli.Registry, "_get", lambda _, path: body)
    with pytest.raises(RuntimeError, match="Malformed skills index at .*index.json"):
        _run(["skills", "list"])
    with pytest.raises(RuntimeError, match=reason):
        _run(["skills", "list"])


def test_repeated_agent_is_listed_once(registry, env):
    changes = _run_json(["skills", "install", "-a", "claude-code", "-a", "claude-code"])
    assert [c["agents"] for c in changes] == [["claude-code"]]


def _mock_registry(monkeypatch, handler):
    monkeypatch.setenv(skills_cli.REGISTRY_URL_ENV, "https://registry.test/skills")
    registry = skills_cli.Registry()
    registry._client = httpx.Client(transport=httpx.MockTransport(handler))
    return registry


def test_registry_reuses_one_client(monkeypatch):
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, content=b"ok")

    registry = _mock_registry(monkeypatch, handler)
    client = registry._client
    with registry:
        assert registry._get("a/SKILL.md") == b"ok"
        assert registry._get("b/SKILL.md") == b"ok"
        assert registry._client is client
    assert client.is_closed
    assert urls == [
        "https://registry.test/skills/a/SKILL.md",
        "https://registry.test/skills/b/SKILL.md",
    ]


def test_registry_reports_http_errors_with_the_url(monkeypatch):
    registry = _mock_registry(monkeypatch, lambda request: httpx.Response(404))
    with pytest.raises(RuntimeError, match="GET .*/skills/index.json returned 404"):
        registry._get("index.json")


def test_registry_reports_unreachable_host_with_the_url(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("Connection refused", request=request)

    registry = _mock_registry(monkeypatch, handler)
    with pytest.raises(RuntimeError) as exc_info:
        registry._get("index.json")
    assert str(exc_info.value) == (
        "Could not reach the fal skills registry at "
        "https://registry.test/skills/index.json: Connection refused"
    )
    assert exc_info.value.__cause__ is None


def test_help_states_each_commands_agent_default():
    for command, default in [
        ("install", skills_cli.DETECTED_AGENTS_HELP),
        ("remove", skills_cli.ALL_AGENTS_HELP),
        ("update", skills_cli.ALL_AGENTS_HELP),
    ]:
        stdout = io.StringIO()
        with pytest.raises(FalParserExit), contextlib.redirect_stdout(stdout):
            parse_args(["skills", command, "--help"])
        assert " ".join(default.split()) in " ".join(stdout.getvalue().split())


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junctions")
def test_remove_unlinks_a_junction_without_touching_its_target(tmp_path):
    import _winapi

    target = tmp_path / "target"
    target.mkdir()
    (target / "SKILL.md").write_text("keep\n")
    link = tmp_path / "skills" / "linked"
    link.parent.mkdir()
    _winapi.CreateJunction(str(target), str(link))

    skills_cli._remove(link)

    assert not os.path.lexists(link)
    assert (target / "SKILL.md").read_text() == "keep\n"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows read-only files")
def test_remove_deletes_read_only_files(tmp_path):
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("x\n")
    os.chmod(skill / "SKILL.md", stat.S_IREAD)

    skills_cli._remove(skill)

    assert not skill.exists()
