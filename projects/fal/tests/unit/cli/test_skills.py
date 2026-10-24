import contextlib
import hashlib
import io
import json
import os
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


def _tree(root):
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


def _install_universal(project):
    _run(["skills", "install", "-a", "universal"])
    return project / ".agents" / "skills" / "fal-serverless"


def test_parse_install():
    args = parse_args(
        ["skills", "install", "-a", "claude-code", "-a", "codex", "--global"]
    )
    assert args.func == skills_cli._install
    assert args.agents == ["claude-code", "codex"]
    assert args.is_global is True
    assert args.names == []
    assert args.force is False


def test_parse_rejects_unknown_agent():
    with pytest.raises(FalParserExit):
        parse_args(["skills", "install", "--agent", "nope"])


def test_help_states_each_commands_agent_default():
    for command, default in [
        ("install", skills_cli.DETECTED_AGENTS_HELP),
        ("update", skills_cli.ALL_AGENTS_HELP),
        ("list", skills_cli.ALL_AGENTS_HELP),
        ("remove", skills_cli.ALL_AGENTS_HELP),
    ]:
        stdout = io.StringIO()
        with pytest.raises(FalParserExit), contextlib.redirect_stdout(stdout):
            parse_args(["skills", command, "--help"])
        assert " ".join(default.split()) in " ".join(stdout.getvalue().split())


def test_detect_agents_always_includes_universal(tmp_path):
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".cursor").mkdir()
    assert skills_cli.detect_agents(tmp_path) == ["universal", "claude-code", "cursor"]


def test_install_defaults_into_detected_agents(registry, env):
    home, project = env
    (home / ".claude").mkdir()
    (home / ".cursor").mkdir()

    changes = _run_json(["skills", "install"])

    assert [(c["path"], c["agents"], c["action"]) for c in changes] == [
        (
            str(project / ".agents/skills/fal-serverless"),
            ["universal", "cursor"],
            "installed",
        ),
        (str(project / ".claude/skills/fal-serverless"), ["claude-code"], "installed"),
    ]
    installed = project / ".claude/skills/fal-serverless/references/more.md"
    assert installed.read_text() == "details\n"
    assert not (project / ".agents/skills/other-skill").exists()
    # Downloaded once, written to both directories.
    assert registry.requests.count("fal-serverless/SKILL.md") == 1


def test_install_globally_shares_the_home_agents_dir(registry, env):
    home, project = env

    changes = _run_json(
        ["skills", "install", "-a", "cursor", "-a", "codex", "--global"]
    )

    assert [(c["path"], c["agents"]) for c in changes] == [
        (str(home / ".agents/skills/fal-serverless"), ["cursor", "codex"])
    ]
    assert list(project.iterdir()) == []


def test_install_all_and_unknown(registry, env):
    changes = _run_json(["skills", "install", "--all", "-a", "universal"])
    assert sorted(c["skill"] for c in changes) == ["fal-serverless", "other-skill"]
    with pytest.raises(ValueError, match="Unknown skill"):
        _run(["skills", "install", "nope"])
    with pytest.raises(ValueError, match="not both"):
        _run(["skills", "install", "x", "--all"])


def test_reinstall_is_unchanged_and_crlf_counts_as_unchanged(registry, env):
    _, project = env
    installed = _install_universal(project)
    skill_md = installed / "SKILL.md"
    skill_md.write_bytes(skill_md.read_bytes().replace(b"\n", b"\r\n"))

    changes = _run_json(["skills", "install", "-a", "universal"])

    assert [c["action"] for c in changes] == ["unchanged"]


def test_update_replaces_the_whole_skill(registry, env):
    _, project = env
    _run(["skills", "install", "-a", "claude-code"])
    registry.add("fal-serverless", {"SKILL.md": "v2\n"})

    changes = _run_json(["skills", "update"])

    assert [(c["skill"], c["action"]) for c in changes] == [
        ("fal-serverless", "updated")
    ]
    installed = project / ".claude/skills/fal-serverless"
    assert set(_tree(installed)) == {"SKILL.md", ".fal-skill"}
    assert not (project / ".agents").exists()


def test_update_downloads_each_skill_once(registry, env):
    _run(["skills", "install", "-a", "universal", "-a", "claude-code"])
    registry.add("fal-serverless", {"SKILL.md": "v2\n"})
    registry.requests.clear()

    changes = _run_json(["skills", "update"])

    assert [c["action"] for c in changes] == ["updated", "updated"]
    assert registry.requests.count("fal-serverless/SKILL.md") == 1


def _edit(installed):
    (installed / "SKILL.md").write_text("my edits\n")


def _add_file(installed):
    (installed / "references" / "notes.md").write_text("mine\n")


def _delete_file(installed):
    (installed / "references" / "more.md").unlink()


def _add_link(installed):
    try:
        os.symlink(installed.parent, installed / "link")
    except OSError:
        pytest.skip("symlinks unavailable")


@pytest.mark.parametrize("change", [_edit, _add_file, _delete_file, _add_link])
@pytest.mark.parametrize(
    "command", [["install"], ["update"], ["remove", "fal-serverless"]]
)
def test_local_changes_are_kept_unless_forced(registry, env, change, command):
    _, project = env
    installed = _install_universal(project)
    registry.add("fal-serverless", {"SKILL.md": "v2\n"})
    change(installed)
    before = _tree(installed)
    argv = ["skills", *command, "-a", "universal"]

    changes = _run_json(argv)

    assert [(c["action"], c["reason"]) for c in changes] == [("skipped", "modified")]
    assert _tree(installed) == before

    forced = _run_json([*argv, "--force"])

    assert [c["action"] for c in forced] == (
        ["removed"] if command[0] == "remove" else ["updated"]
    )
    if command[0] == "remove":
        assert not os.path.lexists(installed)
    else:
        assert set(_tree(installed)) == {"SKILL.md", ".fal-skill"}


def test_modified_hint_names_force(registry, env):
    _, project = env
    _edit(_install_universal(project))
    registry.add("fal-serverless", {"SKILL.md": "v2\n"})

    output = _run(["skills", "update"])

    assert "! skipped fal-serverless in .agents/skills/fal-serverless" in output
    assert "pass --force" in output


def _own_skill(path):
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text("mine\n")


def _bad_marker(content):
    def make(path):
        _own_skill(path)
        (path / skills_cli.MARKER).write_text(content)

    return make


def _file(path):
    path.parent.mkdir(parents=True)
    path.write_text("mine\n")


def _link_to_fal_skill(path):
    target = Path.cwd() / "elsewhere"
    target.mkdir()
    os.replace(_install_universal(Path.cwd()), target / "x")
    path.parent.mkdir(parents=True)
    try:
        os.symlink(target / "x", path)
    except OSError:
        pytest.skip("symlinks unavailable")


@pytest.mark.parametrize(
    "make",
    [
        _own_skill,
        _bad_marker("{not json"),
        _bad_marker("[]"),
        _bad_marker('{"files": []}'),
        _file,
        _link_to_fal_skill,
    ],
)
@pytest.mark.parametrize(
    "command",
    [["install"], ["install", "--all"], ["update"], ["remove", "fal-serverless"]],
)
def test_skills_fal_does_not_own_are_never_touched(registry, env, make, command):
    _, project = env
    path = project / ".claude/skills/fal-serverless"
    make(path)

    def tree():
        return {k: v for k, v in _tree(project).items() if "other-skill" not in k}

    before = tree()

    changes = _run_json(["skills", *command, "-a", "claude-code", "--force"])

    change = [c for c in changes if c["skill"] == "fal-serverless"]
    if command != ["update"]:
        assert [(c["action"], c["reason"]) for c in change] == [
            ("skipped", "unmanaged")
        ]
    else:
        assert change == []
    assert tree() == before


def test_remove_checks_every_agent_by_default(registry, env):
    _, project = env
    _run(["skills", "install", "--all", "-a", "universal", "-a", "claude-code"])

    changes = _run_json(["skills", "remove", "fal-serverless"])

    assert sorted(c["path"] for c in changes) == sorted(
        str(project / d / "fal-serverless")
        for d in [".agents/skills", ".claude/skills"]
    )
    assert all(c["action"] == "removed" for c in changes)
    assert (project / ".agents/skills/other-skill").is_dir()


@pytest.mark.parametrize("command", [["update"], ["remove", "fal-serverless"]])
def test_empty_update_and_remove_say_where_they_looked(registry, env, command):
    assert "in this project." in _run(["skills", *command])
    assert "in your home directory." in _run(["skills", *command, "--global"])


def test_list_finds_installs_in_undetected_agents(registry, env):
    _, project = env
    _run(["skills", "install", "-a", "claude-code"])
    registry.add("fal-serverless", {"SKILL.md": "v2\n"})

    rows = _run_json(["skills", "list"])

    assert {r["skill"]: r["installed"] for r in rows} == {
        "fal-serverless": [
            {
                "path": str(project / ".claude/skills/fal-serverless"),
                "agents": ["claude-code"],
                "status": "outdated",
            }
        ],
        "other-skill": [],
    }
    assert [r["skill"] for r in rows if r["default"]] == ["fal-serverless"]


def test_list_table_shows_short_paths(registry, env):
    home, _ = env
    _run(["skills", "install", "-a", "universal"])
    _run(["skills", "install", "-a", "claude-code", "--global"])

    assert "current: .agents/skills/fal-serverless" in _run(["skills", "list"])
    table = _run(["skills", "list", "--global"])
    assert "current: ~/.claude/skills/fal-serverless" in table
    assert str(home) not in table
    assert "fal-serverless (default)" in table
    assert "other-skill description" in table


BAD_PATHS = [
    "../escape.md",
    "/etc/passwd",
    "a/../../b.md",
    "a//b.md",
    "./a.md",
    "..\\escape.md",
    "C:/x.md",
    "\\\\server\\share\\x.md",
    "CON",
    "nul.txt",
    "a/aux",
    "trailing.",
    ".fal-skill",
]


@pytest.mark.parametrize("bad_path", BAD_PATHS)
def test_install_rejects_unsafe_paths(registry, env, bad_path):
    _, project = env
    registry.add("fal-serverless", {bad_path: "x"})

    with pytest.raises(RuntimeError, match="unsafe skill file path"):
        _run(["skills", "install", "-a", "universal"])
    assert list(project.iterdir()) == []


def test_install_allows_dotfiles(registry, env):
    _, project = env
    registry.add("fal-serverless", {"SKILL.md": "x\n", "runtime/.gitignore": "*\n"})

    installed = _install_universal(project)

    assert (installed / "runtime" / ".gitignore").read_text() == "*\n"
    assert [c["action"] for c in _run_json(["skills", "update"])] == ["unchanged"]


BAD_NAMES = ["..", ".", "", "a/b", "a\\b", "C:x", "con", "Upper", "a--b", "a-", "x."]


@pytest.mark.parametrize("bad_name", BAD_NAMES)
def test_install_rejects_unsafe_skill_names(registry, env, bad_name):
    _, project = env
    registry.skills = {bad_name: {"SKILL.md": b"pwned\n"}}

    with pytest.raises(RuntimeError, match="Invalid skill name"):
        _run(["skills", "install", "--all", "-a", "universal"])
    assert list(project.iterdir()) == []


@pytest.mark.parametrize("bad_name", [*BAD_NAMES, "fal-serverless/references"])
def test_remove_rejects_unsafe_skill_names(registry, env, bad_name):
    _, project = env
    _install_universal(project)
    before = _tree(project)

    with pytest.raises(ValueError, match="Invalid skill name"):
        _run(["skills", "remove", bad_name, "-a", "universal"])
    assert _tree(project) == before


def _tamper_index(registry, monkeypatch):
    index = registry.index()
    index["skills"][1]["files"][0]["sha256"] = "0" * 64
    original = registry.get

    def tampered(_, path):
        return json.dumps(index).encode() if path == "index.json" else original(path)

    monkeypatch.setattr(skills_cli.Registry, "_get", tampered)


def test_checksum_mismatch_writes_nothing(registry, env, monkeypatch):
    _, project = env
    _tamper_index(registry, monkeypatch)

    with pytest.raises(RuntimeError, match="try again in a few minutes"):
        _run(["skills", "install", "--all", "-a", "universal"])
    assert not (project / ".agents").exists()


def test_failed_write_keeps_the_previous_install(registry, env, monkeypatch):
    _, project = env
    _install_universal(project)
    skills_dir = project / ".agents/skills"
    before = _tree(skills_dir)
    registry.add("fal-serverless", {"SKILL.md": "v2\n", "references/more.md": "v2\n"})
    original = Path.write_bytes
    calls = []

    def flaky(self, data):
        calls.append(self)
        if len(calls) == 2:
            raise OSError("disk full")
        return original(self, data)

    monkeypatch.setattr(Path, "write_bytes", flaky)

    changes = _run_json(["skills", "update"])

    assert [(c["action"], c["error"]) for c in changes] == [("failed", "disk full")]
    assert _tree(skills_dir) == before
    assert [p.name for p in skills_dir.iterdir()] == ["fal-serverless"]


def test_failed_swap_restores_the_previous_install(registry, env, monkeypatch):
    _, project = env
    installed = _install_universal(project)
    before = _tree(installed)
    registry.add("fal-serverless", {"SKILL.md": "v2\n"})
    original = os.replace

    def flaky(src, dst):
        if Path(src).name == "new":
            raise OSError("in use")
        return original(src, dst)

    monkeypatch.setattr(os, "replace", flaky)

    changes = _run_json(["skills", "update"])

    assert [(c["action"], c["error"]) for c in changes] == [("failed", "in use")]
    assert _tree(installed) == before
    assert [p.name for p in installed.parent.iterdir()] == ["fal-serverless"]


def test_a_failing_target_is_reported_and_the_rest_continue(registry, env, monkeypatch):
    _, project = env
    original = skills_cli.write_skill

    def flaky(skill, files, skills_dir):
        if skills_dir.parent.name == ".claude":
            raise PermissionError("denied")
        return original(skill, files, skills_dir)

    monkeypatch.setattr(skills_cli, "write_skill", flaky)

    changes = _run_json(["skills", "install", "-a", "universal", "-a", "claude-code"])

    assert [c["action"] for c in changes] == ["installed", "failed"]
    assert (project / ".agents/skills/fal-serverless/SKILL.md").is_file()


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


class StrictEncodingStream(io.StringIO):
    encoding = "cp1252"

    def write(self, text):
        text.encode(self.encoding)
        return super().write(text)


def test_output_renders_on_cp1252(registry, env):
    registry.descriptions = {"fal-serverless": "Arrows → here. More text."}

    def console():
        return Console(
            file=StrictEncodingStream(), record=True, width=200, force_terminal=False
        )

    install = console()
    _run(["skills", "install", "-a", "universal"], console=install)
    assert "+ installed fal-serverless" in install.file.getvalue()
    listing = console()
    _run(["skills", "list"], console=listing)
    assert "Arrows \\u2192 here" in listing.file.getvalue()


def test_registry_markup_is_printed_verbatim(registry, env, tmp_path, monkeypatch):
    description = "Use [optional] args, see [/x] and :fire: [docs](http://x)."
    registry.descriptions = {"fal-serverless": description}
    project = tmp_path / "app[wip]"
    project.mkdir()
    monkeypatch.chdir(project)

    assert _run_json(["skills", "list"])[0]["description"] == description
    assert description.rstrip(".") in _run(["skills", "list"])
    changes = _run_json(["skills", "install", "-a", "universal"])
    assert changes[0]["path"] == str(project / ".agents/skills/fal-serverless")


def _mock_registry(monkeypatch, handler):
    monkeypatch.setenv(skills_cli.REGISTRY_URL_ENV, "https://registry.test/skills/")
    registry = skills_cli.Registry()
    registry._client = httpx.Client(transport=httpx.MockTransport(handler))
    return registry


def test_registry_reuses_one_client_and_honours_the_env_url(monkeypatch):
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, content=b"ok")

    registry = _mock_registry(monkeypatch, handler)
    client = registry._client
    with registry:
        assert registry._get("a/SKILL.md") == b"ok"
        assert registry._get("b/SKILL.md") == b"ok"
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


def test_a_skill_that_appears_during_download_is_left_alone(registry, env, monkeypatch):
    _, project = env
    path = project / ".agents/skills/fal-serverless"
    original = skills_cli.Registry.download

    def racing(self, skill):
        _own_skill(path)
        return original(self, skill)

    monkeypatch.setattr(skills_cli.Registry, "download", racing)

    changes = _run_json(["skills", "install", "-a", "universal"])

    assert [c["action"] for c in changes] == ["failed"]
    assert _tree(path) == {"SKILL.md": b"mine\n"}
