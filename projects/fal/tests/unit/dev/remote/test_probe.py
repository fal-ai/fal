import json
from pathlib import Path

from fal.dev import config
from fal.dev.box import REMOTE


def test_probe_survives_a_whitespace_only_hot_error(monkeypatch):
    import importlib

    monkeypatch.syspath_prepend(str(REMOTE))
    monkeypatch.syspath_prepend(str(Path(config.__file__).parent))
    probe = importlib.import_module("probe")
    monkeypatch.setattr(probe.signals, "tmux_session", lambda name: True)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"error": "  \\n "}'

    monkeypatch.setattr(probe.urllib.request, "urlopen", lambda *a, **k: Response())

    assert probe._hot()["error"] == ""


def test_the_probe_as_installed_on_the_box_satisfies_the_dashboard(tmp_path):
    import shutil
    import subprocess
    import sys

    from fal.dev.dash.model import parse_probe

    for script in (REMOTE / "probe.py", REMOTE / "signals.py", Path(config.__file__)):
        shutil.copy(script, tmp_path / script.name)
    settings = json.dumps({"workspace": str(tmp_path), "check_dirty": False})
    output = subprocess.run(
        [sys.executable, str(tmp_path / "probe.py"), settings],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    ).stdout

    assert parse_probe(output).keys() >= {"gpus", "activity", "work", "hot"}
