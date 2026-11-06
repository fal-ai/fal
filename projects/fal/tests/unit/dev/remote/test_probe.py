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


def test_a_file_with_an_old_mtime_but_a_new_ctime_counts_as_unsaved(
    monkeypatch, tmp_path
):
    import importlib
    import os
    import time

    monkeypatch.syspath_prepend(str(REMOTE))
    probe = importlib.import_module("probe")
    work = tmp_path / "work"
    work.mkdir()
    stamp = tmp_path / "stamp"
    stamp.write_text("")
    monkeypatch.setattr(probe.config, "LOCAL_WORK", str(work))
    monkeypatch.setattr(probe.config, "SAVE_STAMP", str(stamp))
    time.sleep(0.05)
    (work / "big.bin").write_text("copied with its original mtime")
    os.utime(work / "big.bin", (1, 1))  # what rsync -a and cp -p leave behind

    assert probe._work(str(tmp_path), True)["dirty"] is True
