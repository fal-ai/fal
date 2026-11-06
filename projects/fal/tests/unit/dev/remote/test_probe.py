import json
from pathlib import Path
from types import SimpleNamespace

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


def _signals(monkeypatch):
    import importlib

    monkeypatch.syspath_prepend(str(REMOTE))
    monkeypatch.syspath_prepend(str(Path(config.__file__).parent))
    return importlib.import_module("signals")


def test_system_reads_the_container_cgroup(monkeypatch, tmp_path):
    signals = _signals(monkeypatch)
    (tmp_path / "cpu.max").write_text("400000 100000\n")
    (tmp_path / "memory.current").write_text(str(3 * 2**30) + "\n")
    (tmp_path / "memory.max").write_text(str(8 * 2**30) + "\n")
    usage = iter([1_000_000, 1_500_000])
    monkeypatch.setattr(signals, "_cpu_usage_usec", lambda root: next(usage))
    clock = iter([10.0, 10.5])
    fake_time = SimpleNamespace(monotonic=lambda: next(clock), sleep=lambda s: None)
    monkeypatch.setattr(signals, "time", fake_time)
    reading = signals.system(root=str(tmp_path))
    # 0.5 s of CPU over 0.5 s, against a 4-CPU quota.
    assert reading == {
        "cpu_pct": 25.0,
        "cpus": 4,
        "mem_used_bytes": 3 * 2**30,
        "mem_total_bytes": 8 * 2**30,
    }


def test_system_without_a_cgroup_falls_back_to_the_host(monkeypatch, tmp_path):
    signals = _signals(monkeypatch)
    monkeypatch.setattr(signals, "_meminfo", lambda: (2 * 2**30, 16 * 2**30))
    monkeypatch.setattr(
        signals.os, "sched_getaffinity", lambda pid: {0, 1}, raising=False
    )
    reading = signals.system(root=str(tmp_path / "missing"))
    assert reading == {
        "cpu_pct": None,
        "cpus": 2,
        "mem_used_bytes": 2 * 2**30,
        "mem_total_bytes": 16 * 2**30,
    }


def test_system_leaves_out_reclaimable_cache_and_an_unlimited_quota(
    monkeypatch, tmp_path
):
    signals = _signals(monkeypatch)
    (tmp_path / "cpu.max").write_text("max 100000\n")
    (tmp_path / "memory.current").write_text(str(10 * 2**30) + "\n")
    (tmp_path / "memory.stat").write_text(
        "anon 1\ninactive_file %d\nactive_file 5\n" % (6 * 2**30)
    )
    (tmp_path / "memory.max").write_text("max\n")
    monkeypatch.setattr(signals, "_cpu_usage_usec", lambda root: None)
    monkeypatch.setattr(signals, "_meminfo", lambda: (1, 64 * 2**30))
    monkeypatch.setattr(
        signals.os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False
    )
    reading = signals.system(root=str(tmp_path))
    assert reading["cpus"] == 3
    assert reading["mem_used_bytes"] == 4 * 2**30
    assert reading["mem_total_bytes"] == 64 * 2**30


def test_a_failing_system_reading_does_not_fail_the_probe(monkeypatch):
    import importlib

    monkeypatch.syspath_prepend(str(REMOTE))
    monkeypatch.syspath_prepend(str(Path(config.__file__).parent))
    probe = importlib.import_module("probe")

    def boom(*args, **kwargs):
        raise RuntimeError("no cgroup")

    monkeypatch.setattr(probe.signals, "system", boom)
    assert probe._system() is None
