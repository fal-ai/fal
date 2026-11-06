import json
import time
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

    probe = parse_probe(output)
    assert probe.keys() >= {"gpus", "activity", "work", "hot"}
    # Optional readings: present on this box's probe, never required by the dash.
    assert "disks" in probe and "processes" in probe["system"]


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
    reading = signals.system(root=str(tmp_path), proc=str(tmp_path / "no-proc"))
    # 0.5 s of CPU over 0.5 s, against a 4-CPU quota.
    assert reading == {
        "cpu_pct": 25.0,
        "cpus": 4,
        "mem_used_bytes": 3 * 2**30,
        "mem_total_bytes": 8 * 2**30,
        "net_in_bps": None,
        "net_out_bps": None,
        "processes": [],
    }


def test_system_without_a_cgroup_falls_back_to_the_host(monkeypatch, tmp_path):
    signals = _signals(monkeypatch)
    monkeypatch.setattr(signals, "_meminfo", lambda: (2 * 2**30, 16 * 2**30))
    monkeypatch.setattr(
        signals.os, "sched_getaffinity", lambda pid: {0, 1}, raising=False
    )
    reading = signals.system(
        sample_seconds=0, root=str(tmp_path / "missing"), proc=str(tmp_path / "none")
    )
    assert reading == {
        "cpu_pct": None,
        "cpus": 2,
        "mem_used_bytes": 2 * 2**30,
        "mem_total_bytes": 16 * 2**30,
        "net_in_bps": None,
        "net_out_bps": None,
        "processes": [],
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
    reading = signals.system(sample_seconds=0, root=str(tmp_path))
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


def _fake_stat(root, pid, comm, ticks, rss_pages):
    """A /proc/<pid>/stat line: 52 fields, with the ones the probe reads set."""
    fields = ["S"] + ["0"] * 49
    fields[11], fields[12], fields[21] = str(ticks), "0", str(rss_pages)
    (root / str(pid)).mkdir(exist_ok=True)
    (root / str(pid) / "stat").write_text(f"{pid} ({comm}) {' '.join(fields)}\n")


def test_disks_count_a_filesystem_once_and_skip_what_is_missing(monkeypatch):
    signals = _signals(monkeypatch)
    devices = {"/work": 1, "/data": 2, "/": 1}

    def stat(path):
        if path not in devices:
            raise FileNotFoundError(path)
        return SimpleNamespace(st_dev=devices[path])

    monkeypatch.setattr(signals.os, "stat", stat)
    monkeypatch.setattr(
        signals.os,
        "statvfs",
        lambda path: SimpleNamespace(
            f_blocks=1000, f_bfree=250, f_bavail=200, f_frsize=4096
        ),
    )
    found = signals.disks(["/work", "/data", "/", "/nope"])
    assert found == [
        {
            "paths": ["/work", "/"],
            "used_bytes": 750 * 4096,
            "total_bytes": 1000 * 4096,
            "free_bytes": 200 * 4096,
        },
        {
            "paths": ["/data"],
            "used_bytes": 750 * 4096,
            "total_bytes": 1000 * 4096,
            "free_bytes": 200 * 4096,
        },
    ]


def test_disks_leave_out_a_filesystem_with_no_size(monkeypatch):
    signals = _signals(monkeypatch)
    monkeypatch.setattr(signals.os, "stat", lambda p: SimpleNamespace(st_dev=1))
    monkeypatch.setattr(
        signals.os,
        "statvfs",
        lambda p: SimpleNamespace(f_blocks=0, f_bfree=0, f_frsize=4096),
    )
    assert signals.disks(["/"]) == []


def test_net_bytes_sum_every_interface_but_loopback(monkeypatch, tmp_path):
    signals = _signals(monkeypatch)
    (tmp_path / "net").mkdir()
    (tmp_path / "net" / "dev").write_text(
        "Inter-|   Receive |  Transmit\n"
        " face |bytes packets errs drop fifo frame compressed multicast|bytes ...\n"
        "    lo: 999 1 0 0 0 0 0 0 999 1 0 0 0 0 0 0\n"
        "  eth0: 1000 1 0 0 0 0 0 0 200 1 0 0 0 0 0 0\n"
        "  eth1: 500 1 0 0 0 0 0 0 50 1 0 0 0 0 0 0\n"
    )
    assert signals._net_bytes(str(tmp_path)) == (1500, 250)
    assert signals._net_bytes(str(tmp_path / "missing")) is None


def test_cpu_ticks_read_a_name_with_spaces_and_parens(monkeypatch, tmp_path):
    signals = _signals(monkeypatch)
    _fake_stat(tmp_path, 7, "my (odd) proc", ticks=120, rss_pages=10)
    (tmp_path / "self").mkdir()  # not a pid
    found = signals._cpu_ticks(str(tmp_path))
    page = signals.os.sysconf("SC_PAGE_SIZE")
    assert found == {7: ("my (odd) proc", 120, 10 * page)}


def test_a_python_process_is_named_by_its_script(monkeypatch, tmp_path):
    signals = _signals(monkeypatch)
    (tmp_path / "9").mkdir()
    (tmp_path / "9" / "cmdline").write_bytes(
        b"/usr/bin/python3\0-u\0/work/app/serve.py\0--port\0" + b"1\0"
    )
    assert signals._command_name(9, "python3", str(tmp_path)) == "serve.py"
    assert signals._command_name(9, "nginx", str(tmp_path)) == "nginx"
    # No readable command line: the kernel's own name.
    assert signals._command_name(10, "python3", str(tmp_path)) == "python3"


def test_system_shares_one_window_for_cpu_network_and_processes(monkeypatch, tmp_path):
    signals = _signals(monkeypatch)
    (tmp_path / "cpu.max").write_text("200000 100000\n")
    monkeypatch.setattr(signals, "_cpu_usage_usec", lambda root: None)
    monkeypatch.setattr(signals, "_meminfo", lambda: (1, 2))
    monkeypatch.setattr(signals.os, "sysconf", lambda name: 100)
    nets = iter([(1000, 100), (2000, 600)])
    monkeypatch.setattr(signals, "_net_bytes", lambda proc: next(nets))
    me = signals.os.getpid()
    samples = iter(
        [
            {
                1: ("busy", 0, 4096),
                2: ("idle", 0, 9000),
                3: ("big", 0, 50),
                me: ("p", 0, 1),
            },
            {
                1: ("busy", 25, 4096),
                2: ("idle", 0, 9000),
                3: ("big", 0, 50),
                me: ("p", 500, 1),
                4: ("new", 99, 1),
            },
        ]
    )
    monkeypatch.setattr(signals, "_cpu_ticks", lambda proc: next(samples))
    monkeypatch.setattr(signals, "_command_name", lambda pid, comm, proc: comm)
    clock = iter([10.0, 10.5])
    sleeps = []
    monkeypatch.setattr(
        signals,
        "time",
        SimpleNamespace(monotonic=lambda: next(clock), sleep=sleeps.append),
    )
    reading = signals.system(root=str(tmp_path))
    assert sleeps == [0.25]  # one window for everything
    assert reading["net_in_bps"] == 2000 and reading["net_out_bps"] == 1000
    # 25 ticks at 100/s over 0.5 s is half a core; the probe itself and a process
    # that started inside the window are left out; ties go to the larger RSS.
    assert [p["name"] for p in reading["processes"]] == ["busy", "idle", "big"]
    assert reading["processes"][0] == {
        "pid": 1,
        "name": "busy",
        "cpu_pct": 50.0,
        "rss_bytes": 4096,
    }


def test_only_the_busiest_processes_are_kept(monkeypatch):
    signals = _signals(monkeypatch)
    monkeypatch.setattr(signals.os, "sysconf", lambda name: 100)
    monkeypatch.setattr(signals, "_command_name", lambda pid, comm, proc: comm)
    before = {pid: (f"p{pid}", 0, 1) for pid in range(1, 10)}
    after = {pid: (f"p{pid}", pid, 1) for pid in range(1, 10)}
    top = signals._busiest(before, after, 1.0)
    assert [p["pid"] for p in top] == [9, 8, 7, 6, 5]


def test_gpu_memory_is_summed_per_process(monkeypatch):
    signals = _signals(monkeypatch)

    def run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="4242, 1024\n4242, 512\n77, 8\n")

    monkeypatch.setattr(signals.subprocess, "run", run)
    assert signals.gpu_memory_by_pid() == {4242: 1536, 77: 8}

    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(signals.subprocess, "run", missing)
    assert signals.gpu_memory_by_pid() == {}


def test_probe_adds_gpu_memory_to_the_processes_it_lists(monkeypatch):
    import importlib

    monkeypatch.syspath_prepend(str(REMOTE))
    monkeypatch.syspath_prepend(str(Path(config.__file__).parent))
    probe = importlib.import_module("probe")
    monkeypatch.setattr(
        probe.signals,
        "system",
        lambda: {"processes": [{"pid": 5, "name": "a"}, {"pid": 6, "name": "b"}]},
    )
    monkeypatch.setattr(probe.signals, "gpu_memory_by_pid", lambda: {6: 2048})
    reading = probe._system()
    assert reading["processes"] == [
        {"pid": 5, "name": "a"},
        {"pid": 6, "name": "b", "gpu_mib": 2048},
    ]


def test_a_failing_disk_reading_does_not_fail_the_probe(monkeypatch):
    import importlib

    monkeypatch.syspath_prepend(str(REMOTE))
    monkeypatch.syspath_prepend(str(Path(config.__file__).parent))
    probe = importlib.import_module("probe")

    def boom(paths):
        raise RuntimeError("statvfs")

    monkeypatch.setattr(probe.signals, "disks", boom)
    assert probe._disks() is None


def test_a_reload_records_when_it_happened(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "hot_server_under_test", REMOTE / "hot_server.py"
    )
    hot_server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hot_server)
    (tmp_path / "infer.py").write_text("def infer(state): return 1\n")
    project = hot_server.Project(tmp_path)
    project._reload_code(["infer.py"])
    assert abs(project.last_reload["at"] - time.time()) < 5


def test_system_reads_a_cgroup_v1_container(monkeypatch, tmp_path):
    signals = _signals(monkeypatch)
    cpu = tmp_path / "cpu,cpuacct"
    memory = tmp_path / "memory"
    cpu.mkdir()
    memory.mkdir()
    (cpu / "cpu.cfs_quota_us").write_text("-1\n")
    (cpu / "cpu.cfs_period_us").write_text("100000\n")
    (memory / "memory.usage_in_bytes").write_text(str(5 * 2**30) + "\n")
    (memory / "memory.stat").write_text("cache 9\ninactive_file %d\n" % 2**30)
    (memory / "memory.limit_in_bytes").write_text(str(100 * 2**30) + "\n")
    usage = iter([2_000_000_000, 2_250_000_000])  # nanoseconds
    monkeypatch.setattr(signals, "_cpu_usage_usec", lambda root: next(usage) // 1000)
    clock = iter([10.0, 10.5])
    fake_time = SimpleNamespace(monotonic=lambda: next(clock), sleep=lambda s: None)
    monkeypatch.setattr(signals, "time", fake_time)
    monkeypatch.setattr(signals, "_meminfo", lambda: (1, 1500 * 2**30))
    monkeypatch.setattr(
        signals.os, "sched_getaffinity", lambda pid: set(range(6)), raising=False
    )
    reading = signals.system(root=str(tmp_path), proc=str(tmp_path / "noproc"))
    # No quota (-1): the 6 visible CPUs are the limit.
    assert reading["cpus"] == 6
    assert reading["mem_used_bytes"] == 4 * 2**30
    assert reading["mem_total_bytes"] == 100 * 2**30


def test_cgroup_v1_cpu_usage_and_unlimited_memory(monkeypatch, tmp_path):
    signals = _signals(monkeypatch)
    (tmp_path / "cpu,cpuacct").mkdir()
    (tmp_path / "memory").mkdir()
    (tmp_path / "cpu,cpuacct" / "cpuacct.usage").write_text("5579052407\n")
    (tmp_path / "memory" / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    assert signals._cpu_usage_usec(str(tmp_path)) == 5579052
    assert signals._memory_limit(str(tmp_path)) is None


def test_no_second_nvidia_smi_call_when_the_gpus_could_not_be_read(monkeypatch):
    import importlib

    monkeypatch.syspath_prepend(str(REMOTE))
    monkeypatch.syspath_prepend(str(Path(config.__file__).parent))
    probe = importlib.import_module("probe")
    monkeypatch.setattr(
        probe.signals, "system", lambda: {"processes": [{"pid": 6, "name": "b"}]}
    )

    def stuck():
        raise AssertionError("nvidia-smi must not run again")

    monkeypatch.setattr(probe.signals, "gpu_memory_by_pid", stuck)
    reading = probe._system(gpus_answered=False)
    assert reading["processes"] == [{"pid": 6, "name": "b"}]
