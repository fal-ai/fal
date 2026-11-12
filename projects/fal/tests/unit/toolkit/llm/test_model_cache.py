from __future__ import annotations

import json
import os
import sys
import threading
import time
import types

import httpx
import pytest
from llm_helpers import FAKE, _make_probe

from fal.toolkit.llm import LLMApp, _app, _cache


def test_compile_marker_absent_then_present(tmp_path, monkeypatch):
    first = _make_probe(tmp_path, monkeypatch)
    first.setup()
    try:
        assert first._boot["compile_cache_warm"] is False
        assert "previous_boot" not in first._boot
    finally:
        first.teardown()
    markers = list((tmp_path / "engine" / "ns").glob("boot-*.json"))
    assert len(markers) == 1
    assert not list((tmp_path / "engine" / "ns").glob("*.tmp"))

    second = _make_probe(tmp_path, monkeypatch)
    second.setup()
    try:
        assert second._boot["compile_cache_warm"] is True
        assert set(second._boot["previous_boot"]) == {
            "total_setup_s",
            "engine_ready_s",
        }
    finally:
        second.teardown()

    # A different engine config is a different compile cache.
    other = _make_probe(tmp_path, monkeypatch, engine_args=["--tp", "8"])
    assert other._config_key() != second._config_key()


def test_compile_marker_keyed_on_resolved_snapshot(tmp_path, monkeypatch):
    seen: list = []

    def boot(snapshot):
        snapshot.mkdir(exist_ok=True)
        app = _make_probe(tmp_path, monkeypatch, model="org/m")

        def engine_command(path):
            seen.append(path)
            # Like a preset gated on the snapshot's config.json.
            gated = ["--gated"] if os.path.isdir(path) else []
            port = str(app.engine_port)
            return [sys.executable, FAKE, "--port", port, "--model", path, *gated]

        app.engine_command = engine_command  # type: ignore[method-assign]
        app._resolve_model = lambda: str(snapshot)  # type: ignore[method-assign]
        app.setup()
        try:
            return dict(app._boot)
        finally:
            app.teardown()

    first = boot(tmp_path / "snap-a")
    # Keyed after resolve, from the real snapshot path (not the Hub id).
    assert "org/m" not in seen and seen[0] == str(tmp_path / "snap-a")
    assert first["compile_cache_warm"] is False
    # A new snapshot dir of the same config is still the same compile cache.
    second = boot(tmp_path / "snap-b")
    assert second["compile_cache_warm"] is True
    assert len(list((tmp_path / "engine" / "ns").glob("boot-*.json"))) == 1


def _fake_snapshot_download(monkeypatch, tmp_path):
    # huggingface_hub is not a dev dependency, so stand in for the module.
    snap = tmp_path / "snap"
    snap.mkdir(exist_ok=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"x")

    def fake(repo_id, **kwargs):
        calls.append(bool(kwargs.get("local_files_only")))
        calls.revisions.append(kwargs.get("revision"))
        return str(snap)

    class Calls(list):
        def __init__(self):
            super().__init__()
            self.revisions = []

    calls = Calls()
    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = fake  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return calls


def test_resolve_local_first_gated_by_marker(tmp_path, monkeypatch):
    calls = _fake_snapshot_download(monkeypatch, tmp_path)
    app = _make_probe(tmp_path, monkeypatch, model="org/model")
    app._boot = {}

    # No marker: an interrupted download could be in the HF cache, so go to
    # the network and record the completed download.
    assert app._resolve_model() == str(tmp_path / "snap")
    assert calls == [False]
    assert app._boot["resolve_source"] == "network"

    # Marker present: the recorded snapshot is reused with no Hub call, so a
    # newer partial download on the shared volume is never picked up.
    calls.clear()
    assert app._resolve_model() == str(tmp_path / "snap")
    assert calls == []
    assert app._boot["resolve_source"] == "local"


def test_resolve_pruned_local_snapshot_downloads(tmp_path, monkeypatch):
    calls = _fake_snapshot_download(monkeypatch, tmp_path)
    app = _make_probe(tmp_path, monkeypatch, model="org/model")
    app._boot = {}
    app._resolve_model()
    (tmp_path / "snap" / "config.json").unlink()
    calls.clear()
    app._resolve_model()
    assert calls == [False]
    assert app._boot["resolve_source"] == "network"

    # A pruned weight blob leaves a dangling symlink.
    (tmp_path / "snap" / "config.json").write_text("{}")
    (tmp_path / "snap" / "w.safetensors").symlink_to(tmp_path / "gone")
    calls.clear()
    app._resolve_model()
    assert calls == [False]


def test_resolve_local_dir_skips_hub(tmp_path, monkeypatch):
    calls = _fake_snapshot_download(monkeypatch, tmp_path)
    app = _make_probe(tmp_path, monkeypatch)
    app._boot = {}
    assert app._resolve_model() == app.model
    assert calls == [] and app._boot["resolve_source"] == "path"


def test_revision_passed_and_in_marker_key(tmp_path, monkeypatch):
    calls = _fake_snapshot_download(monkeypatch, tmp_path)
    app = _make_probe(tmp_path, monkeypatch, model="org/model", model_revision="v1")
    app._boot = {}
    app._resolve_model()
    app._resolve_model()
    assert calls == [False]
    assert calls.revisions == ["v1"]
    assert _cache.snapshot_marker_path("org/model", "v1").exists()
    assert not _cache.snapshot_marker_path("org/model").exists()


def test_snapshot_marker_follows_hf_hub_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "custom" / "hub"))
    path = _cache.snapshot_marker_path("org/model")
    assert path.parent == tmp_path / "custom" / "llm_app_snapshots"


def test_resolve_without_persistent_caches_skips_marker(tmp_path, monkeypatch):
    calls = _fake_snapshot_download(monkeypatch, tmp_path)
    app = _make_probe(tmp_path, monkeypatch, model="org/model", persistent_caches=False)
    app._boot = {}
    app._resolve_model()
    app._resolve_model()
    assert calls == [False, False]
    assert not _cache.snapshot_marker_path("org/model").exists()


def test_config_key_engine_env_and_port_forms(tmp_path, monkeypatch, capsys):
    base = _make_probe(tmp_path, monkeypatch)
    with_env = _make_probe(tmp_path, monkeypatch, engine_env={"A": "1"})
    assert base._config_key() != with_env._config_key()

    def cmd_with(port_args):
        app = _make_probe(tmp_path, monkeypatch)
        app.engine_command = lambda p: ["serve", *port_args, "--tp", "2"]  # type: ignore[method-assign]
        return app._config_key()

    assert cmd_with(["--port", "1"]) == cmd_with(["--port=2"]) == cmd_with([])

    broken = _make_probe(tmp_path, monkeypatch)

    def boom(p):
        raise ValueError("bad")

    broken.engine_command = boom  # type: ignore[method-assign]
    broken._config_key()
    assert "falls back to engine_args" in capsys.readouterr().out


def test_config_key_uses_resolved_model_config(tmp_path, monkeypatch):
    app = _make_probe(tmp_path, monkeypatch)
    seen = []

    def cmd(path):
        seen.append(path)
        # Like a config-gated preset: the flag depends on the snapshot's files.
        extra = ["--kv-cache-dtype", "fp8"] if os.path.isdir(path) else []
        return ["serve", path, *extra]

    app.engine_command = cmd  # type: ignore[method-assign]
    snap_a, snap_b = tmp_path / "snap-a", tmp_path / "snap-b"
    snap_a.mkdir()
    snap_b.mkdir()
    keyed = app._config_key(str(snap_a))
    assert seen[-1] == str(snap_a)
    # The snapshot dir is replaced by the model id; the gated flag is kept.
    assert keyed == app._config_key(str(snap_b))
    # Without the snapshot's files the gated flag is absent, so the key differs.
    assert keyed != app._config_key(str(tmp_path / "missing"))


def test_config_key_follows_the_served_repo(tmp_path, monkeypatch):
    app = _make_probe(tmp_path, monkeypatch, model="org/m")
    app.engine_command = lambda path: ["serve", path]  # type: ignore[method-assign]
    snap = str(tmp_path / "snap")
    app._model_repo = "org/m-FP8"
    fp8 = app._config_key(snap)
    app._model_repo = "org/m"
    # Same flags, different weights: an FP8 checkpoint and the bf16 model
    # compile different kernels, so they must not share a boot marker.
    assert app._config_key(snap) != fp8
    app._model_repo = "org/m-FP8"
    assert app._config_key(str(tmp_path / "other-snap")) == fp8


def test_write_marker_leaves_no_tmp_and_tolerates_concurrency(tmp_path):
    path = tmp_path / "d" / "m.json"
    errors = []

    def write(i):
        try:
            for _ in range(20):
                _cache.write_marker(path, {"i": i})
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert list(path.parent.iterdir()) == [path]
    assert _cache.read_marker(path) is not None


def test_write_marker_cleans_tmp_on_error(tmp_path):
    with pytest.raises(TypeError):
        _cache.write_marker(tmp_path / "m.json", {"x": object()})
    assert list(tmp_path.iterdir()) == []


A_SHA = "a" * 40


B_SHA = "b" * 40


OLD = "2026-01-01T00:00:00+00:00"


NEW = "2026-02-01T00:00:00+00:00"


class _Hub:
    """Stub huggingface_hub: snapshot_download lands in a per-commit dir."""

    def __init__(self, tmp_path, monkeypatch):
        self.root = tmp_path / "snaps"
        self.downloads: list = []
        self.next_commit = A_SHA
        self.dates = {A_SHA: OLD, B_SHA: NEW}
        self.upstream = A_SHA
        self.fail_info = False
        self.info_calls: list = []

        def snapshot_download(repo_id, **kwargs):
            self.downloads.append(kwargs.get("revision"))
            return str(self.snap(self.next_commit))

        module = types.ModuleType("huggingface_hub")
        module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)
        monkeypatch.setattr(_app, "_hub_commit", self.hub_commit)

    def snap(self, commit):
        path = self.root / commit
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}")
        (path / "model.safetensors").write_bytes(b"x")
        return path

    def hub_commit(self, model, revision, timeout):
        self.info_calls.append(revision)
        if self.fail_info:
            return None
        # A pinned lookup describes the commit the download landed on.
        sha = self.upstream if revision is None else self.next_commit
        return sha, self.dates.get(sha)


def _resolver(tmp_path, monkeypatch, **attrs):
    hub = _Hub(tmp_path, monkeypatch)
    app = _make_probe(tmp_path, monkeypatch, model="org/model", **attrs)
    app._boot = {}
    return hub, app


def _unpinned():
    return _cache.read_marker(_cache.snapshot_marker_path("org/model"))


def test_marker_records_commit_and_date(tmp_path, monkeypatch):
    hub, app = _resolver(tmp_path, monkeypatch)
    path = app._resolve_model()
    assert _unpinned() == {
        "model": "org/model",
        "revision": None,
        "path": path,
        "commit": A_SHA,
        "commit_date": OLD,
    }


def test_marker_commit_falls_back_to_dir_name_without_hub(tmp_path, monkeypatch):
    hub, app = _resolver(tmp_path, monkeypatch)
    hub.fail_info = True
    app._resolve_model()
    assert _unpinned()["commit"] == A_SHA
    assert _unpinned()["commit_date"] is None


def test_advance_replaces_older_keeps_newer_noop_same(tmp_path, monkeypatch):
    hub, app = _resolver(tmp_path, monkeypatch)
    app._resolve_model()
    a_path = str(hub.snap(A_SHA))
    b_path = str(hub.snap(B_SHA))
    advance = _cache.advance_snapshot_marker

    assert advance("org/model", b_path, B_SHA, NEW) is True
    assert _unpinned()["path"] == b_path
    # An older snapshot downloaded later does not win.
    assert advance("org/model", a_path, A_SHA, OLD) is False
    assert _unpinned()["commit"] == B_SHA
    # Same commit is a no-op.
    assert advance("org/model", b_path, B_SHA, NEW) is False
    # Unknown new date never replaces a known commit.
    assert advance("org/model", a_path, A_SHA, None) is False
    assert _unpinned()["commit"] == B_SHA


def test_advance_unknown_existing_date_and_old_format(tmp_path, monkeypatch):
    hub, app = _resolver(tmp_path, monkeypatch)
    app._resolve_model()
    b_path = str(hub.snap(B_SHA))
    marker = _cache.snapshot_marker_path("org/model")
    # Known commit with unknown date is replaced by a dated one.
    _cache.write_marker(marker, {**_unpinned(), "commit_date": None})
    assert _cache.advance_snapshot_marker("org/model", b_path, B_SHA, NEW)
    assert _unpinned()["commit_date"] == NEW

    # An old-format marker still resolves locally, and is replaced on advance.
    old_path = str(hub.snap(A_SHA))
    _cache.write_marker(marker, {"model": "org/model", "path": old_path})
    hub.downloads.clear()
    assert app._resolve_model() == old_path
    assert hub.downloads == []
    assert app._served_commit == A_SHA
    assert _cache.advance_snapshot_marker("org/model", b_path, B_SHA, NEW)
    assert _unpinned()["commit"] == B_SHA

    # Missing or unreadable marker is replaced regardless of dates.
    marker.write_text("not json")
    assert _cache.advance_snapshot_marker("org/model", old_path, A_SHA, None)
    assert _unpinned()["commit"] == A_SHA


def test_advance_repairs_marker_whose_snapshot_was_pruned(tmp_path, monkeypatch):
    hub, app = _resolver(tmp_path, monkeypatch)
    app._resolve_model()
    (hub.snap(A_SHA) / "config.json").unlink()
    # No date for the replacement, but the recorded copy is gone.
    b_path = str(hub.snap(B_SHA))
    assert _cache.advance_snapshot_marker("org/model", b_path, B_SHA, None)
    assert _unpinned()["path"] == b_path


def test_pinned_branch_off_main_does_not_advance_unpinned(tmp_path, monkeypatch):
    hub, app = _resolver(tmp_path, monkeypatch)
    app._resolve_model()
    # A newer-dated commit that is not main's head (a dev branch, a PR ref).
    hub.next_commit = B_SHA
    pinned = _make_probe(tmp_path, monkeypatch, model="org/model", model_revision="dev")
    pinned._boot = {}
    pinned._resolve_model()
    assert _unpinned()["commit"] == A_SHA


def test_pinned_download_advances_unpinned_marker(tmp_path, monkeypatch):
    hub, app = _resolver(tmp_path, monkeypatch, model_revision="v2")
    hub.next_commit = B_SHA
    hub.upstream = B_SHA  # the pinned revision is main's current head
    path = app._resolve_model()
    pinned = _cache.read_marker(_cache.snapshot_marker_path("org/model", "v2"))
    assert pinned["revision"] == "v2" and pinned["commit"] == B_SHA
    assert _unpinned()["path"] == path

    unpinned = _make_probe(tmp_path, monkeypatch, model="org/model")
    unpinned._boot = {}
    hub.downloads.clear()
    assert unpinned._resolve_model() == path
    assert hub.downloads == []
    assert unpinned._boot["resolve_source"] == "local"


def test_pin_hint_only_when_unpinned(tmp_path, monkeypatch, capsys):
    hub, app = _resolver(tmp_path, monkeypatch)
    app._resolve_model()
    assert "Pin model_revision" in capsys.readouterr().out
    (tmp_path / "p").mkdir()
    hub, pinned = _resolver(tmp_path / "p", monkeypatch, model_revision="v1")
    pinned._resolve_model()
    assert "Pin model_revision" not in capsys.readouterr().out


def _gauge():
    from prometheus_client import REGISTRY  # noqa: PLC0415

    return REGISTRY.get_sample_value(_app._UPSTREAM_GAUGE)


def test_upstream_newer_sets_boot_fields_and_gauge(tmp_path, monkeypatch):
    hub, app = _resolver(tmp_path, monkeypatch)
    app._resolve_model()
    marker_before = _unpinned()
    hub.upstream = B_SHA
    hub.downloads.clear()
    app._check_upstream()
    assert app._boot["model_commit"] == A_SHA
    assert app._boot["upstream_commit"] == B_SHA
    assert app._boot["upstream_newer"] is True
    assert _gauge() == 1.0
    # Report only: no download, markers untouched.
    assert hub.downloads == [] and _unpinned() == marker_before

    hub.upstream = A_SHA
    app._check_upstream()
    assert app._boot["upstream_newer"] is False
    assert _gauge() == 0.0


def test_upstream_check_queries_the_served_repo(tmp_path, monkeypatch):
    hub, app = _resolver(tmp_path, monkeypatch)
    app._resolve_model()
    asked = []
    monkeypatch.setattr(
        _app, "_hub_commit", lambda repo, rev, timeout: asked.append(repo) or None
    )
    # A recipe checkpoint was served: its own main is the comparison.
    app._model_repo = "org/model-FP8"
    app._check_upstream()
    assert asked == ["org/model-FP8"]


def test_upstream_check_failure_and_start_conditions(tmp_path, monkeypatch):
    hub, app = _resolver(tmp_path, monkeypatch)
    app._resolve_model()
    hub.fail_info = True
    app._check_upstream()
    assert "upstream_newer" not in app._boot and "model_commit" not in app._boot

    # Pinned apps and local paths never start the check.
    ran = []
    monkeypatch.setattr(LLMApp, "_check_upstream", lambda self: ran.append(1))
    (tmp_path / "p").mkdir()
    (tmp_path / "q").mkdir()
    _, pinned = _resolver(tmp_path / "p", monkeypatch, model_revision="v1")
    pinned._resolve_model()
    pinned._start_upstream_check()
    local = _make_probe(tmp_path / "q", monkeypatch)
    local._boot = {}
    local._resolve_model()
    local._start_upstream_check()
    # The unpinned Hub app does start it.
    app._start_upstream_check()
    deadline = time.monotonic() + 5
    while not ran and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ran == [1]


def test_kv_cpu_offload_adds_connector_sized_from_container_ram(monkeypatch):
    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 100 << 30)

    class A(LLMApp):
        model = "m"
        engine_presets = False
        kv_cpu_offload = 0.5

    cmd = A().engine_command("/m")
    cfg = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
    assert cfg["kv_connector"] == "OffloadingConnector"
    assert cfg["kv_connector_extra_config"]["cpu_bytes_to_use"] == 50 << 30


def test_kv_cpu_offload_off_by_default_and_user_flag_wins(monkeypatch):
    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 100 << 30)

    class Off(LLMApp):
        model = "m"
        engine_presets = False

    class User(LLMApp):
        model = "m"
        engine_presets = False
        kv_cpu_offload = 0.5
        engine_args = ["--kv-transfer-config", "{}"]

    assert "--kv-transfer-config" not in Off().engine_command("/m")
    cmd = User().engine_command("/m")
    assert cmd.count("--kv-transfer-config") == 1 and "{}" in cmd


def test_kv_cpu_offload_rejects_bad_fraction_and_sglang(monkeypatch):
    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 100 << 30)

    with pytest.raises(ValueError, match="kv_cpu_offload"):

        class Bad(LLMApp):
            model = "m"
            engine_presets = False
            kv_cpu_offload = 1.5

    class Sg(LLMApp):
        model = "m"
        engine = "sglang"
        engine_presets = False
        kv_cpu_offload = 0.5

    assert "--kv-transfer-config" not in Sg().engine_command("/m")


def test_kv_cpu_offload_caps_fraction_and_detects_equals_form(monkeypatch):
    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 100 << 30)

    with pytest.raises(ValueError, match="kv_cpu_offload"):

        class High(LLMApp):
            model = "m"
            engine_presets = False
            kv_cpu_offload = 0.95

    class Eq(LLMApp):
        model = "m"
        engine_presets = False
        kv_cpu_offload = 0.5
        engine_args = ["--kv_transfer_config={}"]

    assert "--kv-transfer-config" not in Eq().engine_command("/m")


def test_config_key_ignores_kv_connector_size(monkeypatch):
    class A(LLMApp):
        model = "m"
        engine_presets = False
        kv_cpu_offload = 0.5

    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 100 << 30)
    small = A()._config_key("/m")
    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 200 << 30)
    assert A()._config_key("/m") == small


GIB = 1 << 30


def _cgroup_tree(tmp_path, proc_cgroup, files):
    root = tmp_path / "cgroup"
    for rel, value in files.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(value)
    root.mkdir(exist_ok=True)
    proc = tmp_path / "proc_cgroup"
    proc.write_text(proc_cgroup)
    return str(root), str(proc)


@pytest.mark.parametrize(
    "proc_cgroup, files, expected",
    [
        # cgroup v2, own cgroup resolved from /proc/self/cgroup.
        ("0::/pods/p1/ctr\n", {"pods/p1/ctr/memory.max": str(60 * GIB)}, 60 * GIB),
        # A parent's tighter limit binds, and memory.high counts.
        (
            "0::/pods/p1/ctr\n",
            {"pods/p1/ctr/memory.max": "max", "pods/p1/memory.max": str(50 * GIB)},
            50 * GIB,
        ),
        (
            "0::/pods/p1/ctr\n",
            {
                "pods/p1/ctr/memory.max": str(60 * GIB),
                "pods/p1/ctr/memory.high": str(40 * GIB),
            },
            40 * GIB,
        ),
        # A cgroup namespace shows the container's own cgroup as the root.
        ("0::/\n", {"memory.max": str(30 * GIB)}, 30 * GIB),
        # cgroup v1 memory controller.
        (
            "5:memory:/docker/abc\n0::/\n",
            {"memory/docker/abc/memory.limit_in_bytes": str(40 * GIB)},
            40 * GIB,
        ),
        # The fal runner case: files present but empty, or unlimited.
        ("0::/\n", {"memory.max": "", "memory.high": ""}, 0),
        (
            "5:memory:/\n",
            {
                "memory/memory.limit_in_bytes": "9223372036854771712",
                "memory.max": "max",
            },
            0,
        ),
        # Nothing readable at all.
        ("", {}, 0),
        # A malformed line does not hide the memory controller after it.
        (
            "garbage\n\n5:memory:/docker/abc\n",
            {"memory/docker/abc/memory.limit_in_bytes": str(40 * GIB)},
            40 * GIB,
        ),
        # Hybrid: v2 and v1 limits both count.
        (
            "0::/pods/p1\n5:memory:/docker/abc\n",
            {
                "pods/p1/memory.max": str(50 * GIB),
                "memory/docker/abc/memory.limit_in_bytes": str(45 * GIB),
            },
            45 * GIB,
        ),
        # No cgroup namespace: /proc shows a host path, but the mount is the
        # container's own cgroup, so its limit sits at the mount root.
        ("0::/docker/abc\n", {"memory.max": str(60 * GIB)}, 60 * GIB),
    ],
)
def test_container_ram_reads_the_tightest_cgroup_limit_never_meminfo(
    tmp_path, proc_cgroup, files, expected
):
    root, proc = _cgroup_tree(tmp_path, proc_cgroup, files)
    assert _app._container_ram_bytes(root, proc) == expected


@pytest.mark.parametrize(
    "gpu, num_gpus, per_gpu_gib",
    [
        (("NVIDIA RTX PRO 6000 Blackwell Server Edition", 97887), 1, 60),
        (("NVIDIA RTX PRO 6000 Blackwell Server Edition", 97887), 2, 60),
        (("NVIDIA H200", 143771), 1, 112),
        (("NVIDIA H200 NVL", 143771), 1, 112),
        (("NVIDIA B200", 183359), 1, 210),
        (("NVIDIA H100 80GB HBM3", 81559), 1, 60),
        (("NVIDIA H100 80GB HBM3 MIG 3g.40gb", 40192), 1, 16),
        (("NVIDIA A30", 24576), 1, 16),
        (None, 1, 16),
    ],
)
def test_kv_cpu_offload_without_cgroup_limit_uses_fal_ram_per_gpu(
    monkeypatch, capsys, gpu, num_gpus, per_gpu_gib
):
    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 0)

    class A(LLMApp):
        model = "m"
        engine_presets = False
        kv_cpu_offload = 0.5

    A.num_gpus = num_gpus
    monkeypatch.setattr(A, "_gpu_name", staticmethod(lambda: gpu))
    monkeypatch.setattr(A, "_gpu_type", staticmethod(lambda: "RTX"))
    app = A()
    cmd = app.engine_command("/m")
    app.engine_command("/m")
    cfg = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
    expected = int(per_gpu_gib * GIB * num_gpus * 0.5)
    assert cfg["kv_connector_extra_config"]["cpu_bytes_to_use"] == expected
    out = capsys.readouterr().out
    assert out.count("no container memory limit is readable") == 1


def test_kv_cpu_offload_gib_overrides_size_and_turns_tier_on(monkeypatch):
    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 100 * GIB)

    class Sized(LLMApp):
        model = "m"
        engine_presets = False
        kv_cpu_offload_gib = 24

    cmd = Sized().engine_command("/m")
    cfg = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
    assert cfg["kv_connector_extra_config"]["cpu_bytes_to_use"] == 24 * GIB

    class Off(Sized):
        kv_cpu_offload = 0

    assert "--kv-transfer-config" not in Off().engine_command("/m")

    for bad in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="kv_cpu_offload_gib"):

            class Bad(LLMApp):
                model = "m"
                engine_presets = False
                kv_cpu_offload_gib = bad


def test_kv_cpu_offload_gib_is_capped_by_a_readable_limit(monkeypatch, capsys):
    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 60 * GIB)

    class Big(LLMApp):
        model = "m"
        engine_presets = False
        kv_cpu_offload_gib = 100

    cmd = Big().engine_command("/m")
    cfg = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
    assert cfg["kv_connector_extra_config"]["cpu_bytes_to_use"] == int(60 * GIB * 0.8)
    assert "exceeds" in capsys.readouterr().out


def test_unused_kv_tier_is_not_sized(monkeypatch, capsys):
    # No warning or GPU probe when the tier would be dropped anyway.
    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 0)

    def no_probe():
        raise AssertionError("GPU probed for an unused tier")

    class Sg(LLMApp):
        model = "m"
        engine = "sglang"
        engine_presets = False
        kv_cpu_offload = 0.5

    class User(LLMApp):
        model = "m"
        engine_presets = False
        kv_cpu_offload = 0.5
        engine_args = ["--kv-transfer-config", "{}"]

    for cls in (Sg, User):
        monkeypatch.setattr(cls, "_gpu_name", staticmethod(no_probe))
        cls().engine_command("/m")
    assert "WARNING" not in capsys.readouterr().out


def test_default_cache_namespace_keys_gpu_engine_and_torch(tmp_path, monkeypatch):
    monkeypatch.setattr(_app, "_engine_version", lambda engine: "9.9")
    monkeypatch.setattr(_app, "_torch_tag", lambda: "2.8.0")
    monkeypatch.setattr(_app.LLMApp, "_gpu_type", staticmethod(lambda: "RTXPRO6000"))
    app = _make_probe(tmp_path, monkeypatch, cache_namespace=None)
    app.setup()
    cache_dir = os.environ["VLLM_CACHE_ROOT"]
    assert f"{app.engine}-9.9-2.8.0-RTXPRO6000" in cache_dir
    for key in ("TORCH_EXTENSIONS_DIR", "CUTE_DSL_CACHE_DIR"):
        assert key in os.environ


def test_persistent_caches_off_leaves_engine_env_alone(tmp_path, monkeypatch):
    app = _make_probe(tmp_path, monkeypatch, persistent_caches=False)
    app.setup()
    assert "VLLM_CACHE_ROOT" not in os.environ
    assert "TORCH_EXTENSIONS_DIR" not in os.environ


def test_models_refused_while_draining(served):
    app, url, _, _ = served
    app._draining = True
    try:
        assert httpx.get(f"{url}/v1/models").status_code == 503
    finally:
        app._draining = False
