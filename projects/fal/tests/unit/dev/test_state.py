from fal.dev import state


def test_state_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")

    state.update(tmp_path, "host", name="demo", hot_dir="/work/k")
    state.update(tmp_path, "host", hot_dir=None)

    data = state.load(tmp_path, "host")
    assert data["name"] == "demo"
    assert "hot_dir" not in data
    assert state.load(tmp_path, "other-host") == {}
    assert state.load(tmp_path / "sub", "host") == {}


def test_box_state_is_kept_per_box_and_host(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")

    state.update_box("devbox-a", "host", team="red")
    state.update_box("devbox-b", "host", team="blue")

    assert state.load_box("devbox-a", "host")["team"] == "red"
    assert state.load_box("devbox-b", "host")["team"] == "blue"
    assert state.load_box("devbox-a", "other-host") == {}


def test_state_ignores_corrupt_file(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    path = state._folder(tmp_path, None)
    path.parent.mkdir(parents=True)
    path.write_text("not json")

    assert state.load(tmp_path, None) == {}


def test_state_update_skips_unchanged_writes(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    state.update(tmp_path, None, name="demo")
    path = state._folder(tmp_path, None)
    before = path.stat().st_mtime_ns

    state.update(tmp_path, None, name="demo")

    assert path.stat().st_mtime_ns == before
