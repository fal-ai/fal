import sys

import pytest

from dev.remote.fakes import make_dist

pytestmark = pytest.mark.usefixtures("dev_account")


def test_child_reports_versions_and_processes_pth_files(tmp_path, monkeypatch):
    from fal.dev.remote import child as app_child

    make_dist(tmp_path, "overlaypkg", "9.9.9")
    (tmp_path / "extra.pth").write_text(str(tmp_path / "extra") + "\n")
    (tmp_path / "extra").mkdir()
    monkeypatch.setenv("FAL_DEV_OVERLAY", str(tmp_path))
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    app_child._activate_overlay()
    assert str(tmp_path / "extra") in sys.path
    found = app_child._versions(["overlaypkg", "not-a-package-xyz"])
    assert found["overlaypkg"]["version"] == "9.9.9"
    assert found["not-a-package-xyz"] is None
