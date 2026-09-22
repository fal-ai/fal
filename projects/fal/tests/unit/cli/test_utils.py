from fal.cli._utils import get_app_data_from_toml


def _write_project(tmp_path, name: str, body: str):
    project_dir = tmp_path / name
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(body)
    return project_dir


def test_get_app_data_from_toml_does_not_leak_secrets_across_chdir(
    tmp_path, monkeypatch
):
    # Two distinct projects that happen to define an app under the same
    # name, each with its own secrets and auth mode. A process that deploys
    # both in sequence (an orchestrator, a notebook, a CI worker) must not
    # have project B's lookup return project A's secrets just because
    # project A was resolved first in this process.
    project_a = _write_project(
        tmp_path,
        "project_a",
        """
        [tool.fal.apps.my-app]
        ref = "src/app_a/inference.py::AppA"
        secrets = ["SECRET_A"]
        auth = "private"
        """,
    )
    project_b = _write_project(
        tmp_path,
        "project_b",
        """
        [tool.fal.apps.my-app]
        ref = "src/app_b/inference.py::AppB"
        secrets = ["SECRET_B"]
        auth = "public"
        """,
    )

    monkeypatch.chdir(project_a)
    app_a = get_app_data_from_toml("my-app")

    monkeypatch.chdir(project_b)
    app_b = get_app_data_from_toml("my-app")

    assert app_a.auth == "private"
    assert app_a.options.host["secrets"] == ["SECRET_A"]

    assert app_b.auth == "public"
    assert app_b.options.host["secrets"] == ["SECRET_B"]
