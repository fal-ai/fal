from __future__ import annotations

from fal.files import find_project_root, find_pyproject_toml, parse_pyproject_toml


def is_app_name(app_ref: tuple[str, str | None]) -> bool:
    is_single_file = app_ref[1] is None
    is_python_file = app_ref[0].endswith(".py")

    return is_single_file and not is_python_file


def get_app_data_from_toml(app_name):
    toml_path = find_pyproject_toml()

    if toml_path is None:
        raise ValueError("No pyproject.toml file found.")

    fal_data = parse_pyproject_toml(toml_path)
    apps = fal_data.get("apps", {})

    try:
        app_data = apps[app_name]
    except KeyError:
        raise ValueError(f"App {app_name} not found in pyproject.toml")

    try:
        app_ref = app_data["ref"]
    except KeyError:
        raise ValueError(f"App {app_name} does not have a ref key in pyproject.toml")

    # Convert the app_ref to a path relative to the project root
    project_root, _ = find_project_root(None)
    app_ref = str(project_root / app_ref)

    app_auth = app_data.get("auth", "private")
    app_deployment_strategy = app_data.get("deployment_strategy", "recreate")

    return app_ref, app_auth, app_deployment_strategy
