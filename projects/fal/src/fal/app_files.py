from __future__ import annotations

import os
import sys
from pathlib import Path


def get_app_files_relative_path(
    local_file_path: str | None, app_files_context_dir: str | None
) -> str | None:
    if local_file_path is None:
        return None

    base_path = Path(local_file_path).resolve()
    if base_path.is_dir():
        original_script_dir = base_path
    else:
        original_script_dir = base_path.parent

    if app_files_context_dir:
        context_path = Path(app_files_context_dir)
        if context_path.is_absolute():
            final_script_dir = context_path.resolve()
        else:
            final_script_dir = (original_script_dir / context_path).resolve()

        relative_path = os.path.relpath(original_script_dir, final_script_dir)
        if os.sep != "/":
            relative_path = relative_path.replace(os.sep, "/")
        return relative_path
    else:
        return "."


def include_app_files_path(app_files_relative_path: str | None) -> None:
    base_cloud_dir = Path("/app")

    # In case of container apps, the /app directory is not created by default
    # so we need to check if it exists before proceeding.
    if not base_cloud_dir.exists():
        return

    if not app_files_relative_path or app_files_relative_path == ".":
        final_path = base_cloud_dir
    else:
        final_path = base_cloud_dir / app_files_relative_path

    # Create the final path if it doesn't exist. This is for cases where the app
    # is not in root and its parent directory is not in app_files.
    final_path.mkdir(parents=True, exist_ok=True)

    # Add local files deployment paths to sys.path so imports work correctly in
    # the isolate agent.
    for path in (final_path, base_cloud_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.append(path_str)

    # Change the current working directory to the app path so app files are
    # accessible through relative paths.
    if Path.cwd() != final_path:
        os.chdir(str(final_path))
