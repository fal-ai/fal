from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path

import openapi_fal_rest.api.files.check_dir_hash as check_dir_hash_api
import openapi_fal_rest.api.files.upload_local_file as upload_local_file_api
import openapi_fal_rest.models.body_upload_local_file as upload_file_model
import openapi_fal_rest.models.hash_check as hash_check_model
import openapi_fal_rest.types as rest_types
from pathspec import PathSpec

from fal.rest_client import REST_CLIENT


def _check_hash(target_path: str, hash_string: str) -> bool:
    response = check_dir_hash_api.sync_detailed(
        target_path,
        client=REST_CLIENT,
        json_body=hash_check_model.HashCheck(hash_string),
    )

    res: bool = response.parsed  # type: ignore
    return response.status_code == 200 and res


def _upload_file(source_path: str, target_path: str, unzip: bool = False):
    with open(source_path, "rb") as file_to_upload:
        body = upload_file_model.BodyUploadLocalFile(
            rest_types.File(
                payload=file_to_upload,
                # We need to set a file_name, otherwise the server errors
                # processing the file
                file_name=os.path.basename(source_path),
            )
        )

        response = upload_local_file_api.sync_detailed(
            target_path,
            client=REST_CLIENT,
            unzip=unzip,
            multipart_data=body,
        )

    if response.status_code != 200:
        raise Exception(
            "Failed to upload file. "
            "Server returned status code "
            f"{response.status_code} and message {response.parsed}"
        )


def _compute_directory_hash(dir_path: str) -> str:
    hash = hashlib.sha256()
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file != ".fal_hash":
                with open(file_path, "rb") as f:
                    hash.update(f.read())
    return hash.hexdigest()


def _load_gitignore_patterns(dir_path: str) -> list:
    # TODO: consider looking at .gitignore files in child directories
    gitignore_path = os.path.join(dir_path, ".gitignore")
    if os.path.exists(gitignore_path):
        with open(gitignore_path) as f:
            gitignore_patterns = f.read().splitlines()
    else:
        gitignore_patterns = []
    return gitignore_patterns


def _is_ignored(file_path: str, gitignore_patterns: list[str]) -> bool:
    pathspec = PathSpec.from_lines("gitwildmatch", gitignore_patterns)
    return pathspec.match_file(file_path)


def _zip_directory(dir_path: str, zip_path: str) -> None:
    gitignore_patterns = _load_gitignore_patterns(dir_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, dir_path)

                if not _is_ignored(relative_path, gitignore_patterns):
                    arcname = relative_path
                    zipf.write(file_path, arcname)


def sync_dir(local_dir: str | Path, remote_dir: str, force_upload=False) -> str:
    local_dir_abs = os.path.expanduser(local_dir)
    if not os.path.isabs(remote_dir) or not remote_dir.startswith("/data"):
        raise ValueError(
            "'remote_dir' must be an absolute path starting with `/data`, "
            "e.g. '/data/sync/my_dir'"
        )

    remote_dir = remote_dir.replace("/data/", "", 1)

    # Compute the local directory hash
    local_hash = _compute_directory_hash(local_dir_abs)

    print(f"Syncing {local_dir} with {remote_dir}...")

    if _check_hash(remote_dir, local_hash) and not force_upload:
        print(f"{remote_dir} already uploaded and matches {local_dir}")
        return remote_dir

    with open(os.path.join(local_dir_abs, ".fal_hash"), "w") as f:
        f.write(local_hash)

    # Zip the local directory
    zip_path = f"{local_dir_abs}.zip"

    _zip_directory(local_dir_abs, zip_path)

    # Upload the zipped directory to the serverless environment
    _upload_file(zip_path, remote_dir, unzip=True)

    os.remove(zip_path)

    print("Done")

    # Return the full path to the remote directory
    return remote_dir
