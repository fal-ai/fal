from __future__ import annotations

import hashlib
import os
import zipfile

import requests
from fal_serverless.api import FAL_SERVERLESS_DEFAULT_URL
from fal_serverless.auth import USER
from pathspec import PathSpec

CHUNK_SIZE = 1024 * 1024 * 10  # 10 MB


def _check_hash(target_path: str, hash_string: str, token: str) -> bool:
    url = f"{REST_URL}/files/dir/check_hash/{target_path}"
    headers = {"Authorization": token, "Content-Type": "application/json"}
    result = requests.post(url=url, headers=headers, json={"hash": hash_string})
    return result.status_code == 200 and result.json() is True


def _upload_file(
    source_path: str, target_path: str, token: str, unzip: bool = False
) -> None:

    url = f"{REST_URL}/files/file/local/{target_path}"
    headers = {"Authorization": token}

    with open(source_path, "rb") as file_to_upload:
        files = {"file_upload": file_to_upload}
        params = {"unzip": unzip}
        response = requests.post(url, headers=headers, files=files, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(
            f"Failed to upload file. Server returned status code {response.status_code} and message {response.text}"
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


def sync_dir(local_dir: str, remote_dir: str, force_upload=False) -> str:
    local_dir_abs = os.path.expanduser(local_dir)
    if not os.path.isabs(remote_dir) or not remote_dir.startswith("/data"):
        raise ValueError(
            "'remote_dir' must be an absolute path starting with `/data`, e.g. '/data/sync/my_dir'"
        )

    remote_dir = remote_dir.replace("/data/", "", 1)

    # Compute the local directory hash
    local_hash = _compute_directory_hash(local_dir_abs)

    token = USER.bearer_token

    print(f"Syncing {local_dir} with {remote_dir}...")

    if _check_hash(remote_dir, local_hash, token) and not force_upload:
        print(f"{remote_dir} already uploaded and matches {local_dir}")
        return remote_dir

    with open(os.path.join(local_dir_abs, ".fal_hash"), "w") as f:
        f.write(local_hash)

    # Zip the local directory
    zip_path = f"{local_dir_abs}.zip"

    _zip_directory(local_dir_abs, zip_path)

    # Upload the zipped directory to the serverless environment
    _upload_file(zip_path, remote_dir, token, unzip=True)

    os.remove(zip_path)

    print("Done")

    # Return the full path to the remote directory
    return remote_dir


def _get_rest_host_url(url: str) -> str:
    assert url.startswith("api."), "Expected FAL_HOST format to be `api.<env>.fal.ai`"
    return "https://" + url.replace("api", "rest", 1)  # to replace just once


REST_URL = _get_rest_host_url(FAL_SERVERLESS_DEFAULT_URL)
