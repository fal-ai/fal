from __future__ import annotations

import hashlib
import os
import shutil
import zipfile

from .api import isolated

CHUNK_SIZE = 1024 * 1024 * 10  # 10 MB


def _read_file_chunk(file_path: str, chunk_number: int) -> bytes:
    with open(file_path, "rb") as file:
        file.seek(chunk_number * CHUNK_SIZE)
        return file.read(CHUNK_SIZE)


@isolated()
def _write_file_chunk(destination_path: str, chunk_data: bytes) -> None:
    with open(destination_path, "ab") as file:
        file.write(chunk_data)


@isolated()
def _unzip_target_directory(zip_file_path: str, target_direcroty: str) -> None:
    shutil.rmtree(target_direcroty, ignore_errors=True)
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(target_direcroty)
    os.remove(zip_file_path)


@isolated()
def _check_hash(target_path: str, hash_string: str) -> bool:
    try:
        with open(os.path.join(target_path, ".fal_hash")) as f:
            return hash_string == f.read()
    except FileNotFoundError:
        return False


@isolated()
def _clear_destination_file(destination_path):
    os.makedirs("/data/sync", exist_ok=True)
    with open(destination_path, "wb") as f:
        f.truncate(0)


def _upload_file(source_path: str, destination_path: str) -> None:
    file_size = os.path.getsize(source_path)
    total_chunks = (file_size // CHUNK_SIZE) + (1 if file_size % CHUNK_SIZE else 0)

    # Clear the destination file
    _clear_destination_file(destination_path)
    for chunk_number in range(total_chunks):
        chunk_data = _read_file_chunk(source_path, chunk_number)
        _write_file_chunk(destination_path, chunk_data)


def _compute_directory_hash(dir_path: str) -> str:
    hash = hashlib.sha256()
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, "rb") as f:
                hash.update(f.read())
    return hash.hexdigest()


def _zip_directory(dir_path: str, zip_path: str) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = file_path[len(dir_path) :]
                zipf.write(file_path, arcname)


def sync_dir(local_dir: str, remote_dir: str, force_upload=False) -> str:
    if not os.path.isabs(remote_dir) or not remote_dir.startswith("/data"):
        raise ValueError(
            "'remote_dir' must be an absolute path starting with `/data`, e.g. '/data/sync/my_dir'"
        )

    # Compute the local directory hash
    local_hash = _compute_directory_hash(local_dir)

    print(f"Syncing {local_dir} with {remote_dir}...")

    if _check_hash(remote_dir, local_hash) and not force_upload:
        print(f"{remote_dir} already uploaded and matches {local_dir}")
        return remote_dir

    with open(os.path.join(local_dir, ".fal_hash"), "w") as f:
        f.write(local_hash)

    # Zip the local directory
    zip_path = f"{local_dir}.zip"

    _zip_directory(local_dir, zip_path)

    # Upload the zipped directory to the serverless environment
    zip_remote_path = os.path.join("/data/sync", os.path.basename(zip_path))
    _upload_file(zip_path, zip_remote_path)
    _unzip_target_directory(zip_remote_path, remote_dir)

    # Remove the zipped directory
    os.remove(zip_path)
    print("Done")

    # Return the full path to the remote directory
    return remote_dir
