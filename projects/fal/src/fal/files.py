import asyncio
import os
import posixpath
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp
from fsspec import AbstractFileSystem

if TYPE_CHECKING:
    import httpx

USER_AGENT = "fal-sdk/1.14.0 (python)"
TUSD_THRESHOLD = 20 * 1024 * 1024


def _compute_md5(lpath, chunk_size=8192):
    import hashlib

    hasher = hashlib.md5()
    with open(lpath, "rb") as fobj:
        for chunk in iter(lambda: fobj.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class FalFileSystem(AbstractFileSystem):
    @cached_property
    def _client(self) -> "httpx.Client":
        from httpx import Client

        from fal.flags import REST_URL
        from fal.sdk import get_default_credentials

        creds = get_default_credentials()
        return Client(
            base_url=REST_URL,
            headers={
                **creds.to_headers(),
                "User-Agent": USER_AGENT,
            },
        )

    @cached_property
    def _tusd_uploader(self):
        from fal.flags import REST_URL
        from fal.sdk import get_default_credentials
        from fal.tusd.tusd import TusdUploader

        creds = get_default_credentials()
        server_url = f"{REST_URL}/files/tusd/"
        headers = {
            **creds.to_headers(),
            "User-Agent": USER_AGENT,
        }

        return TusdUploader(server_url, headers)

    def _request(self, method, path, **kwargs):
        from fal.exceptions import FalServerlessException

        response = self._client.request(method, path, **kwargs)
        if response.status_code != 200:
            try:
                detail = response.json()["detail"]
            except Exception:
                detail = response.text
            raise FalServerlessException(detail)
        return response

    def _abspath(self, rpath):
        if rpath.startswith("/"):
            return rpath

        cwd = "/data"
        if rpath in [".", ""]:
            return cwd

        return posixpath.join(cwd, rpath)

    def _ls(self, path):
        response = self._request("GET", f"/files/list/{path}")
        files = response.json()
        return sorted(
            (
                {
                    "name": entry["path"],
                    "size": entry["size"],
                    "type": "file" if entry["is_file"] else "directory",
                    "mtime": entry["updated_time"],
                }
                for entry in files
            ),
            key=lambda x: x["name"],
        )

    def ls(self, path, detail=True, **kwargs):
        abs_path = self._abspath(path)
        if abs_path in self.dircache:
            entries = self.dircache[abs_path]
        elif abs_path in ["/", "", "."]:
            entries = [
                {
                    "name": "/data",
                    "size": 0,
                    "type": "directory",
                    "mtime": 0,
                }
            ]
        else:
            entries = self._ls(abs_path)
        self.dircache[abs_path] = entries

        if detail:
            return entries

        return [entry["name"] for entry in entries]

    def info(self, path, **kwargs):
        abs_path = self._abspath(path)
        if abs_path == "/":
            return {
                "name": "/",
                "size": 0,
                "type": "directory",
                "mtime": 0,
            }
        parent = posixpath.dirname(abs_path)
        entries = self.ls(parent, detail=True)
        for entry in entries:
            if entry["name"] == abs_path:
                return entry
        raise FileNotFoundError(f"File not found: {abs_path}")

    def get_file(self, rpath, lpath, **kwargs):
        abs_rpath = self._abspath(rpath)
        if self.isdir(abs_rpath):
            os.makedirs(lpath, exist_ok=True)
            return

        with open(lpath, "wb") as fobj:
            response = self._request("GET", f"/files/file/{abs_rpath}")
            fobj.write(response.content)

    def put_file(self, lpath, rpath, mode="overwrite", **kwargs):
        if os.path.isdir(lpath):
            return

        abs_rpath = self._abspath(rpath)

        size = os.path.getsize(lpath)
        if size >= TUSD_THRESHOLD:
            max_retries = 3
            retry_count = 0

            while retry_count < max_retries:
                try:
                    self._tusd_uploader.upload(lpath, file_dir=rpath)
                    break
                except Exception as e:
                    status = None
                    if isinstance(e, aiohttp.ClientResponseError):
                        status = e.status
                    elif hasattr(e, "status"):
                        status = e.status
                    elif hasattr(e, "status_code"):
                        status = e.status_code

                    if status == 413:
                        from fal.exceptions import FalFilesException

                        raise FalFilesException(
                            "Max File size exceeded. File larger than 100GB."
                        )
                    elif status == 400:
                        from fal.exceptions import FalFilesException

                        raise FalFilesException("Invalid filename")
                    elif status and status >= 500:
                        from fal.exceptions import FalFilesException

                        raise FalFilesException("Internal server error")
                    elif status == 409:
                        retry_count += 1
                        if retry_count >= max_retries:
                            from fal.exceptions import FalFilesException

                            raise FalFilesException("File exists")
                        continue
                    elif status == 404:
                        from fal.tusd.cache import remove_from_cache
                        from fal.tusd.tusd import compute_hash

                        file_hash = compute_hash(Path(lpath))
                        asyncio.run(remove_from_cache(file_hash=file_hash))
                        retry_count += 1
                        if retry_count >= max_retries:
                            from fal.exceptions import FalFilesException

                            raise FalFilesException("Not found")
                        continue
                    else:
                        raise
        else:
            with open(lpath, "rb") as fobj:
                self._request(
                    "POST",
                    f"/files/file/local/{abs_rpath}",
                    files={"file_upload": (posixpath.basename(lpath), fobj)},
                )
        self.dircache.clear()

    def put_file_from_url(self, url, rpath, mode="overwrite", **kwargs):
        abs_rpath = self._abspath(rpath)
        self._request(
            "POST",
            f"/files/file/url/{abs_rpath}",
            json={"url": url},
        )
        self.dircache.clear()

    def rm(self, path, **kwargs):
        abs_path = self._abspath(path)
        self._request(
            "DELETE",
            f"/files/file/{abs_path}",
        )
        self.dircache.clear()

    def rename(self, path, destination, **kwargs):
        abs_path = self._abspath(path)
        abs_dest = self._abspath(destination)
        self._request(
            "POST",
            f"/files/rename/{abs_path}",
            json={"destination": abs_dest},
        )
        self.dircache.clear()

    def mv(self, path1, path2, recursive=False, maxdepth=None, **kwargs):
        self.rename(path1, path2)
