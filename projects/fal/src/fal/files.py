import concurrent.futures
import math
import os
import posixpath
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import TYPE_CHECKING

from fsspec import AbstractFileSystem

if TYPE_CHECKING:
    import httpx

USER_AGENT = "fal-sdk/1.14.0 (python)"
MULTIPART_THRESHOLD = 10 * 1024 * 1024  # 10MB
MULTIPART_CHUNK_SIZE = 10 * 1024 * 1024  # 10MB
MULTIPART_WORKERS = 10


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
        from httpx import Client, Timeout

        from fal.flags import REST_URL
        from fal.sdk import get_default_credentials

        creds = get_default_credentials()
        return Client(
            base_url=REST_URL,
            headers={
                **creds.to_headers(),
                "User-Agent": USER_AGENT,
            },
            timeout=Timeout(
                connect=30,
                read=4 * 60,  # multipart complete can take time
                write=5 * 60,  # we could be uploading slowly
                pool=30,
            ),
        )

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

    def _put_file_part(self, rpath, lpath, upload_id, part_number, chunk_size):
        offset = (part_number - 1) * chunk_size
        with open(lpath, "rb") as fobj:
            fobj.seek(offset)
            chunk = fobj.read(chunk_size)
            response = self._request(
                "PUT",
                f"/files/file/multipart/{rpath}/{upload_id}/{part_number}",
                files={"file_upload": (posixpath.basename(lpath), chunk)},
            )
            data = response.json()
            return {
                "part_number": data["part_number"],
                "etag": data["etag"],
            }

    def _put_file_multipart(self, lpath, rpath, size, progress):
        response = self._request(
            "POST",
            f"/files/file/multipart/{rpath}/initiate",
        )
        upload_id = response.json()["upload_id"]

        parts = []
        num_parts = math.ceil(size / MULTIPART_CHUNK_SIZE)
        md5 = _compute_md5(lpath)

        task = progress.add_task(f"{os.path.basename(lpath)}", total=num_parts)

        with ThreadPoolExecutor(max_workers=MULTIPART_WORKERS) as executor:
            futures = []

            for part_number in range(1, num_parts + 1):
                futures.append(
                    executor.submit(
                        self._put_file_part,
                        rpath,
                        lpath,
                        upload_id,
                        part_number,
                        MULTIPART_CHUNK_SIZE,
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                parts.append(future.result())
                progress.advance(task)

        response = self._request(
            "POST",
            f"/files/file/multipart/{rpath}/{upload_id}/complete",
            json={"parts": parts},
        )
        data = response.json()
        if data["etag"] != md5:
            raise RuntimeError(
                f"MD5 mismatch on {rpath}: {data['etag']} != {md5}, "
                "please contact support"
            )

    def put_file(self, lpath, rpath, mode="overwrite", **kwargs):
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

        if os.path.isdir(lpath):
            return

        abs_rpath = self._abspath(rpath)

        size = os.path.getsize(lpath)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            if size > MULTIPART_THRESHOLD:
                self._put_file_multipart(lpath, abs_rpath, size, progress)
            else:
                task = progress.add_task(f"{os.path.basename(lpath)}", total=1)
                with open(lpath, "rb") as fobj:
                    self._request(
                        "POST",
                        f"/files/file/local/{abs_rpath}",
                        files={"file_upload": (posixpath.basename(lpath), fobj)},
                    )
                progress.advance(task)
        self.dircache.clear()

    def put_file_from_url(self, url, rpath, mode="overwrite", **kwargs):
        abs_rpath = self._abspath(rpath)
        self._request(
            "POST",
            f"/files/file/url/{abs_rpath}",
            json={"url": url},
            timeout=10 * 60,  # 10 minutes in seconds
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
        # Delegate to server-side rename
        self.rename(path1, path2)
