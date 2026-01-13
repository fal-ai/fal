import hashlib
import os
import posixpath
from functools import cached_property
from typing import TYPE_CHECKING, Optional

from fsspec import AbstractFileSystem

from fal.upload import (
    MULTIPART_CHUNK_SIZE,
    MULTIPART_MAX_CONCURRENCY,
    MULTIPART_THRESHOLD,
    DataFileMultipartUpload,
)

if TYPE_CHECKING:
    import httpx

USER_AGENT = "fal-sdk/1.14.0 (python)"


def _compute_md5(lpath, chunk_size=8192):
    hasher = hashlib.md5()
    with open(lpath, "rb") as fobj:
        for chunk in iter(lambda: fobj.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class FalFileSystem(AbstractFileSystem):
    def __init__(
        self,
        *,
        host: Optional[str] = None,
        team: Optional[str] = None,
        profile: Optional[str] = None,
        **kwargs,
    ):
        self.host = host
        self.team = team
        self.profile = profile
        super().__init__(**kwargs)

    @cached_property
    def _client(self) -> "httpx.Client":
        from httpx import Client, Timeout

        from fal.api.client import SyncServerlessClient

        client = SyncServerlessClient(
            host=self.host,
            team=self.team,
            profile=self.profile,
        )

        return Client(
            base_url=client._rest_url,
            headers={
                **client._credentials.to_headers(),
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

    def _put_file_multipart(self, lpath, rpath, size, progress):
        md5 = _compute_md5(lpath)

        num_parts = max(1, (size + MULTIPART_CHUNK_SIZE - 1) // MULTIPART_CHUNK_SIZE)
        task = progress.add_task(f"{os.path.basename(lpath)}", total=num_parts)

        def on_part_complete(part_number: int):
            progress.advance(task)

        multipart = DataFileMultipartUpload(
            client=self._client,
            target_path=rpath,
            chunk_size=MULTIPART_CHUNK_SIZE,
            max_concurrency=MULTIPART_MAX_CONCURRENCY,
        )

        etag = multipart.upload_file(lpath, on_part_complete=on_part_complete)

        if etag and etag != md5:
            raise RuntimeError(
                f"MD5 mismatch on {rpath}: {etag} != {md5}, " "please contact support"
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
