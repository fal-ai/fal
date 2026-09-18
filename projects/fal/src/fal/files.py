import os
import posixpath
from functools import cached_property
from typing import TYPE_CHECKING, Optional

from fsspec import AbstractFileSystem

from fal._user_agent import USER_AGENT
from fal.upload import (
    MULTIPART_CHUNK_SIZE,
    MULTIPART_MAX_CONCURRENCY,
    MULTIPART_THRESHOLD,
    DataFileMultipartUpload,
    ProgressFileReader,
)

if TYPE_CHECKING:
    import httpx


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
        from rich.markup import escape

        # A task description is parsed as Rich markup, so a name containing
        # tag-like brackets would be mangled or raise.
        name = escape(os.path.basename(lpath))
        task = progress.add_task(f"Uploading {name}", total=size)

        def on_bytes_uploaded(uploaded: int):
            progress.update(task, completed=uploaded)

        multipart = DataFileMultipartUpload(
            client=self._client,
            target_path=rpath,
            chunk_size=MULTIPART_CHUNK_SIZE,
            max_concurrency=MULTIPART_MAX_CONCURRENCY,
        )

        etag = multipart.upload_file(
            lpath, on_bytes_uploaded=on_bytes_uploaded, compute_md5=True
        )

        # Concurrent parts report their running totals outside the tracker lock,
        # so the last callback to arrive is not necessarily the highest. Settle
        # the bar on the size that was actually uploaded.
        progress.update(task, completed=size)

        # The digest is taken from the bytes that were read and sent, so this
        # compares the stored object against the upload, not against the file on
        # disk. An edit that changes the file's length fails the byte-count check
        # in upload_file; one that keeps it identical is not detected here.
        md5 = multipart.content_md5
        if etag and etag != md5:
            raise RuntimeError(
                f"MD5 mismatch on {rpath}: {etag} != {md5}, please contact support"
            )

    def put_file(self, lpath, rpath, mode="overwrite", **kwargs):
        from rich.markup import escape
        from rich.progress import (
            BarColumn,
            DownloadColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
        )

        if os.path.isdir(lpath):
            return

        abs_rpath = self._abspath(rpath)

        size = os.path.getsize(lpath)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            DownloadColumn(),
        ) as progress:
            if size > MULTIPART_THRESHOLD:
                self._put_file_multipart(lpath, abs_rpath, size, progress)
            else:
                # A zero total renders as an indeterminate pulse forever.
                total = size or 1
                task = progress.add_task(escape(os.path.basename(lpath)), total=total)
                with open(lpath, "rb") as fobj:
                    reader = ProgressFileReader(
                        fobj,
                        lambda uploaded: progress.update(task, completed=uploaded),
                    )
                    self._request(
                        "POST",
                        f"/files/file/local/{abs_rpath}",
                        files={"file_upload": (posixpath.basename(lpath), reader)},
                    )
                progress.update(task, completed=total)
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
