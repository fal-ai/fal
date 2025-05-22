import os
import posixpath
from functools import cached_property
from typing import TYPE_CHECKING

from fsspec import AbstractFileSystem

if TYPE_CHECKING:
    import httpx

USER_AGENT = "fal-sdk/1.14.0 (python)"


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

    def _ls(self, path):
        response = self._client.get(f"/files/list/{path}")
        response.raise_for_status()
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
        abs_path = "/" + path.lstrip("/")
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
        parent = posixpath.dirname(path)
        entries = self.ls(parent, detail=True)
        for entry in entries:
            if entry["name"] == path:
                return entry
        raise FileNotFoundError(f"File not found: {path}")

    def get_file(self, rpath, lpath, **kwargs):
        if self.isdir(rpath):
            os.makedirs(lpath, exist_ok=True)
            return

        with open(lpath, "wb") as fobj:
            response = self._client.get(f"/files/file/{rpath.lstrip('/')}")
            response.raise_for_status()
            fobj.write(response.content)

    def put_file(self, lpath, rpath, mode="overwrite", **kwargs):
        if os.path.isdir(lpath):
            return

        with open(lpath, "rb") as fobj:
            response = self._client.post(
                f"/files/file/local/{rpath.lstrip('/')}",
                files={"file_upload": (posixpath.basename(lpath), fobj, "text/plain")},
            )
            response.raise_for_status()
        self.dircache.clear()

    def rm(self, path, **kwargs):
        response = self._client.delete(f"/files/file/{path.lstrip('/')}")
        response.raise_for_status()
        self.dircache.clear()
