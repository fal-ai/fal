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

    def ls(self, path, detail=True, **kwargs):
        response = self._client.get(f"/files/list/{path.lstrip('/')}")
        response.raise_for_status()
        files = response.json()
        if detail:
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
        else:
            return sorted(entry["path"] for entry in files)

    def info(self, path, **kwargs):
        parent = posixpath.dirname(path)
        entries = self.ls(parent, detail=True)
        for entry in entries:
            if entry["name"] == path:
                return entry
        raise FileNotFoundError(f"File not found: {path}")

    def get_file(self, rpath, lpath, **kwargs):
        with open(lpath, "wb") as fobj:
            response = self._client.get(f"/files/file/{rpath.lstrip('/')}")
            response.raise_for_status()
            fobj.write(response.content)

    def put_file(self, lpath, rpath, mode="overwrite", **kwargs):
        with open(lpath, "rb") as fobj:
            response = self._client.post(
                f"/files/file/local/{rpath.lstrip('/')}",
                files={"file_upload": (posixpath.basename(lpath), fobj, "text/plain")},
            )
            response.raise_for_status()

    def rm(self, path, **kwargs):
        response = self._client.delete(f"/files/file/{path.lstrip('/')}")
        response.raise_for_status()
