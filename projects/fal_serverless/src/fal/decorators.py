from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

import openapi_fal_rest.api.files.file_exists as file_exists_api
import openapi_fal_rest.models.file_spec as file_spec_model
from fal.rest_client import REST_CLIENT
from fal.toolkit import mainify

if sys.version_info >= (3, 11):
    from typing import Concatenate
else:
    from typing_extensions import Concatenate

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

import fal.flags as flags
from fal.api import FalServerlessError, InternalFalServerlessError, function

ArgsT = ParamSpec("ArgsT")
ReturnT = TypeVar("ReturnT")


def file_exists(
    file_path: Path | str, calculate_checksum: bool
) -> file_spec_model.FileSpec | None:
    file_str = str(file_path)
    response = file_exists_api.sync_detailed(
        file_str,
        client=REST_CLIENT,
        calculate_checksum=calculate_checksum,
    )

    if response.status_code == 200:
        return response.parsed  # type: ignore
    elif response.status_code == 404:
        return None
    else:
        raise InternalFalServerlessError(
            f"Failed to check if file exists: {response.status_code} {response.parsed}"
        )


@mainify
@dataclass
class FileSpec:
    path: Path
    size: int
    is_file: bool
    checksum_sha256: str | None = None
    checksum_md5: str | None = None

    @classmethod
    def from_model(cls, file_spec: file_spec_model.FileSpec) -> FileSpec:
        return cls(
            path=Path(file_spec.path),
            size=file_spec.size,
            is_file=file_spec.is_file,
            checksum_sha256=file_spec.checksum_sha256
            if file_spec.checksum_sha256
            else None,
            checksum_md5=file_spec.checksum_md5 if file_spec.checksum_md5 else None,
        )


def setup(
    file_path: Path | str,
    checksum_sha256: str | None = None,
    checksum_md5: str | None = None,
    force: bool = False,
    *,
    _func_name: str | None = None,
    **isolated_config: Any,
):
    file_path = Path(file_path)
    check_path = file_path.relative_to("/data")

    def wrapper(
        func: Callable[Concatenate[ArgsT], ReturnT]
    ) -> Callable[Concatenate[ArgsT], Path]:
        name = _func_name or func.__name__

        @wraps(func)
        def internal_wrapper(
            *args: ArgsT.args,
            **kwargs: ArgsT.kwargs,
        ) -> Path:
            checksum = bool(checksum_sha256 or checksum_md5)

            file = file_exists(check_path, calculate_checksum=checksum)

            if not file or force or flags.FORCE_SETUP:
                config = {
                    "machine_type": "S",
                    **isolated_config,
                }
                function(**config)(func)(*args, **kwargs)  # type: ignore

                file = file_exists(check_path, calculate_checksum=checksum)

            if not file:
                raise FalServerlessError(
                    f"Setup function {name} did not create expected path."
                )

            if checksum:
                if not file.is_file:
                    raise FalServerlessError(
                        f"Setup function {name} created a directory instead of a file."
                    )

                if checksum_sha256 and file.checksum_sha256 != checksum_sha256:
                    raise FalServerlessError(
                        f"Setup function {name} created file with unexpected SHA256 checksum.\n"
                        f"Expected {checksum_sha256} but got {file.checksum_sha256}"
                    )

                if checksum_md5 and file.checksum_md5 != checksum_md5:
                    raise FalServerlessError(
                        f"Setup function {name} created file with unexpected MD5 checksum.\n"
                        f"Expected {checksum_md5} but got {file.checksum_md5}"
                    )

            return Path(file.path)

        return internal_wrapper

    return wrapper


DownloadType = Literal["python", "wget", "curl"]


def download_file(
    url: str,
    target_location: str | Path,
    *,
    checksum_sha256: str | None = None,
    checksum_md5: str | None = None,
    force: bool = False,
    tool: DownloadType = "python",
    _func_name: str = "download_file",
):
    target_path = Path(target_location)

    @setup(
        target_path,
        checksum_sha256,
        checksum_md5,
        force=force,
        # isolated configs
        requirements=["urllib3"],
        _func_name=_func_name,
    )
    def download(force: bool):
        import os
        from shutil import copyfileobj
        from urllib.request import Request, urlopen

        print(f"Downloading {url} to {target_path}")

        # Make sure the directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # TODO: how can we randomize the user agent to avoid being blocked?
            user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0"

            req = Request(url, headers={"User-Agent": user_agent})

            if target_path.exists() and not force:
                print(f"File {target_path} already exists, skipping download")
                return

            if tool == "python":
                with urlopen(req) as response, target_path.open("wb") as f:
                    copyfileobj(response, f)
            elif tool == "curl":
                command = f"curl {req.full_url} -o {target_path}"

                print(command)
                res = os.system(command)

                if res != 0:
                    raise Exception(f"curl failed with exit code {res}")
            elif tool == "wget":
                command = f"wget {req.full_url} -O {target_path}"

                print(command)
                res = os.system(command)

                if res != 0:
                    raise Exception(f"wget failed with exit code {res}")
            else:
                raise Exception(f"Unknown download tool {tool}")

        except Exception as e:
            msg = f"Failed to download {url} to {target_path}\n{e}"
            print(msg)

            os.system(f"rm -rf {target_path}")

            # raise exception in generally-available class
            raise FileNotFoundError(msg) from None

    return download(force)


def download_weights(
    url: str,
    *,
    checksum_sha256: str | None = None,
    checksum_md5: str | None = None,
    force: bool = False,
    tool: DownloadType = "python",
):
    import hashlib

    url_id = hashlib.sha256(url.encode("utf-8")).hexdigest()
    # This is not a protected path, so the user may change stuff internally
    url_path = Path(f"/data/.fal/downloads/{url_id}")

    return download_file(
        url,
        url_path,
        checksum_sha256=checksum_sha256,
        checksum_md5=checksum_md5,
        force=force,
        tool=tool,
        _func_name="download_weights",
    )
