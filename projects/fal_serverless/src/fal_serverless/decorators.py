from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

import fast_api_client.api.files.file_exists as file_exists_api
import fast_api_client.models.file_spec as file_spec_model
from fal_serverless.rest_client import REST_CLIENT

if sys.version_info >= (3, 11):
    from typing import Concatenate
else:
    from typing_extensions import Concatenate

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

import fal_serverless.flags as flags
from fal_serverless.api import FalServerlessError, InternalFalServerlessError, isolated

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
    **isolated_config: Any,
):
    def wrapper(
        func: Callable[Concatenate[ArgsT], ReturnT]
    ) -> Callable[Concatenate[ArgsT], FileSpec]:
        @wraps(func)
        def internal_wrapper(
            *args: ArgsT.args,
            **kwargs: ArgsT.kwargs,
        ) -> FileSpec:
            checksum = bool(checksum_sha256 or checksum_md5)

            file = file_exists(file_path, calculate_checksum=checksum)

            if not file or force or flags.FORCE_SETUP:
                config = {
                    "machine_type": "S",
                    **isolated_config,
                }
                isolated(**config)(func)(*args, **kwargs)  # type: ignore

                file = file_exists(file_path, calculate_checksum=checksum)

            if not file:
                raise FalServerlessError(
                    f"Setup function {func.__name__} did not create expected path."
                )

            if checksum:
                if not file.is_file:
                    raise FalServerlessError(
                        f"Setup function {func.__name__} created a directory instead of a file."
                    )

                if checksum_sha256 and file.checksum_sha256 != checksum_sha256:
                    raise FalServerlessError(
                        f"Setup function {func.__name__} created file with unexpected SHA256 checksum.\n"
                        f"Expected {checksum_sha256} but got {file.checksum_sha256}"
                    )

                if checksum_md5 and file.checksum_md5 != checksum_md5:
                    raise FalServerlessError(
                        f"Setup function {func.__name__} created file with unexpected MD5 checksum.\n"
                        f"Expected {checksum_md5} but got {file.checksum_md5}"
                    )

            return FileSpec.from_model(file)

        return internal_wrapper

    return wrapper


def download_file(
    url: str,
    check_location: str | Path,
    checksum_sha256: str | None = None,
    checksum_md5: str | None = None,
    force: bool = False,
) -> FileSpec:
    check_path = Path(check_location)

    @setup(
        check_path,
        checksum_sha256,
        checksum_md5,
        force=force,
        # isolated configs
        requirements=["urllib3"],
    )
    def download():
        from urllib.request import urlretrieve

        print(f"Downloading {url} to {check_path}")

        # Make sure the directory exists
        check_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            urlretrieve(url, check_path)
        except Exception as e:
            # raise exception in generally-available class
            raise FileNotFoundError(
                f"Failed to download {url} to {check_path}\n{e}"
            ) from None

    return download()


def download_weights(
    url: str,
    checksum_sha256: str | None = None,
    checksum_md5: str | None = None,
    force: bool = False,
) -> FileSpec:
    import hashlib

    url_id = hashlib.sha256(url.encode("utf-8")).hexdigest()
    # This is not a protected path, so the user may change stuff internally
    url_path = Path(f"/data/.fal/downloads/{url_id}")

    return download_file(url, url_path, checksum_sha256, checksum_md5, force)
