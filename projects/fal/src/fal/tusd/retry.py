"""High-level functions that do retrying in case of communication errors."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

import aiohttp
import tenacity
import yarl
from aiotus import RetryConfiguration, common, core, creation
from aiotus.retry import _make_retrying, _sanitize_metadata
from fal.tusd.cache import cache_upload, get_cached_upload

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from contextlib import AbstractAsyncContextManager
    from typing import BinaryIO


async def upload_single(  # noqa: PLR0913
    endpoint: str | yarl.URL,
    file: BinaryIO,
    metadata: common.Metadata | None = None,
    client_session: aiohttp.ClientSession | None = None,
    config: RetryConfiguration | None = None,
    headers: Mapping[str, str] | None = None,
    chunksize: int = 4 * 1024 * 1024,
) -> yarl.URL | None:
    """Upload a file to a tus server.

    This function creates an upload on the server and then uploads
    the data to that location.

    In case of a communication error, this function retries the upload.

    :param endpoint: The creation endpoint of the server.
    :param file: The file to upload.
    :param metadata: Additional metadata for the upload.
    :param client_session: An aiohttp ClientSession to use.
    :param config: Settings to customize the retry behaviour.
    :param headers: Optional headers used in the request.
    :param chunksize: The size of individual chunks to upload at a time.
    :return: The location where the file was uploaded to (if the upload succeeded).
    """
    if config is None:
        config = RetryConfiguration()

    url = yarl.URL(endpoint)

    # Compare the hash against upload cache. If exists skip creation and continue
    file_hash = file.parent_hash # type: ignore
    start_position = file.tell()
    maybe_broken_upload = await get_cached_upload(file_hash, start_position, chunksize)

    metadata = _sanitize_metadata(metadata)
    retrying_create = _make_retrying("upload creation", config)
    retrying_upload_file = _make_retrying("upload single", config)

    try:
        ctx: aiohttp.ClientSession | AbstractAsyncContextManager[aiohttp.ClientSession]
        if client_session is None:
            ctx = aiohttp.ClientSession()
        else:
            ctx = contextlib.nullcontext(client_session)

        async with ctx as session:
            if not maybe_broken_upload:
                async for attempt in retrying_create:
                    with attempt:
                        location = await creation.create(
                            session,
                            url,
                            file,
                            metadata,
                            ssl=config.ssl,
                            headers=headers,
                        )

                # Set the file url to cache
                if file_hash:
                    await cache_upload(
                        file_hash, str(location), start_position, chunksize
                    )

                if not location.is_absolute():
                    location = url / location.path
            else:
                # Use the url from the upload cache and set the location
                location = yarl.URL(maybe_broken_upload)

            async for attempt in retrying_upload_file:
                with attempt:
                    await core.upload_buffer(
                        session,
                        location,
                        file,
                        ssl=config.ssl,
                        chunksize=chunksize,
                        headers=headers,
                    )

            return location
    except tenacity.RetryError as e:
        logger.error(
            f"Unable to upload file, even after retrying: {e.last_attempt.exception()}"
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"Unable to upload file: {e}")
        raise

    return None
