from http import HTTPStatus
from typing import Any, Dict, Optional, Union

import httpx

from ... import errors
from ...client import Client
from ...models.http_validation_error import HTTPValidationError
from ...models.uploaded_file_result import UploadedFileResult
from ...types import Response


def _get_kwargs(
    full_path: str,
    *,
    client: Client,
) -> Dict[str, Any]:
    url = "{}/storage/link/{full_path}".format(client.base_url, full_path=full_path)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "follow_redirects": client.follow_redirects,
    }


def _parse_response(
    *, client: Client, response: httpx.Response
) -> Optional[Union[HTTPValidationError, UploadedFileResult]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = UploadedFileResult.from_dict(response.json())

        return response_200
    if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Client, response: httpx.Response
) -> Response[Union[HTTPValidationError, UploadedFileResult]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    full_path: str,
    *,
    client: Client,
) -> Response[Union[HTTPValidationError, UploadedFileResult]]:
    """Get Signed Link From Shared Storage

     Sign a link to a file in the shared storage bucket.
    These can be from the upload_to_shared_storage endpoint
    Or from a model upload.

    Examples:
    - /{PROJECT_ID}_toolkit_bucket/github_2745502/fal_ai_sdxl_1690489104504.png
    - /fal_file_storage/d0bdd5b7c2c6495cb4d763547bdd8fde.png

    Args:
        full_path (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, UploadedFileResult]]
    """

    kwargs = _get_kwargs(
        full_path=full_path,
        client=client,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    full_path: str,
    *,
    client: Client,
) -> Optional[Union[HTTPValidationError, UploadedFileResult]]:
    """Get Signed Link From Shared Storage

     Sign a link to a file in the shared storage bucket.
    These can be from the upload_to_shared_storage endpoint
    Or from a model upload.

    Examples:
    - /{PROJECT_ID}_toolkit_bucket/github_2745502/fal_ai_sdxl_1690489104504.png
    - /fal_file_storage/d0bdd5b7c2c6495cb4d763547bdd8fde.png

    Args:
        full_path (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, UploadedFileResult]
    """

    return sync_detailed(
        full_path=full_path,
        client=client,
    ).parsed


async def asyncio_detailed(
    full_path: str,
    *,
    client: Client,
) -> Response[Union[HTTPValidationError, UploadedFileResult]]:
    """Get Signed Link From Shared Storage

     Sign a link to a file in the shared storage bucket.
    These can be from the upload_to_shared_storage endpoint
    Or from a model upload.

    Examples:
    - /{PROJECT_ID}_toolkit_bucket/github_2745502/fal_ai_sdxl_1690489104504.png
    - /fal_file_storage/d0bdd5b7c2c6495cb4d763547bdd8fde.png

    Args:
        full_path (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, UploadedFileResult]]
    """

    kwargs = _get_kwargs(
        full_path=full_path,
        client=client,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    full_path: str,
    *,
    client: Client,
) -> Optional[Union[HTTPValidationError, UploadedFileResult]]:
    """Get Signed Link From Shared Storage

     Sign a link to a file in the shared storage bucket.
    These can be from the upload_to_shared_storage endpoint
    Or from a model upload.

    Examples:
    - /{PROJECT_ID}_toolkit_bucket/github_2745502/fal_ai_sdxl_1690489104504.png
    - /fal_file_storage/d0bdd5b7c2c6495cb4d763547bdd8fde.png

    Args:
        full_path (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, UploadedFileResult]
    """

    return (
        await asyncio_detailed(
            full_path=full_path,
            client=client,
        )
    ).parsed
