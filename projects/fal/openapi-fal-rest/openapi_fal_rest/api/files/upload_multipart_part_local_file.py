from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.body_upload_multipart_part_local_file import BodyUploadMultipartPartLocalFile
from ...models.http_validation_error import HTTPValidationError
from ...models.upload_multipart_part_response import UploadMultipartPartResponse
from ...types import Response


def _get_kwargs(
    target_path: str,
    upload_id: str,
    part_number: int,
    *,
    body: BodyUploadMultipartPartLocalFile,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": f"/files/file/multipart/{target_path}/{upload_id}/{part_number}",
    }

    _body = body.to_multipart()

    _kwargs["files"] = _body

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, UploadMultipartPartResponse]]:
    if response.status_code == 200:
        response_200 = UploadMultipartPartResponse.from_dict(response.json())

        return response_200
    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[HTTPValidationError, UploadMultipartPartResponse]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    target_path: str,
    upload_id: str,
    part_number: int,
    *,
    client: Union[AuthenticatedClient, Client],
    body: BodyUploadMultipartPartLocalFile,
) -> Response[Union[HTTPValidationError, UploadMultipartPartResponse]]:
    """Upload Multipart Part Local File

    Args:
        target_path (str):
        upload_id (str):
        part_number (int):
        body (BodyUploadMultipartPartLocalFile):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, UploadMultipartPartResponse]]
    """

    kwargs = _get_kwargs(
        target_path=target_path,
        upload_id=upload_id,
        part_number=part_number,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    target_path: str,
    upload_id: str,
    part_number: int,
    *,
    client: Union[AuthenticatedClient, Client],
    body: BodyUploadMultipartPartLocalFile,
) -> Optional[Union[HTTPValidationError, UploadMultipartPartResponse]]:
    """Upload Multipart Part Local File

    Args:
        target_path (str):
        upload_id (str):
        part_number (int):
        body (BodyUploadMultipartPartLocalFile):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, UploadMultipartPartResponse]
    """

    return sync_detailed(
        target_path=target_path,
        upload_id=upload_id,
        part_number=part_number,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    target_path: str,
    upload_id: str,
    part_number: int,
    *,
    client: Union[AuthenticatedClient, Client],
    body: BodyUploadMultipartPartLocalFile,
) -> Response[Union[HTTPValidationError, UploadMultipartPartResponse]]:
    """Upload Multipart Part Local File

    Args:
        target_path (str):
        upload_id (str):
        part_number (int):
        body (BodyUploadMultipartPartLocalFile):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, UploadMultipartPartResponse]]
    """

    kwargs = _get_kwargs(
        target_path=target_path,
        upload_id=upload_id,
        part_number=part_number,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    target_path: str,
    upload_id: str,
    part_number: int,
    *,
    client: Union[AuthenticatedClient, Client],
    body: BodyUploadMultipartPartLocalFile,
) -> Optional[Union[HTTPValidationError, UploadMultipartPartResponse]]:
    """Upload Multipart Part Local File

    Args:
        target_path (str):
        upload_id (str):
        part_number (int):
        body (BodyUploadMultipartPartLocalFile):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, UploadMultipartPartResponse]
    """

    return (
        await asyncio_detailed(
            target_path=target_path,
            upload_id=upload_id,
            part_number=part_number,
            client=client,
            body=body,
        )
    ).parsed
