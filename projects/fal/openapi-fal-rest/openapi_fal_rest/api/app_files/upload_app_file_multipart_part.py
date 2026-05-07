from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.body_upload_app_file_multipart_part import BodyUploadAppFileMultipartPart
from ...models.http_validation_error import HTTPValidationError
from ...models.upload_multipart_part_response import UploadMultipartPartResponse
from ...types import Response


def _get_kwargs(
    hash_: str,
    upload_id: str,
    part_number: int,
    *,
    body: BodyUploadAppFileMultipartPart,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": f"/files/app/multipart/{hash_}/{upload_id}/{part_number}",
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
    hash_: str,
    upload_id: str,
    part_number: int,
    *,
    client: Union[AuthenticatedClient, Client],
    body: BodyUploadAppFileMultipartPart,
) -> Response[Union[HTTPValidationError, UploadMultipartPartResponse]]:
    """Upload App File Multipart Part

    Args:
        hash_ (str):
        upload_id (str):
        part_number (int):
        body (BodyUploadAppFileMultipartPart):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, UploadMultipartPartResponse]]
    """

    kwargs = _get_kwargs(
        hash_=hash_,
        upload_id=upload_id,
        part_number=part_number,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    hash_: str,
    upload_id: str,
    part_number: int,
    *,
    client: Union[AuthenticatedClient, Client],
    body: BodyUploadAppFileMultipartPart,
) -> Optional[Union[HTTPValidationError, UploadMultipartPartResponse]]:
    """Upload App File Multipart Part

    Args:
        hash_ (str):
        upload_id (str):
        part_number (int):
        body (BodyUploadAppFileMultipartPart):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, UploadMultipartPartResponse]
    """

    return sync_detailed(
        hash_=hash_,
        upload_id=upload_id,
        part_number=part_number,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    hash_: str,
    upload_id: str,
    part_number: int,
    *,
    client: Union[AuthenticatedClient, Client],
    body: BodyUploadAppFileMultipartPart,
) -> Response[Union[HTTPValidationError, UploadMultipartPartResponse]]:
    """Upload App File Multipart Part

    Args:
        hash_ (str):
        upload_id (str):
        part_number (int):
        body (BodyUploadAppFileMultipartPart):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, UploadMultipartPartResponse]]
    """

    kwargs = _get_kwargs(
        hash_=hash_,
        upload_id=upload_id,
        part_number=part_number,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    hash_: str,
    upload_id: str,
    part_number: int,
    *,
    client: Union[AuthenticatedClient, Client],
    body: BodyUploadAppFileMultipartPart,
) -> Optional[Union[HTTPValidationError, UploadMultipartPartResponse]]:
    """Upload App File Multipart Part

    Args:
        hash_ (str):
        upload_id (str):
        part_number (int):
        body (BodyUploadAppFileMultipartPart):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, UploadMultipartPartResponse]
    """

    return (
        await asyncio_detailed(
            hash_=hash_,
            upload_id=upload_id,
            part_number=part_number,
            client=client,
            body=body,
        )
    ).parsed
