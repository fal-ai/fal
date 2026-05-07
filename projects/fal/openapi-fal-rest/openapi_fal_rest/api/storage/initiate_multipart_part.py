from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.initiate_multipart_part_response import InitiateMultipartPartResponse
from ...types import Response


def _get_kwargs(
    key: str,
    upload_id: str,
    part_number: int,
) -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/storage/files/{key}/multipart/{upload_id}/{part_number}",
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, InitiateMultipartPartResponse]]:
    if response.status_code == 200:
        response_200 = InitiateMultipartPartResponse.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, InitiateMultipartPartResponse]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    key: str,
    upload_id: str,
    part_number: int,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[HTTPValidationError, InitiateMultipartPartResponse]]:
    """Initiate Multipart Part

    Args:
        key (str):
        upload_id (str):
        part_number (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, InitiateMultipartPartResponse]]
    """

    kwargs = _get_kwargs(
        key=key,
        upload_id=upload_id,
        part_number=part_number,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    key: str,
    upload_id: str,
    part_number: int,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[HTTPValidationError, InitiateMultipartPartResponse]]:
    """Initiate Multipart Part

    Args:
        key (str):
        upload_id (str):
        part_number (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, InitiateMultipartPartResponse]
    """

    return sync_detailed(
        key=key,
        upload_id=upload_id,
        part_number=part_number,
        client=client,
    ).parsed


async def asyncio_detailed(
    key: str,
    upload_id: str,
    part_number: int,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[HTTPValidationError, InitiateMultipartPartResponse]]:
    """Initiate Multipart Part

    Args:
        key (str):
        upload_id (str):
        part_number (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, InitiateMultipartPartResponse]]
    """

    kwargs = _get_kwargs(
        key=key,
        upload_id=upload_id,
        part_number=part_number,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    key: str,
    upload_id: str,
    part_number: int,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[HTTPValidationError, InitiateMultipartPartResponse]]:
    """Initiate Multipart Part

    Args:
        key (str):
        upload_id (str):
        part_number (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, InitiateMultipartPartResponse]
    """

    return (
        await asyncio_detailed(
            key=key,
            upload_id=upload_id,
            part_number=part_number,
            client=client,
        )
    ).parsed
