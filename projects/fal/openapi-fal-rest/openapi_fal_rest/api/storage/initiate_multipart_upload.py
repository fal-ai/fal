from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.initiate_multipart_upload_storage_type import InitiateMultipartUploadStorageType
from ...models.presigned_upload_url import PresignedUploadUrl
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    storage_type: Union[Unset, InitiateMultipartUploadStorageType] = InitiateMultipartUploadStorageType.FAL_CDN_V3,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_storage_type: Union[Unset, str] = UNSET
    if not isinstance(storage_type, Unset):
        json_storage_type = storage_type.value

    params["storage_type"] = json_storage_type

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/storage/upload/initiate-multipart",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, PresignedUploadUrl]]:
    if response.status_code == 200:
        response_200 = PresignedUploadUrl.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, PresignedUploadUrl]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    storage_type: Union[Unset, InitiateMultipartUploadStorageType] = InitiateMultipartUploadStorageType.FAL_CDN_V3,
) -> Response[Union[HTTPValidationError, PresignedUploadUrl]]:
    """Initiate Multipart Upload

    Args:
        storage_type (Union[Unset, InitiateMultipartUploadStorageType]):  Default:
            InitiateMultipartUploadStorageType.FAL_CDN_V3.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, PresignedUploadUrl]]
    """

    kwargs = _get_kwargs(
        storage_type=storage_type,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    storage_type: Union[Unset, InitiateMultipartUploadStorageType] = InitiateMultipartUploadStorageType.FAL_CDN_V3,
) -> Optional[Union[HTTPValidationError, PresignedUploadUrl]]:
    """Initiate Multipart Upload

    Args:
        storage_type (Union[Unset, InitiateMultipartUploadStorageType]):  Default:
            InitiateMultipartUploadStorageType.FAL_CDN_V3.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, PresignedUploadUrl]
    """

    return sync_detailed(
        client=client,
        storage_type=storage_type,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    storage_type: Union[Unset, InitiateMultipartUploadStorageType] = InitiateMultipartUploadStorageType.FAL_CDN_V3,
) -> Response[Union[HTTPValidationError, PresignedUploadUrl]]:
    """Initiate Multipart Upload

    Args:
        storage_type (Union[Unset, InitiateMultipartUploadStorageType]):  Default:
            InitiateMultipartUploadStorageType.FAL_CDN_V3.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, PresignedUploadUrl]]
    """

    kwargs = _get_kwargs(
        storage_type=storage_type,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    storage_type: Union[Unset, InitiateMultipartUploadStorageType] = InitiateMultipartUploadStorageType.FAL_CDN_V3,
) -> Optional[Union[HTTPValidationError, PresignedUploadUrl]]:
    """Initiate Multipart Upload

    Args:
        storage_type (Union[Unset, InitiateMultipartUploadStorageType]):  Default:
            InitiateMultipartUploadStorageType.FAL_CDN_V3.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, PresignedUploadUrl]
    """

    return (
        await asyncio_detailed(
            client=client,
            storage_type=storage_type,
        )
    ).parsed
