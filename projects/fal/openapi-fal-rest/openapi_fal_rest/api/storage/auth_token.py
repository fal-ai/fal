from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.auth_token_request import AuthTokenRequest
from ...models.auth_token_response import AuthTokenResponse
from ...models.auth_token_storage_type import AuthTokenStorageType
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: AuthTokenRequest,
    storage_type: Union[Unset, AuthTokenStorageType] = AuthTokenStorageType.FAL_CDN,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    params: dict[str, Any] = {}

    json_storage_type: Union[Unset, str] = UNSET
    if not isinstance(storage_type, Unset):
        json_storage_type = storage_type.value

    params["storage_type"] = json_storage_type

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/storage/auth/token",
        "params": params,
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[AuthTokenResponse, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = AuthTokenResponse.from_dict(response.json())

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
) -> Response[Union[AuthTokenResponse, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: AuthTokenRequest,
    storage_type: Union[Unset, AuthTokenStorageType] = AuthTokenStorageType.FAL_CDN,
) -> Response[Union[AuthTokenResponse, HTTPValidationError]]:
    """Auth Token

    Args:
        storage_type (Union[Unset, AuthTokenStorageType]):  Default: AuthTokenStorageType.FAL_CDN.
        body (AuthTokenRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AuthTokenResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
        storage_type=storage_type,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    body: AuthTokenRequest,
    storage_type: Union[Unset, AuthTokenStorageType] = AuthTokenStorageType.FAL_CDN,
) -> Optional[Union[AuthTokenResponse, HTTPValidationError]]:
    """Auth Token

    Args:
        storage_type (Union[Unset, AuthTokenStorageType]):  Default: AuthTokenStorageType.FAL_CDN.
        body (AuthTokenRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AuthTokenResponse, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        body=body,
        storage_type=storage_type,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: AuthTokenRequest,
    storage_type: Union[Unset, AuthTokenStorageType] = AuthTokenStorageType.FAL_CDN,
) -> Response[Union[AuthTokenResponse, HTTPValidationError]]:
    """Auth Token

    Args:
        storage_type (Union[Unset, AuthTokenStorageType]):  Default: AuthTokenStorageType.FAL_CDN.
        body (AuthTokenRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AuthTokenResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
        storage_type=storage_type,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: AuthTokenRequest,
    storage_type: Union[Unset, AuthTokenStorageType] = AuthTokenStorageType.FAL_CDN,
) -> Optional[Union[AuthTokenResponse, HTTPValidationError]]:
    """Auth Token

    Args:
        storage_type (Union[Unset, AuthTokenStorageType]):  Default: AuthTokenStorageType.FAL_CDN.
        body (AuthTokenRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AuthTokenResponse, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            storage_type=storage_type,
        )
    ).parsed
