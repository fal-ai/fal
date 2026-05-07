from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.app_metadata_response_app_metadata import AppMetadataResponseAppMetadata
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    app_user_id: str,
    app_alias_or_id: str,
) -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/applications/{app_user_id}/{app_alias_or_id}/metadata",
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[AppMetadataResponseAppMetadata, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = AppMetadataResponseAppMetadata.from_dict(response.json())

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
) -> Response[Union[AppMetadataResponseAppMetadata, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    app_user_id: str,
    app_alias_or_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[AppMetadataResponseAppMetadata, HTTPValidationError]]:
    """App Metadata

    Args:
        app_user_id (str):
        app_alias_or_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AppMetadataResponseAppMetadata, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        app_user_id=app_user_id,
        app_alias_or_id=app_alias_or_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    app_user_id: str,
    app_alias_or_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[AppMetadataResponseAppMetadata, HTTPValidationError]]:
    """App Metadata

    Args:
        app_user_id (str):
        app_alias_or_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AppMetadataResponseAppMetadata, HTTPValidationError]
    """

    return sync_detailed(
        app_user_id=app_user_id,
        app_alias_or_id=app_alias_or_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    app_user_id: str,
    app_alias_or_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[AppMetadataResponseAppMetadata, HTTPValidationError]]:
    """App Metadata

    Args:
        app_user_id (str):
        app_alias_or_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AppMetadataResponseAppMetadata, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        app_user_id=app_user_id,
        app_alias_or_id=app_alias_or_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    app_user_id: str,
    app_alias_or_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[AppMetadataResponseAppMetadata, HTTPValidationError]]:
    """App Metadata

    Args:
        app_user_id (str):
        app_alias_or_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AppMetadataResponseAppMetadata, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            app_user_id=app_user_id,
            app_alias_or_id=app_alias_or_id,
            client=client,
        )
    ).parsed
