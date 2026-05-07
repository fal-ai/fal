from http import HTTPStatus
from typing import Any, Optional, Union, cast
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response


def _get_kwargs(
    tag_id: UUID,
    *,
    entity_type: str,
    entity_id: str,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["entity_type"] = entity_type

    params["entity_id"] = entity_id

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "delete",
        "url": f"/tags/{tag_id}/assign",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, HTTPValidationError]]:
    if response.status_code == 204:
        response_204 = cast(Any, None)
        return response_204
    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[Any, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    tag_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    entity_type: str,
    entity_id: str,
) -> Response[Union[Any, HTTPValidationError]]:
    """Unassign Tag

     Remove a tag from an entity.

    Args:
        tag_id (UUID):
        entity_type (str):
        entity_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        tag_id=tag_id,
        entity_type=entity_type,
        entity_id=entity_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    tag_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    entity_type: str,
    entity_id: str,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Unassign Tag

     Remove a tag from an entity.

    Args:
        tag_id (UUID):
        entity_type (str):
        entity_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return sync_detailed(
        tag_id=tag_id,
        client=client,
        entity_type=entity_type,
        entity_id=entity_id,
    ).parsed


async def asyncio_detailed(
    tag_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    entity_type: str,
    entity_id: str,
) -> Response[Union[Any, HTTPValidationError]]:
    """Unassign Tag

     Remove a tag from an entity.

    Args:
        tag_id (UUID):
        entity_type (str):
        entity_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        tag_id=tag_id,
        entity_type=entity_type,
        entity_id=entity_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    tag_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    entity_type: str,
    entity_id: str,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Unassign Tag

     Remove a tag from an entity.

    Args:
        tag_id (UUID):
        entity_type (str):
        entity_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            tag_id=tag_id,
            client=client,
            entity_type=entity_type,
            entity_id=entity_id,
        )
    ).parsed
