from http import HTTPStatus
from typing import Any, Optional, Union, cast
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    entity_type: str,
    tag_ids: Union[Unset, list[UUID]] = UNSET,
    mode: Union[Unset, str] = "any",
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["entity_type"] = entity_type

    json_tag_ids: Union[Unset, list[str]] = UNSET
    if not isinstance(tag_ids, Unset):
        json_tag_ids = []
        for tag_ids_item_data in tag_ids:
            tag_ids_item = str(tag_ids_item_data)
            json_tag_ids.append(tag_ids_item)

    params["tag_ids"] = json_tag_ids

    params["mode"] = mode

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/tags/assignments/entities",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, list[str]]]:
    if response.status_code == 200:
        response_200 = cast(list[str], response.json())

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
) -> Response[Union[HTTPValidationError, list[str]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    entity_type: str,
    tag_ids: Union[Unset, list[UUID]] = UNSET,
    mode: Union[Unset, str] = "any",
) -> Response[Union[HTTPValidationError, list[str]]]:
    """Get Entity Ids For Tags

     Return entity IDs that match the given tags.

    mode=any (OR): entity has at least one of the tags.
    mode=all (AND): entity has every one of the tags.

    Args:
        entity_type (str):
        tag_ids (Union[Unset, list[UUID]]):
        mode (Union[Unset, str]):  Default: 'any'.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list[str]]]
    """

    kwargs = _get_kwargs(
        entity_type=entity_type,
        tag_ids=tag_ids,
        mode=mode,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    entity_type: str,
    tag_ids: Union[Unset, list[UUID]] = UNSET,
    mode: Union[Unset, str] = "any",
) -> Optional[Union[HTTPValidationError, list[str]]]:
    """Get Entity Ids For Tags

     Return entity IDs that match the given tags.

    mode=any (OR): entity has at least one of the tags.
    mode=all (AND): entity has every one of the tags.

    Args:
        entity_type (str):
        tag_ids (Union[Unset, list[UUID]]):
        mode (Union[Unset, str]):  Default: 'any'.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list[str]]
    """

    return sync_detailed(
        client=client,
        entity_type=entity_type,
        tag_ids=tag_ids,
        mode=mode,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    entity_type: str,
    tag_ids: Union[Unset, list[UUID]] = UNSET,
    mode: Union[Unset, str] = "any",
) -> Response[Union[HTTPValidationError, list[str]]]:
    """Get Entity Ids For Tags

     Return entity IDs that match the given tags.

    mode=any (OR): entity has at least one of the tags.
    mode=all (AND): entity has every one of the tags.

    Args:
        entity_type (str):
        tag_ids (Union[Unset, list[UUID]]):
        mode (Union[Unset, str]):  Default: 'any'.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list[str]]]
    """

    kwargs = _get_kwargs(
        entity_type=entity_type,
        tag_ids=tag_ids,
        mode=mode,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    entity_type: str,
    tag_ids: Union[Unset, list[UUID]] = UNSET,
    mode: Union[Unset, str] = "any",
) -> Optional[Union[HTTPValidationError, list[str]]]:
    """Get Entity Ids For Tags

     Return entity IDs that match the given tags.

    mode=any (OR): entity has at least one of the tags.
    mode=all (AND): entity has every one of the tags.

    Args:
        entity_type (str):
        tag_ids (Union[Unset, list[UUID]]):
        mode (Union[Unset, str]):  Default: 'any'.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list[str]]
    """

    return (
        await asyncio_detailed(
            client=client,
            entity_type=entity_type,
            tag_ids=tag_ids,
            mode=mode,
        )
    ).parsed
