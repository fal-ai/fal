from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.namespace_alias_response import NamespaceAliasResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    active_only: Union[Unset, bool] = True,
    source_user_nickname: Union[Unset, str] = UNSET,
    target_user_nickname: Union[Unset, str] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["active_only"] = active_only

    params["source_user_nickname"] = source_user_nickname

    params["target_user_nickname"] = target_user_nickname

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/admin/namespace_aliases",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, list["NamespaceAliasResponse"]]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = NamespaceAliasResponse.from_dict(response_200_item_data)

            response_200.append(response_200_item)

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
) -> Response[Union[HTTPValidationError, list["NamespaceAliasResponse"]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    active_only: Union[Unset, bool] = True,
    source_user_nickname: Union[Unset, str] = UNSET,
    target_user_nickname: Union[Unset, str] = UNSET,
) -> Response[Union[HTTPValidationError, list["NamespaceAliasResponse"]]]:
    """List Namespace Aliases

     List namespace aliases with optional filtering

    Args:
        active_only (Union[Unset, bool]):  Default: True.
        source_user_nickname (Union[Unset, str]):
        target_user_nickname (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['NamespaceAliasResponse']]]
    """

    kwargs = _get_kwargs(
        active_only=active_only,
        source_user_nickname=source_user_nickname,
        target_user_nickname=target_user_nickname,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    active_only: Union[Unset, bool] = True,
    source_user_nickname: Union[Unset, str] = UNSET,
    target_user_nickname: Union[Unset, str] = UNSET,
) -> Optional[Union[HTTPValidationError, list["NamespaceAliasResponse"]]]:
    """List Namespace Aliases

     List namespace aliases with optional filtering

    Args:
        active_only (Union[Unset, bool]):  Default: True.
        source_user_nickname (Union[Unset, str]):
        target_user_nickname (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['NamespaceAliasResponse']]
    """

    return sync_detailed(
        client=client,
        active_only=active_only,
        source_user_nickname=source_user_nickname,
        target_user_nickname=target_user_nickname,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    active_only: Union[Unset, bool] = True,
    source_user_nickname: Union[Unset, str] = UNSET,
    target_user_nickname: Union[Unset, str] = UNSET,
) -> Response[Union[HTTPValidationError, list["NamespaceAliasResponse"]]]:
    """List Namespace Aliases

     List namespace aliases with optional filtering

    Args:
        active_only (Union[Unset, bool]):  Default: True.
        source_user_nickname (Union[Unset, str]):
        target_user_nickname (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['NamespaceAliasResponse']]]
    """

    kwargs = _get_kwargs(
        active_only=active_only,
        source_user_nickname=source_user_nickname,
        target_user_nickname=target_user_nickname,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    active_only: Union[Unset, bool] = True,
    source_user_nickname: Union[Unset, str] = UNSET,
    target_user_nickname: Union[Unset, str] = UNSET,
) -> Optional[Union[HTTPValidationError, list["NamespaceAliasResponse"]]]:
    """List Namespace Aliases

     List namespace aliases with optional filtering

    Args:
        active_only (Union[Unset, bool]):  Default: True.
        source_user_nickname (Union[Unset, str]):
        target_user_nickname (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['NamespaceAliasResponse']]
    """

    return (
        await asyncio_detailed(
            client=client,
            active_only=active_only,
            source_user_nickname=source_user_nickname,
            target_user_nickname=target_user_nickname,
        )
    ).parsed
