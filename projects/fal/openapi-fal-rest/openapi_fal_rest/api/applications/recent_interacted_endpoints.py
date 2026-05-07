from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.endpoint_access import EndpointAccess
from ...models.http_validation_error import HTTPValidationError
from ...models.page_endpoint_info import PageEndpointInfo
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    search: Union[Unset, str] = UNSET,
    access: Union[Unset, EndpointAccess] = UNSET,
    is_own: Union[Unset, bool] = False,
    is_use: Union[Unset, bool] = False,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["search"] = search

    json_access: Union[Unset, str] = UNSET
    if not isinstance(access, Unset):
        json_access = access.value

    params["access"] = json_access

    params["is_own"] = is_own

    params["is_use"] = is_use

    params["page"] = page

    params["size"] = size

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/applications/endpoints/recent",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, PageEndpointInfo]]:
    if response.status_code == 200:
        response_200 = PageEndpointInfo.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, PageEndpointInfo]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    search: Union[Unset, str] = UNSET,
    access: Union[Unset, EndpointAccess] = UNSET,
    is_own: Union[Unset, bool] = False,
    is_use: Union[Unset, bool] = False,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Response[Union[HTTPValidationError, PageEndpointInfo]]:
    """Recent Interacted Endpoints

    Args:
        search (Union[Unset, str]):
        access (Union[Unset, EndpointAccess]):
        is_own (Union[Unset, bool]):  Default: False.
        is_use (Union[Unset, bool]):  Default: False.
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, PageEndpointInfo]]
    """

    kwargs = _get_kwargs(
        search=search,
        access=access,
        is_own=is_own,
        is_use=is_use,
        page=page,
        size=size,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    search: Union[Unset, str] = UNSET,
    access: Union[Unset, EndpointAccess] = UNSET,
    is_own: Union[Unset, bool] = False,
    is_use: Union[Unset, bool] = False,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Optional[Union[HTTPValidationError, PageEndpointInfo]]:
    """Recent Interacted Endpoints

    Args:
        search (Union[Unset, str]):
        access (Union[Unset, EndpointAccess]):
        is_own (Union[Unset, bool]):  Default: False.
        is_use (Union[Unset, bool]):  Default: False.
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, PageEndpointInfo]
    """

    return sync_detailed(
        client=client,
        search=search,
        access=access,
        is_own=is_own,
        is_use=is_use,
        page=page,
        size=size,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    search: Union[Unset, str] = UNSET,
    access: Union[Unset, EndpointAccess] = UNSET,
    is_own: Union[Unset, bool] = False,
    is_use: Union[Unset, bool] = False,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Response[Union[HTTPValidationError, PageEndpointInfo]]:
    """Recent Interacted Endpoints

    Args:
        search (Union[Unset, str]):
        access (Union[Unset, EndpointAccess]):
        is_own (Union[Unset, bool]):  Default: False.
        is_use (Union[Unset, bool]):  Default: False.
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, PageEndpointInfo]]
    """

    kwargs = _get_kwargs(
        search=search,
        access=access,
        is_own=is_own,
        is_use=is_use,
        page=page,
        size=size,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    search: Union[Unset, str] = UNSET,
    access: Union[Unset, EndpointAccess] = UNSET,
    is_own: Union[Unset, bool] = False,
    is_use: Union[Unset, bool] = False,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Optional[Union[HTTPValidationError, PageEndpointInfo]]:
    """Recent Interacted Endpoints

    Args:
        search (Union[Unset, str]):
        access (Union[Unset, EndpointAccess]):
        is_own (Union[Unset, bool]):  Default: False.
        is_use (Union[Unset, bool]):  Default: False.
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, PageEndpointInfo]
    """

    return (
        await asyncio_detailed(
            client=client,
            search=search,
            access=access,
            is_own=is_own,
            is_use=is_use,
            page=page,
            size=size,
        )
    ).parsed
