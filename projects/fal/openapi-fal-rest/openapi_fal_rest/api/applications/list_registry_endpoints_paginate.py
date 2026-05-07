from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.list_registry_endpoints_paginate_sort import ListRegistryEndpointsPaginateSort
from ...models.page_registry_endpoint import PageRegistryEndpoint
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    public: Union[Unset, bool] = True,
    search_str: Union[Unset, str] = "",
    categories: Union[Unset, list[str]] = UNSET,
    kinds: Union[Unset, list[str]] = UNSET,
    deprecated: Union[Unset, bool] = False,
    pinned: Union[Unset, bool] = UNSET,
    sort: Union[Unset, ListRegistryEndpointsPaginateSort] = ListRegistryEndpointsPaginateSort.RELEVANT,
    tags: Union[Unset, list[str]] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["public"] = public

    params["search_str"] = search_str

    json_categories: Union[Unset, list[str]] = UNSET
    if not isinstance(categories, Unset):
        json_categories = categories

    params["categories"] = json_categories

    json_kinds: Union[Unset, list[str]] = UNSET
    if not isinstance(kinds, Unset):
        json_kinds = kinds

    params["kinds"] = json_kinds

    params["deprecated"] = deprecated

    params["pinned"] = pinned

    json_sort: Union[Unset, str] = UNSET
    if not isinstance(sort, Unset):
        json_sort = sort.value

    params["sort"] = json_sort

    json_tags: Union[Unset, list[str]] = UNSET
    if not isinstance(tags, Unset):
        json_tags = tags

    params["tags"] = json_tags

    params["page"] = page

    params["size"] = size

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/applications/registry/paginate",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, PageRegistryEndpoint]]:
    if response.status_code == 200:
        response_200 = PageRegistryEndpoint.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, PageRegistryEndpoint]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    public: Union[Unset, bool] = True,
    search_str: Union[Unset, str] = "",
    categories: Union[Unset, list[str]] = UNSET,
    kinds: Union[Unset, list[str]] = UNSET,
    deprecated: Union[Unset, bool] = False,
    pinned: Union[Unset, bool] = UNSET,
    sort: Union[Unset, ListRegistryEndpointsPaginateSort] = ListRegistryEndpointsPaginateSort.RELEVANT,
    tags: Union[Unset, list[str]] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Response[Union[HTTPValidationError, PageRegistryEndpoint]]:
    """List Registry Endpoints Paginate

    Args:
        public (Union[Unset, bool]):  Default: True.
        search_str (Union[Unset, str]):  Default: ''.
        categories (Union[Unset, list[str]]):
        kinds (Union[Unset, list[str]]):
        deprecated (Union[Unset, bool]):  Default: False.
        pinned (Union[Unset, bool]):
        sort (Union[Unset, ListRegistryEndpointsPaginateSort]):  Default:
            ListRegistryEndpointsPaginateSort.RELEVANT.
        tags (Union[Unset, list[str]]):
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, PageRegistryEndpoint]]
    """

    kwargs = _get_kwargs(
        public=public,
        search_str=search_str,
        categories=categories,
        kinds=kinds,
        deprecated=deprecated,
        pinned=pinned,
        sort=sort,
        tags=tags,
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
    public: Union[Unset, bool] = True,
    search_str: Union[Unset, str] = "",
    categories: Union[Unset, list[str]] = UNSET,
    kinds: Union[Unset, list[str]] = UNSET,
    deprecated: Union[Unset, bool] = False,
    pinned: Union[Unset, bool] = UNSET,
    sort: Union[Unset, ListRegistryEndpointsPaginateSort] = ListRegistryEndpointsPaginateSort.RELEVANT,
    tags: Union[Unset, list[str]] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Optional[Union[HTTPValidationError, PageRegistryEndpoint]]:
    """List Registry Endpoints Paginate

    Args:
        public (Union[Unset, bool]):  Default: True.
        search_str (Union[Unset, str]):  Default: ''.
        categories (Union[Unset, list[str]]):
        kinds (Union[Unset, list[str]]):
        deprecated (Union[Unset, bool]):  Default: False.
        pinned (Union[Unset, bool]):
        sort (Union[Unset, ListRegistryEndpointsPaginateSort]):  Default:
            ListRegistryEndpointsPaginateSort.RELEVANT.
        tags (Union[Unset, list[str]]):
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, PageRegistryEndpoint]
    """

    return sync_detailed(
        client=client,
        public=public,
        search_str=search_str,
        categories=categories,
        kinds=kinds,
        deprecated=deprecated,
        pinned=pinned,
        sort=sort,
        tags=tags,
        page=page,
        size=size,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    public: Union[Unset, bool] = True,
    search_str: Union[Unset, str] = "",
    categories: Union[Unset, list[str]] = UNSET,
    kinds: Union[Unset, list[str]] = UNSET,
    deprecated: Union[Unset, bool] = False,
    pinned: Union[Unset, bool] = UNSET,
    sort: Union[Unset, ListRegistryEndpointsPaginateSort] = ListRegistryEndpointsPaginateSort.RELEVANT,
    tags: Union[Unset, list[str]] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Response[Union[HTTPValidationError, PageRegistryEndpoint]]:
    """List Registry Endpoints Paginate

    Args:
        public (Union[Unset, bool]):  Default: True.
        search_str (Union[Unset, str]):  Default: ''.
        categories (Union[Unset, list[str]]):
        kinds (Union[Unset, list[str]]):
        deprecated (Union[Unset, bool]):  Default: False.
        pinned (Union[Unset, bool]):
        sort (Union[Unset, ListRegistryEndpointsPaginateSort]):  Default:
            ListRegistryEndpointsPaginateSort.RELEVANT.
        tags (Union[Unset, list[str]]):
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, PageRegistryEndpoint]]
    """

    kwargs = _get_kwargs(
        public=public,
        search_str=search_str,
        categories=categories,
        kinds=kinds,
        deprecated=deprecated,
        pinned=pinned,
        sort=sort,
        tags=tags,
        page=page,
        size=size,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    public: Union[Unset, bool] = True,
    search_str: Union[Unset, str] = "",
    categories: Union[Unset, list[str]] = UNSET,
    kinds: Union[Unset, list[str]] = UNSET,
    deprecated: Union[Unset, bool] = False,
    pinned: Union[Unset, bool] = UNSET,
    sort: Union[Unset, ListRegistryEndpointsPaginateSort] = ListRegistryEndpointsPaginateSort.RELEVANT,
    tags: Union[Unset, list[str]] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Optional[Union[HTTPValidationError, PageRegistryEndpoint]]:
    """List Registry Endpoints Paginate

    Args:
        public (Union[Unset, bool]):  Default: True.
        search_str (Union[Unset, str]):  Default: ''.
        categories (Union[Unset, list[str]]):
        kinds (Union[Unset, list[str]]):
        deprecated (Union[Unset, bool]):  Default: False.
        pinned (Union[Unset, bool]):
        sort (Union[Unset, ListRegistryEndpointsPaginateSort]):  Default:
            ListRegistryEndpointsPaginateSort.RELEVANT.
        tags (Union[Unset, list[str]]):
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, PageRegistryEndpoint]
    """

    return (
        await asyncio_detailed(
            client=client,
            public=public,
            search_str=search_str,
            categories=categories,
            kinds=kinds,
            deprecated=deprecated,
            pinned=pinned,
            sort=sort,
            tags=tags,
            page=page,
            size=size,
        )
    ).parsed
