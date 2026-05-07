from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.page_workflow_item import PageWorkflowItem
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    search: Union[Unset, str] = UNSET,
    user_id: Union[Unset, str] = UNSET,
    is_public: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["search"] = search

    params["user_id"] = user_id

    params["is_public"] = is_public

    params["page"] = page

    params["size"] = size

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/admin/workflows",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, PageWorkflowItem]]:
    if response.status_code == 200:
        response_200 = PageWorkflowItem.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, PageWorkflowItem]]:
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
    user_id: Union[Unset, str] = UNSET,
    is_public: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Response[Union[HTTPValidationError, PageWorkflowItem]]:
    """List Workflows

     List all users' workflows with pagination and optional search filters.

    Args:
        search (Union[Unset, str]): Search by workflow name or title
        user_id (Union[Unset, str]): Filter by user ID or nickname
        is_public (Union[Unset, bool]): Filter by public flag
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, PageWorkflowItem]]
    """

    kwargs = _get_kwargs(
        search=search,
        user_id=user_id,
        is_public=is_public,
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
    user_id: Union[Unset, str] = UNSET,
    is_public: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Optional[Union[HTTPValidationError, PageWorkflowItem]]:
    """List Workflows

     List all users' workflows with pagination and optional search filters.

    Args:
        search (Union[Unset, str]): Search by workflow name or title
        user_id (Union[Unset, str]): Filter by user ID or nickname
        is_public (Union[Unset, bool]): Filter by public flag
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, PageWorkflowItem]
    """

    return sync_detailed(
        client=client,
        search=search,
        user_id=user_id,
        is_public=is_public,
        page=page,
        size=size,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    search: Union[Unset, str] = UNSET,
    user_id: Union[Unset, str] = UNSET,
    is_public: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Response[Union[HTTPValidationError, PageWorkflowItem]]:
    """List Workflows

     List all users' workflows with pagination and optional search filters.

    Args:
        search (Union[Unset, str]): Search by workflow name or title
        user_id (Union[Unset, str]): Filter by user ID or nickname
        is_public (Union[Unset, bool]): Filter by public flag
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, PageWorkflowItem]]
    """

    kwargs = _get_kwargs(
        search=search,
        user_id=user_id,
        is_public=is_public,
        page=page,
        size=size,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    search: Union[Unset, str] = UNSET,
    user_id: Union[Unset, str] = UNSET,
    is_public: Union[Unset, bool] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Optional[Union[HTTPValidationError, PageWorkflowItem]]:
    """List Workflows

     List all users' workflows with pagination and optional search filters.

    Args:
        search (Union[Unset, str]): Search by workflow name or title
        user_id (Union[Unset, str]): Filter by user ID or nickname
        is_public (Union[Unset, bool]): Filter by public flag
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, PageWorkflowItem]
    """

    return (
        await asyncio_detailed(
            client=client,
            search=search,
            user_id=user_id,
            is_public=is_public,
            page=page,
            size=size,
        )
    ).parsed
