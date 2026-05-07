import datetime
from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.notifications_page import NotificationsPage
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    category: Union[Unset, Any] = UNSET,
    subcategory: Union[Unset, Any] = UNSET,
    created_after: Union[Unset, datetime.datetime] = UNSET,
    created_before: Union[Unset, datetime.datetime] = UNSET,
    system_only: Union[Unset, bool] = False,
    user_id: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["category"] = category

    params["subcategory"] = subcategory

    json_created_after: Union[Unset, str] = UNSET
    if not isinstance(created_after, Unset):
        json_created_after = created_after.isoformat()
    params["created_after"] = json_created_after

    json_created_before: Union[Unset, str] = UNSET
    if not isinstance(created_before, Unset):
        json_created_before = created_before.isoformat()
    params["created_before"] = json_created_before

    params["system_only"] = system_only

    params["user_id"] = user_id

    params["page"] = page

    params["size"] = size

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/admin/notifications",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, NotificationsPage]]:
    if response.status_code == 200:
        response_200 = NotificationsPage.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, NotificationsPage]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    category: Union[Unset, Any] = UNSET,
    subcategory: Union[Unset, Any] = UNSET,
    created_after: Union[Unset, datetime.datetime] = UNSET,
    created_before: Union[Unset, datetime.datetime] = UNSET,
    system_only: Union[Unset, bool] = False,
    user_id: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Response[Union[HTTPValidationError, NotificationsPage]]:
    """Admin List Notifications

    Args:
        category (Union[Unset, Any]): Filter by notification category.
        subcategory (Union[Unset, Any]): Filter by notification subcategory.
        created_after (Union[Unset, datetime.datetime]): Inclusive lower bound on created_at.
        created_before (Union[Unset, datetime.datetime]): Inclusive upper bound on created_at.
        system_only (Union[Unset, bool]): When true, return only system notifications (user_id IS
            NULL) — the broadcast rows written by `admin_create_notification` that fan out per-user on
            materialize. When false (default), return every row, including per-user notifications
            written by other producers. Default: False.
        user_id (Union[Unset, str]): Filter by user ID or nickname. Mutually exclusive with
            `system_only=true` — system notifications by definition have no user_id.
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, NotificationsPage]]
    """

    kwargs = _get_kwargs(
        category=category,
        subcategory=subcategory,
        created_after=created_after,
        created_before=created_before,
        system_only=system_only,
        user_id=user_id,
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
    category: Union[Unset, Any] = UNSET,
    subcategory: Union[Unset, Any] = UNSET,
    created_after: Union[Unset, datetime.datetime] = UNSET,
    created_before: Union[Unset, datetime.datetime] = UNSET,
    system_only: Union[Unset, bool] = False,
    user_id: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Optional[Union[HTTPValidationError, NotificationsPage]]:
    """Admin List Notifications

    Args:
        category (Union[Unset, Any]): Filter by notification category.
        subcategory (Union[Unset, Any]): Filter by notification subcategory.
        created_after (Union[Unset, datetime.datetime]): Inclusive lower bound on created_at.
        created_before (Union[Unset, datetime.datetime]): Inclusive upper bound on created_at.
        system_only (Union[Unset, bool]): When true, return only system notifications (user_id IS
            NULL) — the broadcast rows written by `admin_create_notification` that fan out per-user on
            materialize. When false (default), return every row, including per-user notifications
            written by other producers. Default: False.
        user_id (Union[Unset, str]): Filter by user ID or nickname. Mutually exclusive with
            `system_only=true` — system notifications by definition have no user_id.
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, NotificationsPage]
    """

    return sync_detailed(
        client=client,
        category=category,
        subcategory=subcategory,
        created_after=created_after,
        created_before=created_before,
        system_only=system_only,
        user_id=user_id,
        page=page,
        size=size,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    category: Union[Unset, Any] = UNSET,
    subcategory: Union[Unset, Any] = UNSET,
    created_after: Union[Unset, datetime.datetime] = UNSET,
    created_before: Union[Unset, datetime.datetime] = UNSET,
    system_only: Union[Unset, bool] = False,
    user_id: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Response[Union[HTTPValidationError, NotificationsPage]]:
    """Admin List Notifications

    Args:
        category (Union[Unset, Any]): Filter by notification category.
        subcategory (Union[Unset, Any]): Filter by notification subcategory.
        created_after (Union[Unset, datetime.datetime]): Inclusive lower bound on created_at.
        created_before (Union[Unset, datetime.datetime]): Inclusive upper bound on created_at.
        system_only (Union[Unset, bool]): When true, return only system notifications (user_id IS
            NULL) — the broadcast rows written by `admin_create_notification` that fan out per-user on
            materialize. When false (default), return every row, including per-user notifications
            written by other producers. Default: False.
        user_id (Union[Unset, str]): Filter by user ID or nickname. Mutually exclusive with
            `system_only=true` — system notifications by definition have no user_id.
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, NotificationsPage]]
    """

    kwargs = _get_kwargs(
        category=category,
        subcategory=subcategory,
        created_after=created_after,
        created_before=created_before,
        system_only=system_only,
        user_id=user_id,
        page=page,
        size=size,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    category: Union[Unset, Any] = UNSET,
    subcategory: Union[Unset, Any] = UNSET,
    created_after: Union[Unset, datetime.datetime] = UNSET,
    created_before: Union[Unset, datetime.datetime] = UNSET,
    system_only: Union[Unset, bool] = False,
    user_id: Union[Unset, str] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Optional[Union[HTTPValidationError, NotificationsPage]]:
    """Admin List Notifications

    Args:
        category (Union[Unset, Any]): Filter by notification category.
        subcategory (Union[Unset, Any]): Filter by notification subcategory.
        created_after (Union[Unset, datetime.datetime]): Inclusive lower bound on created_at.
        created_before (Union[Unset, datetime.datetime]): Inclusive upper bound on created_at.
        system_only (Union[Unset, bool]): When true, return only system notifications (user_id IS
            NULL) — the broadcast rows written by `admin_create_notification` that fan out per-user on
            materialize. When false (default), return every row, including per-user notifications
            written by other producers. Default: False.
        user_id (Union[Unset, str]): Filter by user ID or nickname. Mutually exclusive with
            `system_only=true` — system notifications by definition have no user_id.
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, NotificationsPage]
    """

    return (
        await asyncio_detailed(
            client=client,
            category=category,
            subcategory=subcategory,
            created_after=created_after,
            created_before=created_before,
            system_only=system_only,
            user_id=user_id,
            page=page,
            size=size,
        )
    ).parsed
