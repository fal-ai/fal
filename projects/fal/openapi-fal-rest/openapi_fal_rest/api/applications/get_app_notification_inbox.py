from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.notification_inbox_page import NotificationInboxPage
from ...models.user_notification_category import UserNotificationCategory
from ...models.user_notification_subcategory import UserNotificationSubcategory
from ...types import UNSET, Response, Unset


def _get_kwargs(
    app_user_id: str,
    app_alias: str,
    *,
    category: Union[Unset, list[UserNotificationCategory]] = UNSET,
    subcategory: Union[Unset, list[UserNotificationSubcategory]] = UNSET,
    read: Union[Unset, str] = "all",
    search: Union[Unset, str] = UNSET,
    cursor: Union[Unset, str] = UNSET,
    limit: Union[Unset, int] = 50,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_category: Union[Unset, list[str]] = UNSET
    if not isinstance(category, Unset):
        json_category = []
        for category_item_data in category:
            category_item = category_item_data.value
            json_category.append(category_item)

    params["category"] = json_category

    json_subcategory: Union[Unset, list[str]] = UNSET
    if not isinstance(subcategory, Unset):
        json_subcategory = []
        for subcategory_item_data in subcategory:
            subcategory_item = subcategory_item_data.value
            json_subcategory.append(subcategory_item)

    params["subcategory"] = json_subcategory

    params["read"] = read

    params["search"] = search

    params["cursor"] = cursor

    params["limit"] = limit

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/applications/{app_user_id}/{app_alias}/notifications/inbox",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, NotificationInboxPage]]:
    if response.status_code == 200:
        response_200 = NotificationInboxPage.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, NotificationInboxPage]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    app_user_id: str,
    app_alias: str,
    *,
    client: Union[AuthenticatedClient, Client],
    category: Union[Unset, list[UserNotificationCategory]] = UNSET,
    subcategory: Union[Unset, list[UserNotificationSubcategory]] = UNSET,
    read: Union[Unset, str] = "all",
    search: Union[Unset, str] = UNSET,
    cursor: Union[Unset, str] = UNSET,
    limit: Union[Unset, int] = 50,
) -> Response[Union[HTTPValidationError, NotificationInboxPage]]:
    """Get App Notification Inbox

    Args:
        app_user_id (str):
        app_alias (str):
        category (Union[Unset, list[UserNotificationCategory]]):
        subcategory (Union[Unset, list[UserNotificationSubcategory]]):
        read (Union[Unset, str]):  Default: 'all'.
        search (Union[Unset, str]):
        cursor (Union[Unset, str]):
        limit (Union[Unset, int]):  Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, NotificationInboxPage]]
    """

    kwargs = _get_kwargs(
        app_user_id=app_user_id,
        app_alias=app_alias,
        category=category,
        subcategory=subcategory,
        read=read,
        search=search,
        cursor=cursor,
        limit=limit,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    app_user_id: str,
    app_alias: str,
    *,
    client: Union[AuthenticatedClient, Client],
    category: Union[Unset, list[UserNotificationCategory]] = UNSET,
    subcategory: Union[Unset, list[UserNotificationSubcategory]] = UNSET,
    read: Union[Unset, str] = "all",
    search: Union[Unset, str] = UNSET,
    cursor: Union[Unset, str] = UNSET,
    limit: Union[Unset, int] = 50,
) -> Optional[Union[HTTPValidationError, NotificationInboxPage]]:
    """Get App Notification Inbox

    Args:
        app_user_id (str):
        app_alias (str):
        category (Union[Unset, list[UserNotificationCategory]]):
        subcategory (Union[Unset, list[UserNotificationSubcategory]]):
        read (Union[Unset, str]):  Default: 'all'.
        search (Union[Unset, str]):
        cursor (Union[Unset, str]):
        limit (Union[Unset, int]):  Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, NotificationInboxPage]
    """

    return sync_detailed(
        app_user_id=app_user_id,
        app_alias=app_alias,
        client=client,
        category=category,
        subcategory=subcategory,
        read=read,
        search=search,
        cursor=cursor,
        limit=limit,
    ).parsed


async def asyncio_detailed(
    app_user_id: str,
    app_alias: str,
    *,
    client: Union[AuthenticatedClient, Client],
    category: Union[Unset, list[UserNotificationCategory]] = UNSET,
    subcategory: Union[Unset, list[UserNotificationSubcategory]] = UNSET,
    read: Union[Unset, str] = "all",
    search: Union[Unset, str] = UNSET,
    cursor: Union[Unset, str] = UNSET,
    limit: Union[Unset, int] = 50,
) -> Response[Union[HTTPValidationError, NotificationInboxPage]]:
    """Get App Notification Inbox

    Args:
        app_user_id (str):
        app_alias (str):
        category (Union[Unset, list[UserNotificationCategory]]):
        subcategory (Union[Unset, list[UserNotificationSubcategory]]):
        read (Union[Unset, str]):  Default: 'all'.
        search (Union[Unset, str]):
        cursor (Union[Unset, str]):
        limit (Union[Unset, int]):  Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, NotificationInboxPage]]
    """

    kwargs = _get_kwargs(
        app_user_id=app_user_id,
        app_alias=app_alias,
        category=category,
        subcategory=subcategory,
        read=read,
        search=search,
        cursor=cursor,
        limit=limit,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    app_user_id: str,
    app_alias: str,
    *,
    client: Union[AuthenticatedClient, Client],
    category: Union[Unset, list[UserNotificationCategory]] = UNSET,
    subcategory: Union[Unset, list[UserNotificationSubcategory]] = UNSET,
    read: Union[Unset, str] = "all",
    search: Union[Unset, str] = UNSET,
    cursor: Union[Unset, str] = UNSET,
    limit: Union[Unset, int] = 50,
) -> Optional[Union[HTTPValidationError, NotificationInboxPage]]:
    """Get App Notification Inbox

    Args:
        app_user_id (str):
        app_alias (str):
        category (Union[Unset, list[UserNotificationCategory]]):
        subcategory (Union[Unset, list[UserNotificationSubcategory]]):
        read (Union[Unset, str]):  Default: 'all'.
        search (Union[Unset, str]):
        cursor (Union[Unset, str]):
        limit (Union[Unset, int]):  Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, NotificationInboxPage]
    """

    return (
        await asyncio_detailed(
            app_user_id=app_user_id,
            app_alias=app_alias,
            client=client,
            category=category,
            subcategory=subcategory,
            read=read,
            search=search,
            cursor=cursor,
            limit=limit,
        )
    ).parsed
