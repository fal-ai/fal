from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.notification_inbox_unread_status import NotificationInboxUnreadStatus
from ...models.user_notification_category import UserNotificationCategory
from ...types import UNSET, Response, Unset


def _get_kwargs(
    user_id: str,
    *,
    category: Union[Unset, list[UserNotificationCategory]] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_category: Union[Unset, list[str]] = UNSET
    if not isinstance(category, Unset):
        json_category = []
        for category_item_data in category:
            category_item = category_item_data.value
            json_category.append(category_item)

    params["category"] = json_category

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/admin/users/{user_id}/notifications/inbox/has-unread",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, NotificationInboxUnreadStatus]]:
    if response.status_code == 200:
        response_200 = NotificationInboxUnreadStatus.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, NotificationInboxUnreadStatus]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    user_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    category: Union[Unset, list[UserNotificationCategory]] = UNSET,
) -> Response[Union[HTTPValidationError, NotificationInboxUnreadStatus]]:
    """Admin Get User Notification Inbox Has Unread

    Args:
        user_id (str):
        category (Union[Unset, list[UserNotificationCategory]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, NotificationInboxUnreadStatus]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        category=category,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    user_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    category: Union[Unset, list[UserNotificationCategory]] = UNSET,
) -> Optional[Union[HTTPValidationError, NotificationInboxUnreadStatus]]:
    """Admin Get User Notification Inbox Has Unread

    Args:
        user_id (str):
        category (Union[Unset, list[UserNotificationCategory]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, NotificationInboxUnreadStatus]
    """

    return sync_detailed(
        user_id=user_id,
        client=client,
        category=category,
    ).parsed


async def asyncio_detailed(
    user_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    category: Union[Unset, list[UserNotificationCategory]] = UNSET,
) -> Response[Union[HTTPValidationError, NotificationInboxUnreadStatus]]:
    """Admin Get User Notification Inbox Has Unread

    Args:
        user_id (str):
        category (Union[Unset, list[UserNotificationCategory]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, NotificationInboxUnreadStatus]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        category=category,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    category: Union[Unset, list[UserNotificationCategory]] = UNSET,
) -> Optional[Union[HTTPValidationError, NotificationInboxUnreadStatus]]:
    """Admin Get User Notification Inbox Has Unread

    Args:
        user_id (str):
        category (Union[Unset, list[UserNotificationCategory]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, NotificationInboxUnreadStatus]
    """

    return (
        await asyncio_detailed(
            user_id=user_id,
            client=client,
            category=category,
        )
    ).parsed
