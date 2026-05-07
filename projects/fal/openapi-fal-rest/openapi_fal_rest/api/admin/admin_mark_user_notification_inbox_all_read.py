from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.mark_all_read_response import MarkAllReadResponse
from ...models.user_notification_category import UserNotificationCategory
from ...models.user_notification_subcategory import UserNotificationSubcategory
from ...types import UNSET, Response, Unset


def _get_kwargs(
    user_id: str,
    *,
    category: Union[Unset, list[UserNotificationCategory]] = UNSET,
    subcategory: Union[Unset, list[UserNotificationSubcategory]] = UNSET,
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

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/admin/users/{user_id}/notifications/inbox/mark-all-read",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, MarkAllReadResponse]]:
    if response.status_code == 200:
        response_200 = MarkAllReadResponse.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, MarkAllReadResponse]]:
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
    subcategory: Union[Unset, list[UserNotificationSubcategory]] = UNSET,
) -> Response[Union[HTTPValidationError, MarkAllReadResponse]]:
    """Admin Mark User Notification Inbox All Read

    Args:
        user_id (str):
        category (Union[Unset, list[UserNotificationCategory]]):
        subcategory (Union[Unset, list[UserNotificationSubcategory]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, MarkAllReadResponse]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        category=category,
        subcategory=subcategory,
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
    subcategory: Union[Unset, list[UserNotificationSubcategory]] = UNSET,
) -> Optional[Union[HTTPValidationError, MarkAllReadResponse]]:
    """Admin Mark User Notification Inbox All Read

    Args:
        user_id (str):
        category (Union[Unset, list[UserNotificationCategory]]):
        subcategory (Union[Unset, list[UserNotificationSubcategory]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, MarkAllReadResponse]
    """

    return sync_detailed(
        user_id=user_id,
        client=client,
        category=category,
        subcategory=subcategory,
    ).parsed


async def asyncio_detailed(
    user_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    category: Union[Unset, list[UserNotificationCategory]] = UNSET,
    subcategory: Union[Unset, list[UserNotificationSubcategory]] = UNSET,
) -> Response[Union[HTTPValidationError, MarkAllReadResponse]]:
    """Admin Mark User Notification Inbox All Read

    Args:
        user_id (str):
        category (Union[Unset, list[UserNotificationCategory]]):
        subcategory (Union[Unset, list[UserNotificationSubcategory]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, MarkAllReadResponse]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        category=category,
        subcategory=subcategory,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    category: Union[Unset, list[UserNotificationCategory]] = UNSET,
    subcategory: Union[Unset, list[UserNotificationSubcategory]] = UNSET,
) -> Optional[Union[HTTPValidationError, MarkAllReadResponse]]:
    """Admin Mark User Notification Inbox All Read

    Args:
        user_id (str):
        category (Union[Unset, list[UserNotificationCategory]]):
        subcategory (Union[Unset, list[UserNotificationSubcategory]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, MarkAllReadResponse]
    """

    return (
        await asyncio_detailed(
            user_id=user_id,
            client=client,
            category=category,
            subcategory=subcategory,
        )
    ).parsed
