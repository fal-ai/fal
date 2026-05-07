from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.page_type_var_customized_notification_delivery import PageTypeVarCustomizedNotificationDelivery
from ...models.user_notification_delivery_status import UserNotificationDeliveryStatus
from ...types import UNSET, Response, Unset


def _get_kwargs(
    user_id: str,
    app_name: str,
    *,
    status: Union[Unset, list[UserNotificationDeliveryStatus]] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_status: Union[Unset, list[str]] = UNSET
    if not isinstance(status, Unset):
        json_status = []
        for status_item_data in status:
            status_item = status_item_data.value
            json_status.append(status_item)

    params["status"] = json_status

    params["page"] = page

    params["size"] = size

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/admin/apps/{user_id}/{app_name}/notifications/delivery/channel/ui",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, PageTypeVarCustomizedNotificationDelivery]]:
    if response.status_code == 200:
        response_200 = PageTypeVarCustomizedNotificationDelivery.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, PageTypeVarCustomizedNotificationDelivery]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    user_id: str,
    app_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    status: Union[Unset, list[UserNotificationDeliveryStatus]] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Response[Union[HTTPValidationError, PageTypeVarCustomizedNotificationDelivery]]:
    """Admin Get App Notification Deliveries

    Args:
        user_id (str):
        app_name (str):
        status (Union[Unset, list[UserNotificationDeliveryStatus]]): The status of the deliveries
            to return
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, PageTypeVarCustomizedNotificationDelivery]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        app_name=app_name,
        status=status,
        page=page,
        size=size,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    user_id: str,
    app_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    status: Union[Unset, list[UserNotificationDeliveryStatus]] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Optional[Union[HTTPValidationError, PageTypeVarCustomizedNotificationDelivery]]:
    """Admin Get App Notification Deliveries

    Args:
        user_id (str):
        app_name (str):
        status (Union[Unset, list[UserNotificationDeliveryStatus]]): The status of the deliveries
            to return
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, PageTypeVarCustomizedNotificationDelivery]
    """

    return sync_detailed(
        user_id=user_id,
        app_name=app_name,
        client=client,
        status=status,
        page=page,
        size=size,
    ).parsed


async def asyncio_detailed(
    user_id: str,
    app_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    status: Union[Unset, list[UserNotificationDeliveryStatus]] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Response[Union[HTTPValidationError, PageTypeVarCustomizedNotificationDelivery]]:
    """Admin Get App Notification Deliveries

    Args:
        user_id (str):
        app_name (str):
        status (Union[Unset, list[UserNotificationDeliveryStatus]]): The status of the deliveries
            to return
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, PageTypeVarCustomizedNotificationDelivery]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        app_name=app_name,
        status=status,
        page=page,
        size=size,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_id: str,
    app_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    status: Union[Unset, list[UserNotificationDeliveryStatus]] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Optional[Union[HTTPValidationError, PageTypeVarCustomizedNotificationDelivery]]:
    """Admin Get App Notification Deliveries

    Args:
        user_id (str):
        app_name (str):
        status (Union[Unset, list[UserNotificationDeliveryStatus]]): The status of the deliveries
            to return
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, PageTypeVarCustomizedNotificationDelivery]
    """

    return (
        await asyncio_detailed(
            user_id=user_id,
            app_name=app_name,
            client=client,
            status=status,
            page=page,
            size=size,
        )
    ).parsed
