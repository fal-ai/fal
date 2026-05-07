import datetime
from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.app_events_page import AppEventsPage
from ...models.application_event_category import ApplicationEventCategory
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    app_user_id: str,
    app_alias: str,
    *,
    since: Union[Unset, datetime.datetime] = UNSET,
    until: Union[Unset, datetime.datetime] = UNSET,
    categories: Union[Unset, list[ApplicationEventCategory]] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_since: Union[Unset, str] = UNSET
    if not isinstance(since, Unset):
        json_since = since.isoformat()
    params["since"] = json_since

    json_until: Union[Unset, str] = UNSET
    if not isinstance(until, Unset):
        json_until = until.isoformat()
    params["until"] = json_until

    json_categories: Union[Unset, list[str]] = UNSET
    if not isinstance(categories, Unset):
        json_categories = []
        for categories_item_data in categories:
            categories_item = categories_item_data.value
            json_categories.append(categories_item)

    params["categories"] = json_categories

    params["page"] = page

    params["size"] = size

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/applications/{app_user_id}/{app_alias}/events",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[AppEventsPage, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = AppEventsPage.from_dict(response.json())

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
) -> Response[Union[AppEventsPage, HTTPValidationError]]:
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
    since: Union[Unset, datetime.datetime] = UNSET,
    until: Union[Unset, datetime.datetime] = UNSET,
    categories: Union[Unset, list[ApplicationEventCategory]] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Response[Union[AppEventsPage, HTTPValidationError]]:
    """App Events

    Args:
        app_user_id (str):
        app_alias (str):
        since (Union[Unset, datetime.datetime]):
        until (Union[Unset, datetime.datetime]):
        categories (Union[Unset, list[ApplicationEventCategory]]):
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AppEventsPage, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        app_user_id=app_user_id,
        app_alias=app_alias,
        since=since,
        until=until,
        categories=categories,
        page=page,
        size=size,
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
    since: Union[Unset, datetime.datetime] = UNSET,
    until: Union[Unset, datetime.datetime] = UNSET,
    categories: Union[Unset, list[ApplicationEventCategory]] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Optional[Union[AppEventsPage, HTTPValidationError]]:
    """App Events

    Args:
        app_user_id (str):
        app_alias (str):
        since (Union[Unset, datetime.datetime]):
        until (Union[Unset, datetime.datetime]):
        categories (Union[Unset, list[ApplicationEventCategory]]):
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AppEventsPage, HTTPValidationError]
    """

    return sync_detailed(
        app_user_id=app_user_id,
        app_alias=app_alias,
        client=client,
        since=since,
        until=until,
        categories=categories,
        page=page,
        size=size,
    ).parsed


async def asyncio_detailed(
    app_user_id: str,
    app_alias: str,
    *,
    client: Union[AuthenticatedClient, Client],
    since: Union[Unset, datetime.datetime] = UNSET,
    until: Union[Unset, datetime.datetime] = UNSET,
    categories: Union[Unset, list[ApplicationEventCategory]] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Response[Union[AppEventsPage, HTTPValidationError]]:
    """App Events

    Args:
        app_user_id (str):
        app_alias (str):
        since (Union[Unset, datetime.datetime]):
        until (Union[Unset, datetime.datetime]):
        categories (Union[Unset, list[ApplicationEventCategory]]):
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AppEventsPage, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        app_user_id=app_user_id,
        app_alias=app_alias,
        since=since,
        until=until,
        categories=categories,
        page=page,
        size=size,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    app_user_id: str,
    app_alias: str,
    *,
    client: Union[AuthenticatedClient, Client],
    since: Union[Unset, datetime.datetime] = UNSET,
    until: Union[Unset, datetime.datetime] = UNSET,
    categories: Union[Unset, list[ApplicationEventCategory]] = UNSET,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Optional[Union[AppEventsPage, HTTPValidationError]]:
    """App Events

    Args:
        app_user_id (str):
        app_alias (str):
        since (Union[Unset, datetime.datetime]):
        until (Union[Unset, datetime.datetime]):
        categories (Union[Unset, list[ApplicationEventCategory]]):
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AppEventsPage, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            app_user_id=app_user_id,
            app_alias=app_alias,
            client=client,
            since=since,
            until=until,
            categories=categories,
            page=page,
            size=size,
        )
    ).parsed
