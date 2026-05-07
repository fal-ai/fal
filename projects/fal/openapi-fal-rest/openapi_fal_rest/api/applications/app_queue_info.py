from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.queue_info import QueueInfo
from ...types import UNSET, Response, Unset


def _get_kwargs(
    app_user_id: str,
    app_alias: str,
    *,
    by_user: Union[Unset, bool] = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["by_user"] = by_user

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/applications/{app_user_id}/{app_alias}/queue",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, QueueInfo]]:
    if response.status_code == 200:
        response_200 = QueueInfo.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, QueueInfo]]:
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
    by_user: Union[Unset, bool] = False,
) -> Response[Union[HTTPValidationError, QueueInfo]]:
    """App Queue Info

    Args:
        app_user_id (str):
        app_alias (str):
        by_user (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, QueueInfo]]
    """

    kwargs = _get_kwargs(
        app_user_id=app_user_id,
        app_alias=app_alias,
        by_user=by_user,
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
    by_user: Union[Unset, bool] = False,
) -> Optional[Union[HTTPValidationError, QueueInfo]]:
    """App Queue Info

    Args:
        app_user_id (str):
        app_alias (str):
        by_user (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, QueueInfo]
    """

    return sync_detailed(
        app_user_id=app_user_id,
        app_alias=app_alias,
        client=client,
        by_user=by_user,
    ).parsed


async def asyncio_detailed(
    app_user_id: str,
    app_alias: str,
    *,
    client: Union[AuthenticatedClient, Client],
    by_user: Union[Unset, bool] = False,
) -> Response[Union[HTTPValidationError, QueueInfo]]:
    """App Queue Info

    Args:
        app_user_id (str):
        app_alias (str):
        by_user (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, QueueInfo]]
    """

    kwargs = _get_kwargs(
        app_user_id=app_user_id,
        app_alias=app_alias,
        by_user=by_user,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    app_user_id: str,
    app_alias: str,
    *,
    client: Union[AuthenticatedClient, Client],
    by_user: Union[Unset, bool] = False,
) -> Optional[Union[HTTPValidationError, QueueInfo]]:
    """App Queue Info

    Args:
        app_user_id (str):
        app_alias (str):
        by_user (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, QueueInfo]
    """

    return (
        await asyncio_detailed(
            app_user_id=app_user_id,
            app_alias=app_alias,
            client=client,
            by_user=by_user,
        )
    ).parsed
