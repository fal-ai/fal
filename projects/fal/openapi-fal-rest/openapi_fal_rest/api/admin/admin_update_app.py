from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.app_info import AppInfo
from ...models.app_info_update import AppInfoUpdate
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    user_id: str,
    app_name: str,
    *,
    body: AppInfoUpdate,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": f"/admin/apps/{user_id}/{app_name}",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[AppInfo, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = AppInfo.from_dict(response.json())

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
) -> Response[Union[AppInfo, HTTPValidationError]]:
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
    body: AppInfoUpdate,
) -> Response[Union[AppInfo, HTTPValidationError]]:
    """Update App

    Args:
        user_id (str):
        app_name (str):
        body (AppInfoUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AppInfo, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        app_name=app_name,
        body=body,
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
    body: AppInfoUpdate,
) -> Optional[Union[AppInfo, HTTPValidationError]]:
    """Update App

    Args:
        user_id (str):
        app_name (str):
        body (AppInfoUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AppInfo, HTTPValidationError]
    """

    return sync_detailed(
        user_id=user_id,
        app_name=app_name,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    user_id: str,
    app_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: AppInfoUpdate,
) -> Response[Union[AppInfo, HTTPValidationError]]:
    """Update App

    Args:
        user_id (str):
        app_name (str):
        body (AppInfoUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AppInfo, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        app_name=app_name,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_id: str,
    app_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: AppInfoUpdate,
) -> Optional[Union[AppInfo, HTTPValidationError]]:
    """Update App

    Args:
        user_id (str):
        app_name (str):
        body (AppInfoUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AppInfo, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            user_id=user_id,
            app_name=app_name,
            client=client,
            body=body,
        )
    ).parsed
