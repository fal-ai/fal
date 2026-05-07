from http import HTTPStatus
from typing import Any, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response


def _get_kwargs(
    *,
    endpoint: str,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["endpoint"] = endpoint

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/admin/endpoints/notify-enterprise-ready",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, str]]:
    if response.status_code == 200:
        response_200 = cast(str, response.json())
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
) -> Response[Union[HTTPValidationError, str]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    endpoint: str,
) -> Response[Union[HTTPValidationError, str]]:
    """Notify Endpoint Enterprise Ready

     Trigger enterprise ready notifications for an endpoint.
    Called after an endpoint is marked as READY to notify team admins.

    This sends email and in-app notifications to all team admins of organizations
    that have notify_enterprise_ready enabled.

    Args:
        endpoint (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, str]]
    """

    kwargs = _get_kwargs(
        endpoint=endpoint,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    endpoint: str,
) -> Optional[Union[HTTPValidationError, str]]:
    """Notify Endpoint Enterprise Ready

     Trigger enterprise ready notifications for an endpoint.
    Called after an endpoint is marked as READY to notify team admins.

    This sends email and in-app notifications to all team admins of organizations
    that have notify_enterprise_ready enabled.

    Args:
        endpoint (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, str]
    """

    return sync_detailed(
        client=client,
        endpoint=endpoint,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    endpoint: str,
) -> Response[Union[HTTPValidationError, str]]:
    """Notify Endpoint Enterprise Ready

     Trigger enterprise ready notifications for an endpoint.
    Called after an endpoint is marked as READY to notify team admins.

    This sends email and in-app notifications to all team admins of organizations
    that have notify_enterprise_ready enabled.

    Args:
        endpoint (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, str]]
    """

    kwargs = _get_kwargs(
        endpoint=endpoint,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    endpoint: str,
) -> Optional[Union[HTTPValidationError, str]]:
    """Notify Endpoint Enterprise Ready

     Trigger enterprise ready notifications for an endpoint.
    Called after an endpoint is marked as READY to notify team admins.

    This sends email and in-app notifications to all team admins of organizations
    that have notify_enterprise_ready enabled.

    Args:
        endpoint (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, str]
    """

    return (
        await asyncio_detailed(
            client=client,
            endpoint=endpoint,
        )
    ).parsed
