from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.scale_app_request import ScaleAppRequest
from ...models.user_app_info import UserAppInfo
from ...types import Response


def _get_kwargs(
    application_name: str,
    *,
    body: ScaleAppRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": f"/applications/scale/{application_name}",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, UserAppInfo]]:
    if response.status_code == 200:
        response_200 = UserAppInfo.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, UserAppInfo]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    application_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: ScaleAppRequest,
) -> Response[Union[HTTPValidationError, UserAppInfo]]:
    """Scale Application

    Args:
        application_name (str):
        body (ScaleAppRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, UserAppInfo]]
    """

    kwargs = _get_kwargs(
        application_name=application_name,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    application_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: ScaleAppRequest,
) -> Optional[Union[HTTPValidationError, UserAppInfo]]:
    """Scale Application

    Args:
        application_name (str):
        body (ScaleAppRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, UserAppInfo]
    """

    return sync_detailed(
        application_name=application_name,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    application_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: ScaleAppRequest,
) -> Response[Union[HTTPValidationError, UserAppInfo]]:
    """Scale Application

    Args:
        application_name (str):
        body (ScaleAppRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, UserAppInfo]]
    """

    kwargs = _get_kwargs(
        application_name=application_name,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    application_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: ScaleAppRequest,
) -> Optional[Union[HTTPValidationError, UserAppInfo]]:
    """Scale Application

    Args:
        application_name (str):
        body (ScaleAppRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, UserAppInfo]
    """

    return (
        await asyncio_detailed(
            application_name=application_name,
            client=client,
            body=body,
        )
    ).parsed
