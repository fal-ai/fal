from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.decorate_auth_methods_response import DecorateAuthMethodsResponse
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    user_str: str,
    *,
    body: list[str],
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/admin/organizations/{user_str}/decorate-auth-methods",
    }

    _body = body

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[DecorateAuthMethodsResponse, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = DecorateAuthMethodsResponse.from_dict(response.json())

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
) -> Response[Union[DecorateAuthMethodsResponse, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: list[str],
) -> Response[Union[DecorateAuthMethodsResponse, HTTPValidationError]]:
    """Decorate Auth Methods

     Decorate auth method strings with human-readable user/key information.

    Args:
        user_str (str):
        body (list[str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[DecorateAuthMethodsResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_str=user_str,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: list[str],
) -> Optional[Union[DecorateAuthMethodsResponse, HTTPValidationError]]:
    """Decorate Auth Methods

     Decorate auth method strings with human-readable user/key information.

    Args:
        user_str (str):
        body (list[str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[DecorateAuthMethodsResponse, HTTPValidationError]
    """

    return sync_detailed(
        user_str=user_str,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: list[str],
) -> Response[Union[DecorateAuthMethodsResponse, HTTPValidationError]]:
    """Decorate Auth Methods

     Decorate auth method strings with human-readable user/key information.

    Args:
        user_str (str):
        body (list[str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[DecorateAuthMethodsResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_str=user_str,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: list[str],
) -> Optional[Union[DecorateAuthMethodsResponse, HTTPValidationError]]:
    """Decorate Auth Methods

     Decorate auth method strings with human-readable user/key information.

    Args:
        user_str (str):
        body (list[str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[DecorateAuthMethodsResponse, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            user_str=user_str,
            client=client,
            body=body,
        )
    ).parsed
