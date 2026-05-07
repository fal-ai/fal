from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.key_scope import KeyScope
from ...models.new_user_key import NewUserKey
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    scope: KeyScope,
    alias: Union[Unset, str] = "",
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_scope = scope.value
    params["scope"] = json_scope

    params["alias"] = alias

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/keys/",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, NewUserKey]]:
    if response.status_code == 201:
        response_201 = NewUserKey.from_dict(response.json())

        return response_201
    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[HTTPValidationError, NewUserKey]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    scope: KeyScope,
    alias: Union[Unset, str] = "",
) -> Response[Union[HTTPValidationError, NewUserKey]]:
    """Create

    Args:
        scope (KeyScope):
        alias (Union[Unset, str]):  Default: ''.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, NewUserKey]]
    """

    kwargs = _get_kwargs(
        scope=scope,
        alias=alias,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    scope: KeyScope,
    alias: Union[Unset, str] = "",
) -> Optional[Union[HTTPValidationError, NewUserKey]]:
    """Create

    Args:
        scope (KeyScope):
        alias (Union[Unset, str]):  Default: ''.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, NewUserKey]
    """

    return sync_detailed(
        client=client,
        scope=scope,
        alias=alias,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    scope: KeyScope,
    alias: Union[Unset, str] = "",
) -> Response[Union[HTTPValidationError, NewUserKey]]:
    """Create

    Args:
        scope (KeyScope):
        alias (Union[Unset, str]):  Default: ''.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, NewUserKey]]
    """

    kwargs = _get_kwargs(
        scope=scope,
        alias=alias,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    scope: KeyScope,
    alias: Union[Unset, str] = "",
) -> Optional[Union[HTTPValidationError, NewUserKey]]:
    """Create

    Args:
        scope (KeyScope):
        alias (Union[Unset, str]):  Default: ''.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, NewUserKey]
    """

    return (
        await asyncio_detailed(
            client=client,
            scope=scope,
            alias=alias,
        )
    ).parsed
