from http import HTTPStatus
from typing import Any, Dict, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.key_scope import KeyScope
from ...models.new_user_key import NewUserKey
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    client: AuthenticatedClient,
    scope: KeyScope,
    alias: Union[Unset, None, str] = "",
) -> Dict[str, Any]:
    url = "{}/keys/".format(client.base_url)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    json_scope = scope.value

    params["scope"] = json_scope

    params["alias"] = alias

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    return {
        "method": "post",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "follow_redirects": client.follow_redirects,
        "params": params,
    }


def _parse_response(*, client: Client, response: httpx.Response) -> Optional[Union[HTTPValidationError, NewUserKey]]:
    if response.status_code == HTTPStatus.CREATED:
        response_201 = NewUserKey.from_dict(response.json())

        return response_201
    if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(*, client: Client, response: httpx.Response) -> Response[Union[HTTPValidationError, NewUserKey]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    scope: KeyScope,
    alias: Union[Unset, None, str] = "",
) -> Response[Union[HTTPValidationError, NewUserKey]]:
    """Create

    Args:
        scope (KeyScope): An enumeration.
        alias (Union[Unset, None, str]):  Default: ''.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, NewUserKey]]
    """

    kwargs = _get_kwargs(
        client=client,
        scope=scope,
        alias=alias,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
    scope: KeyScope,
    alias: Union[Unset, None, str] = "",
) -> Optional[Union[HTTPValidationError, NewUserKey]]:
    """Create

    Args:
        scope (KeyScope): An enumeration.
        alias (Union[Unset, None, str]):  Default: ''.

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
    client: AuthenticatedClient,
    scope: KeyScope,
    alias: Union[Unset, None, str] = "",
) -> Response[Union[HTTPValidationError, NewUserKey]]:
    """Create

    Args:
        scope (KeyScope): An enumeration.
        alias (Union[Unset, None, str]):  Default: ''.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, NewUserKey]]
    """

    kwargs = _get_kwargs(
        client=client,
        scope=scope,
        alias=alias,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    scope: KeyScope,
    alias: Union[Unset, None, str] = "",
) -> Optional[Union[HTTPValidationError, NewUserKey]]:
    """Create

    Args:
        scope (KeyScope): An enumeration.
        alias (Union[Unset, None, str]):  Default: ''.

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
