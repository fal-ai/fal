from http import HTTPStatus
from typing import Any, Dict, Optional, Union

import httpx

from ... import errors
from ...client import Client
from ...models.http_validation_error import HTTPValidationError
from ...models.status import Status
from ...types import Response


def _get_kwargs(
    app_user_id: str,
    app_alias_or_id: str,
    *,
    client: Client,
) -> Dict[str, Any]:
    url = "{}/applications/{app_user_id}/{app_alias_or_id}/status".format(
        client.base_url, app_user_id=app_user_id, app_alias_or_id=app_alias_or_id
    )

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "follow_redirects": client.follow_redirects,
    }


def _parse_response(*, client: Client, response: httpx.Response) -> Optional[Union[HTTPValidationError, Status]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = Status.from_dict(response.json())

        return response_200
    if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(*, client: Client, response: httpx.Response) -> Response[Union[HTTPValidationError, Status]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    app_user_id: str,
    app_alias_or_id: str,
    *,
    client: Client,
) -> Response[Union[HTTPValidationError, Status]]:
    """Get Status

    Args:
        app_user_id (str):
        app_alias_or_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, Status]]
    """

    kwargs = _get_kwargs(
        app_user_id=app_user_id,
        app_alias_or_id=app_alias_or_id,
        client=client,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    app_user_id: str,
    app_alias_or_id: str,
    *,
    client: Client,
) -> Optional[Union[HTTPValidationError, Status]]:
    """Get Status

    Args:
        app_user_id (str):
        app_alias_or_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, Status]
    """

    return sync_detailed(
        app_user_id=app_user_id,
        app_alias_or_id=app_alias_or_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    app_user_id: str,
    app_alias_or_id: str,
    *,
    client: Client,
) -> Response[Union[HTTPValidationError, Status]]:
    """Get Status

    Args:
        app_user_id (str):
        app_alias_or_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, Status]]
    """

    kwargs = _get_kwargs(
        app_user_id=app_user_id,
        app_alias_or_id=app_alias_or_id,
        client=client,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    app_user_id: str,
    app_alias_or_id: str,
    *,
    client: Client,
) -> Optional[Union[HTTPValidationError, Status]]:
    """Get Status

    Args:
        app_user_id (str):
        app_alias_or_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, Status]
    """

    return (
        await asyncio_detailed(
            app_user_id=app_user_id,
            app_alias_or_id=app_alias_or_id,
            client=client,
        )
    ).parsed
