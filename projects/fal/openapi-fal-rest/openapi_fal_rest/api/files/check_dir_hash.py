from http import HTTPStatus
from typing import Any, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.hash_check import HashCheck
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    target_path: str,
    *,
    body: HashCheck,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/files/dir/check_hash/{target_path}",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, bool]]:
    if response.status_code == 200:
        response_200 = cast(bool, response.json())
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
) -> Response[Union[HTTPValidationError, bool]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    target_path: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: HashCheck,
) -> Response[Union[HTTPValidationError, bool]]:
    """Check Dir Hash

    Args:
        target_path (str):
        body (HashCheck):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, bool]]
    """

    kwargs = _get_kwargs(
        target_path=target_path,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    target_path: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: HashCheck,
) -> Optional[Union[HTTPValidationError, bool]]:
    """Check Dir Hash

    Args:
        target_path (str):
        body (HashCheck):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, bool]
    """

    return sync_detailed(
        target_path=target_path,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    target_path: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: HashCheck,
) -> Response[Union[HTTPValidationError, bool]]:
    """Check Dir Hash

    Args:
        target_path (str):
        body (HashCheck):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, bool]]
    """

    kwargs = _get_kwargs(
        target_path=target_path,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    target_path: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: HashCheck,
) -> Optional[Union[HTTPValidationError, bool]]:
    """Check Dir Hash

    Args:
        target_path (str):
        body (HashCheck):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, bool]
    """

    return (
        await asyncio_detailed(
            target_path=target_path,
            client=client,
            body=body,
        )
    ).parsed
