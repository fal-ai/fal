from http import HTTPStatus
from typing import Any, Dict, Optional, Union, cast

import httpx

from ... import errors
from ...client import Client
from ...models.hash_check import HashCheck
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    target_path: str,
    *,
    client: Client,
    json_body: HashCheck,
) -> Dict[str, Any]:
    url = "{}/files/dir/check_hash/{target_path}".format(client.base_url, target_path=target_path)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    json_json_body = json_body.to_dict()

    return {
        "method": "post",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "follow_redirects": client.follow_redirects,
        "json": json_json_body,
    }


def _parse_response(*, client: Client, response: httpx.Response) -> Optional[Union[HTTPValidationError, bool]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = cast(bool, response.json())
        return response_200
    if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(*, client: Client, response: httpx.Response) -> Response[Union[HTTPValidationError, bool]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    target_path: str,
    *,
    client: Client,
    json_body: HashCheck,
) -> Response[Union[HTTPValidationError, bool]]:
    """Check Dir Hash

    Args:
        target_path (str):
        json_body (HashCheck):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, bool]]
    """

    kwargs = _get_kwargs(
        target_path=target_path,
        client=client,
        json_body=json_body,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    target_path: str,
    *,
    client: Client,
    json_body: HashCheck,
) -> Optional[Union[HTTPValidationError, bool]]:
    """Check Dir Hash

    Args:
        target_path (str):
        json_body (HashCheck):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, bool]
    """

    return sync_detailed(
        target_path=target_path,
        client=client,
        json_body=json_body,
    ).parsed


async def asyncio_detailed(
    target_path: str,
    *,
    client: Client,
    json_body: HashCheck,
) -> Response[Union[HTTPValidationError, bool]]:
    """Check Dir Hash

    Args:
        target_path (str):
        json_body (HashCheck):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, bool]]
    """

    kwargs = _get_kwargs(
        target_path=target_path,
        client=client,
        json_body=json_body,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    target_path: str,
    *,
    client: Client,
    json_body: HashCheck,
) -> Optional[Union[HTTPValidationError, bool]]:
    """Check Dir Hash

    Args:
        target_path (str):
        json_body (HashCheck):

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
            json_body=json_body,
        )
    ).parsed
