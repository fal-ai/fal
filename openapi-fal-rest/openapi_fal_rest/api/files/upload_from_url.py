from http import HTTPStatus
from typing import Any, Dict, Optional, Union, cast

import httpx

from ... import errors
from ...client import Client
from ...models.http_validation_error import HTTPValidationError
from ...models.url_file_upload import UrlFileUpload
from ...types import Response


def _get_kwargs(
    file: str,
    *,
    client: Client,
    json_body: UrlFileUpload,
) -> Dict[str, Any]:
    url = "{}/files/file/url/{file}".format(client.base_url, file=file)

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
    file: str,
    *,
    client: Client,
    json_body: UrlFileUpload,
) -> Response[Union[HTTPValidationError, bool]]:
    """Upload Url File

    Args:
        file (str):
        json_body (UrlFileUpload):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, bool]]
    """

    kwargs = _get_kwargs(
        file=file,
        client=client,
        json_body=json_body,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    file: str,
    *,
    client: Client,
    json_body: UrlFileUpload,
) -> Optional[Union[HTTPValidationError, bool]]:
    """Upload Url File

    Args:
        file (str):
        json_body (UrlFileUpload):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, bool]
    """

    return sync_detailed(
        file=file,
        client=client,
        json_body=json_body,
    ).parsed


async def asyncio_detailed(
    file: str,
    *,
    client: Client,
    json_body: UrlFileUpload,
) -> Response[Union[HTTPValidationError, bool]]:
    """Upload Url File

    Args:
        file (str):
        json_body (UrlFileUpload):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, bool]]
    """

    kwargs = _get_kwargs(
        file=file,
        client=client,
        json_body=json_body,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    file: str,
    *,
    client: Client,
    json_body: UrlFileUpload,
) -> Optional[Union[HTTPValidationError, bool]]:
    """Upload Url File

    Args:
        file (str):
        json_body (UrlFileUpload):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, bool]
    """

    return (
        await asyncio_detailed(
            file=file,
            client=client,
            json_body=json_body,
        )
    ).parsed
