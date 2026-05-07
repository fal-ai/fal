from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.sign_file_url_request import SignFileUrlRequest
from ...models.sign_file_url_response import SignFileUrlResponse
from ...types import UNSET, Response


def _get_kwargs(
    *,
    body: SignFileUrlRequest,
    url_query: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    params: dict[str, Any] = {}

    params["url"] = url_query

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/storage/files/sign",
        "params": params,
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, SignFileUrlResponse]]:
    if response.status_code == 200:
        response_200 = SignFileUrlResponse.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, SignFileUrlResponse]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: SignFileUrlRequest,
    url_query: str,
) -> Response[Union[HTTPValidationError, SignFileUrlResponse]]:
    """Sign File Url

    Args:
        url_query (str):
        body (SignFileUrlRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, SignFileUrlResponse]]
    """

    kwargs = _get_kwargs(
        body=body,
        url_query=url_query,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    body: SignFileUrlRequest,
    url_query: str,
) -> Optional[Union[HTTPValidationError, SignFileUrlResponse]]:
    """Sign File Url

    Args:
        url_query (str):
        body (SignFileUrlRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, SignFileUrlResponse]
    """

    return sync_detailed(
        client=client,
        body=body,
        url_query=url_query,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: SignFileUrlRequest,
    url_query: str,
) -> Response[Union[HTTPValidationError, SignFileUrlResponse]]:
    """Sign File Url

    Args:
        url_query (str):
        body (SignFileUrlRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, SignFileUrlResponse]]
    """

    kwargs = _get_kwargs(
        body=body,
        url_query=url_query,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: SignFileUrlRequest,
    url_query: str,
) -> Optional[Union[HTTPValidationError, SignFileUrlResponse]]:
    """Sign File Url

    Args:
        url_query (str):
        body (SignFileUrlRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, SignFileUrlResponse]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            url_query=url_query,
        )
    ).parsed
