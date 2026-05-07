from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.app_file_multipart_request import AppFileMultipartRequest
from ...models.http_validation_error import HTTPValidationError
from ...models.initiate_multipart_upload_response import InitiateMultipartUploadResponse
from ...types import Response


def _get_kwargs(
    hash_: str,
    *,
    body: AppFileMultipartRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/files/app/multipart/{hash_}/initiate",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, InitiateMultipartUploadResponse]]:
    if response.status_code == 200:
        response_200 = InitiateMultipartUploadResponse.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, InitiateMultipartUploadResponse]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    hash_: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: AppFileMultipartRequest,
) -> Response[Union[HTTPValidationError, InitiateMultipartUploadResponse]]:
    """Initiate App File Multipart Upload

    Args:
        hash_ (str):
        body (AppFileMultipartRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, InitiateMultipartUploadResponse]]
    """

    kwargs = _get_kwargs(
        hash_=hash_,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    hash_: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: AppFileMultipartRequest,
) -> Optional[Union[HTTPValidationError, InitiateMultipartUploadResponse]]:
    """Initiate App File Multipart Upload

    Args:
        hash_ (str):
        body (AppFileMultipartRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, InitiateMultipartUploadResponse]
    """

    return sync_detailed(
        hash_=hash_,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    hash_: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: AppFileMultipartRequest,
) -> Response[Union[HTTPValidationError, InitiateMultipartUploadResponse]]:
    """Initiate App File Multipart Upload

    Args:
        hash_ (str):
        body (AppFileMultipartRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, InitiateMultipartUploadResponse]]
    """

    kwargs = _get_kwargs(
        hash_=hash_,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    hash_: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: AppFileMultipartRequest,
) -> Optional[Union[HTTPValidationError, InitiateMultipartUploadResponse]]:
    """Initiate App File Multipart Upload

    Args:
        hash_ (str):
        body (AppFileMultipartRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, InitiateMultipartUploadResponse]
    """

    return (
        await asyncio_detailed(
            hash_=hash_,
            client=client,
            body=body,
        )
    ).parsed
