from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.complete_multipart_upload_request import CompleteMultipartUploadRequest
from ...models.complete_multipart_upload_response import CompleteMultipartUploadResponse
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    target_path: str,
    upload_id: str,
    *,
    body: CompleteMultipartUploadRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/files/file/multipart/{target_path}/{upload_id}/complete",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[CompleteMultipartUploadResponse, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = CompleteMultipartUploadResponse.from_dict(response.json())

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
) -> Response[Union[CompleteMultipartUploadResponse, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    target_path: str,
    upload_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: CompleteMultipartUploadRequest,
) -> Response[Union[CompleteMultipartUploadResponse, HTTPValidationError]]:
    """Complete Multipart Upload Local File

    Args:
        target_path (str):
        upload_id (str):
        body (CompleteMultipartUploadRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[CompleteMultipartUploadResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        target_path=target_path,
        upload_id=upload_id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    target_path: str,
    upload_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: CompleteMultipartUploadRequest,
) -> Optional[Union[CompleteMultipartUploadResponse, HTTPValidationError]]:
    """Complete Multipart Upload Local File

    Args:
        target_path (str):
        upload_id (str):
        body (CompleteMultipartUploadRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[CompleteMultipartUploadResponse, HTTPValidationError]
    """

    return sync_detailed(
        target_path=target_path,
        upload_id=upload_id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    target_path: str,
    upload_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: CompleteMultipartUploadRequest,
) -> Response[Union[CompleteMultipartUploadResponse, HTTPValidationError]]:
    """Complete Multipart Upload Local File

    Args:
        target_path (str):
        upload_id (str):
        body (CompleteMultipartUploadRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[CompleteMultipartUploadResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        target_path=target_path,
        upload_id=upload_id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    target_path: str,
    upload_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: CompleteMultipartUploadRequest,
) -> Optional[Union[CompleteMultipartUploadResponse, HTTPValidationError]]:
    """Complete Multipart Upload Local File

    Args:
        target_path (str):
        upload_id (str):
        body (CompleteMultipartUploadRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[CompleteMultipartUploadResponse, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            target_path=target_path,
            upload_id=upload_id,
            client=client,
            body=body,
        )
    ).parsed
