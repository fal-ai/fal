from http import HTTPStatus
from typing import Any, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    uuid: str,
    *,
    post_request: Union[Unset, bool] = False,
    content_length: Union[Unset, int] = UNSET,
    upload_offset: Union[Unset, int] = UNSET,
    upload_length: Union[Unset, int] = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(content_length, Unset):
        headers["content-length"] = str(content_length)

    if not isinstance(upload_offset, Unset):
        headers["upload-offset"] = str(upload_offset)

    if not isinstance(upload_length, Unset):
        headers["upload-length"] = str(upload_length)

    params: dict[str, Any] = {}

    params["post_request"] = post_request

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "patch",
        "url": f"/files/tus/{uuid}",
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, HTTPValidationError]]:
    if response.status_code == 204:
        response_204 = cast(Any, None)
        return response_204
    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[Any, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    uuid: str,
    *,
    client: Union[AuthenticatedClient, Client],
    post_request: Union[Unset, bool] = False,
    content_length: Union[Unset, int] = UNSET,
    upload_offset: Union[Unset, int] = UNSET,
    upload_length: Union[Unset, int] = UNSET,
) -> Response[Union[Any, HTTPValidationError]]:
    """Core Patch Route

    Args:
        uuid (str):
        post_request (Union[Unset, bool]):  Default: False.
        content_length (Union[Unset, int]):
        upload_offset (Union[Unset, int]):
        upload_length (Union[Unset, int]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        uuid=uuid,
        post_request=post_request,
        content_length=content_length,
        upload_offset=upload_offset,
        upload_length=upload_length,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    uuid: str,
    *,
    client: Union[AuthenticatedClient, Client],
    post_request: Union[Unset, bool] = False,
    content_length: Union[Unset, int] = UNSET,
    upload_offset: Union[Unset, int] = UNSET,
    upload_length: Union[Unset, int] = UNSET,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Core Patch Route

    Args:
        uuid (str):
        post_request (Union[Unset, bool]):  Default: False.
        content_length (Union[Unset, int]):
        upload_offset (Union[Unset, int]):
        upload_length (Union[Unset, int]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return sync_detailed(
        uuid=uuid,
        client=client,
        post_request=post_request,
        content_length=content_length,
        upload_offset=upload_offset,
        upload_length=upload_length,
    ).parsed


async def asyncio_detailed(
    uuid: str,
    *,
    client: Union[AuthenticatedClient, Client],
    post_request: Union[Unset, bool] = False,
    content_length: Union[Unset, int] = UNSET,
    upload_offset: Union[Unset, int] = UNSET,
    upload_length: Union[Unset, int] = UNSET,
) -> Response[Union[Any, HTTPValidationError]]:
    """Core Patch Route

    Args:
        uuid (str):
        post_request (Union[Unset, bool]):  Default: False.
        content_length (Union[Unset, int]):
        upload_offset (Union[Unset, int]):
        upload_length (Union[Unset, int]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        uuid=uuid,
        post_request=post_request,
        content_length=content_length,
        upload_offset=upload_offset,
        upload_length=upload_length,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    uuid: str,
    *,
    client: Union[AuthenticatedClient, Client],
    post_request: Union[Unset, bool] = False,
    content_length: Union[Unset, int] = UNSET,
    upload_offset: Union[Unset, int] = UNSET,
    upload_length: Union[Unset, int] = UNSET,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Core Patch Route

    Args:
        uuid (str):
        post_request (Union[Unset, bool]):  Default: False.
        content_length (Union[Unset, int]):
        upload_offset (Union[Unset, int]):
        upload_length (Union[Unset, int]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            uuid=uuid,
            client=client,
            post_request=post_request,
            content_length=content_length,
            upload_offset=upload_offset,
            upload_length=upload_length,
        )
    ).parsed
