from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.batch_invite_request import BatchInviteRequest
from ...models.batch_invite_response import BatchInviteResponse
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: BatchInviteRequest,
    dry_run: Union[Unset, bool] = False,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    params: dict[str, Any] = {}

    params["dry_run"] = dry_run

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/users/invites/batch",
        "params": params,
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[BatchInviteResponse, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = BatchInviteResponse.from_dict(response.json())

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
) -> Response[Union[BatchInviteResponse, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: BatchInviteRequest,
    dry_run: Union[Unset, bool] = False,
) -> Response[Union[BatchInviteResponse, HTTPValidationError]]:
    """Create Invites Batch

     Create invites in bulk (up to 50 per request).

    Args:
        dry_run (Union[Unset, bool]):  Default: False.
        body (BatchInviteRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[BatchInviteResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
        dry_run=dry_run,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    body: BatchInviteRequest,
    dry_run: Union[Unset, bool] = False,
) -> Optional[Union[BatchInviteResponse, HTTPValidationError]]:
    """Create Invites Batch

     Create invites in bulk (up to 50 per request).

    Args:
        dry_run (Union[Unset, bool]):  Default: False.
        body (BatchInviteRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[BatchInviteResponse, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        body=body,
        dry_run=dry_run,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: BatchInviteRequest,
    dry_run: Union[Unset, bool] = False,
) -> Response[Union[BatchInviteResponse, HTTPValidationError]]:
    """Create Invites Batch

     Create invites in bulk (up to 50 per request).

    Args:
        dry_run (Union[Unset, bool]):  Default: False.
        body (BatchInviteRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[BatchInviteResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
        dry_run=dry_run,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: BatchInviteRequest,
    dry_run: Union[Unset, bool] = False,
) -> Optional[Union[BatchInviteResponse, HTTPValidationError]]:
    """Create Invites Batch

     Create invites in bulk (up to 50 per request).

    Args:
        dry_run (Union[Unset, bool]):  Default: False.
        body (BatchInviteRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[BatchInviteResponse, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            dry_run=dry_run,
        )
    ).parsed
