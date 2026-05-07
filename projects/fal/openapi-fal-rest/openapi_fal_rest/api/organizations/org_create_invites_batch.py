from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.org_batch_invite_request import OrgBatchInviteRequest
from ...models.org_batch_invite_response import OrgBatchInviteResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: OrgBatchInviteRequest,
    dry_run: Union[Unset, bool] = False,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    params: dict[str, Any] = {}

    params["dry_run"] = dry_run

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/organizations/invites/batch",
        "params": params,
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, OrgBatchInviteResponse]]:
    if response.status_code == 200:
        response_200 = OrgBatchInviteResponse.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, OrgBatchInviteResponse]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: OrgBatchInviteRequest,
    dry_run: Union[Unset, bool] = False,
) -> Response[Union[HTTPValidationError, OrgBatchInviteResponse]]:
    """Org Create Invites Batch

     Create invites in bulk with optional per-row team targeting.

    Org admins can invite users to multiple teams within their organization
    in a single request. Each invite can optionally specify a team_nickname
    to target a specific team, otherwise the default_team_nickname is used.

    - default_team_nickname: Required fallback team for invites without team_nickname
    - Each invite can optionally specify team_nickname to override the default
    - Same email can appear multiple times for different teams
    - Max 50 invites per request
    - dry_run=true validates without creating invites

    Args:
        dry_run (Union[Unset, bool]):  Default: False.
        body (OrgBatchInviteRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, OrgBatchInviteResponse]]
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
    body: OrgBatchInviteRequest,
    dry_run: Union[Unset, bool] = False,
) -> Optional[Union[HTTPValidationError, OrgBatchInviteResponse]]:
    """Org Create Invites Batch

     Create invites in bulk with optional per-row team targeting.

    Org admins can invite users to multiple teams within their organization
    in a single request. Each invite can optionally specify a team_nickname
    to target a specific team, otherwise the default_team_nickname is used.

    - default_team_nickname: Required fallback team for invites without team_nickname
    - Each invite can optionally specify team_nickname to override the default
    - Same email can appear multiple times for different teams
    - Max 50 invites per request
    - dry_run=true validates without creating invites

    Args:
        dry_run (Union[Unset, bool]):  Default: False.
        body (OrgBatchInviteRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, OrgBatchInviteResponse]
    """

    return sync_detailed(
        client=client,
        body=body,
        dry_run=dry_run,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: OrgBatchInviteRequest,
    dry_run: Union[Unset, bool] = False,
) -> Response[Union[HTTPValidationError, OrgBatchInviteResponse]]:
    """Org Create Invites Batch

     Create invites in bulk with optional per-row team targeting.

    Org admins can invite users to multiple teams within their organization
    in a single request. Each invite can optionally specify a team_nickname
    to target a specific team, otherwise the default_team_nickname is used.

    - default_team_nickname: Required fallback team for invites without team_nickname
    - Each invite can optionally specify team_nickname to override the default
    - Same email can appear multiple times for different teams
    - Max 50 invites per request
    - dry_run=true validates without creating invites

    Args:
        dry_run (Union[Unset, bool]):  Default: False.
        body (OrgBatchInviteRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, OrgBatchInviteResponse]]
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
    body: OrgBatchInviteRequest,
    dry_run: Union[Unset, bool] = False,
) -> Optional[Union[HTTPValidationError, OrgBatchInviteResponse]]:
    """Org Create Invites Batch

     Create invites in bulk with optional per-row team targeting.

    Org admins can invite users to multiple teams within their organization
    in a single request. Each invite can optionally specify a team_nickname
    to target a specific team, otherwise the default_team_nickname is used.

    - default_team_nickname: Required fallback team for invites without team_nickname
    - Each invite can optionally specify team_nickname to override the default
    - Same email can appear multiple times for different teams
    - Max 50 invites per request
    - dry_run=true validates without creating invites

    Args:
        dry_run (Union[Unset, bool]):  Default: False.
        body (OrgBatchInviteRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, OrgBatchInviteResponse]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            dry_run=dry_run,
        )
    ).parsed
