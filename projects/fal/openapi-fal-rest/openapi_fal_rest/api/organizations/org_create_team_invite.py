from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.sent_user_invite import SentUserInvite
from ...models.user_invite import UserInvite
from ...types import Response


def _get_kwargs(
    team_user_str: str,
    *,
    body: UserInvite,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/organizations/teams/{team_user_str}/invites",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, SentUserInvite]]:
    if response.status_code == 201:
        response_201 = SentUserInvite.from_dict(response.json())

        return response_201
    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[HTTPValidationError, SentUserInvite]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    team_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: UserInvite,
) -> Response[Union[HTTPValidationError, SentUserInvite]]:
    """Org Create Team Invite

     Create an invite for a team within the organization.

    Org admins can invite users to any team within their organization.
    Team is identified by user_id or nickname.
    Requires the caller to be an org admin.

    Args:
        team_user_str (str):
        body (UserInvite): Request model for creating an invite.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, SentUserInvite]]
    """

    kwargs = _get_kwargs(
        team_user_str=team_user_str,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    team_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: UserInvite,
) -> Optional[Union[HTTPValidationError, SentUserInvite]]:
    """Org Create Team Invite

     Create an invite for a team within the organization.

    Org admins can invite users to any team within their organization.
    Team is identified by user_id or nickname.
    Requires the caller to be an org admin.

    Args:
        team_user_str (str):
        body (UserInvite): Request model for creating an invite.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, SentUserInvite]
    """

    return sync_detailed(
        team_user_str=team_user_str,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    team_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: UserInvite,
) -> Response[Union[HTTPValidationError, SentUserInvite]]:
    """Org Create Team Invite

     Create an invite for a team within the organization.

    Org admins can invite users to any team within their organization.
    Team is identified by user_id or nickname.
    Requires the caller to be an org admin.

    Args:
        team_user_str (str):
        body (UserInvite): Request model for creating an invite.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, SentUserInvite]]
    """

    kwargs = _get_kwargs(
        team_user_str=team_user_str,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    team_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: UserInvite,
) -> Optional[Union[HTTPValidationError, SentUserInvite]]:
    """Org Create Team Invite

     Create an invite for a team within the organization.

    Org admins can invite users to any team within their organization.
    Team is identified by user_id or nickname.
    Requires the caller to be an org admin.

    Args:
        team_user_str (str):
        body (UserInvite): Request model for creating an invite.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, SentUserInvite]
    """

    return (
        await asyncio_detailed(
            team_user_str=team_user_str,
            client=client,
            body=body,
        )
    ).parsed
