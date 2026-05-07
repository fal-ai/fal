from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.body_org_add_team_member import BodyOrgAddTeamMember
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    team_user_str: str,
    member_user_str: str,
    *,
    body: BodyOrgAddTeamMember,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/organizations/teams/{team_user_str}/members/{member_user_str}",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = response.json()
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
) -> Response[Union[Any, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    team_user_str: str,
    member_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: BodyOrgAddTeamMember,
) -> Response[Union[Any, HTTPValidationError]]:
    """Org Add Team Member

     Add a member to a team.

    Org admins can add members to teams within their organization.
    The member must either:
    - Already be part of the org or another team in the org, OR
    - Have authenticated via an SSO connection owned by the org

    Member is identified by personal user_id or nickname.
    Requires the caller to be an org admin.

    Args:
        team_user_str (str):
        member_user_str (str):
        body (BodyOrgAddTeamMember):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        team_user_str=team_user_str,
        member_user_str=member_user_str,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    team_user_str: str,
    member_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: BodyOrgAddTeamMember,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Org Add Team Member

     Add a member to a team.

    Org admins can add members to teams within their organization.
    The member must either:
    - Already be part of the org or another team in the org, OR
    - Have authenticated via an SSO connection owned by the org

    Member is identified by personal user_id or nickname.
    Requires the caller to be an org admin.

    Args:
        team_user_str (str):
        member_user_str (str):
        body (BodyOrgAddTeamMember):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return sync_detailed(
        team_user_str=team_user_str,
        member_user_str=member_user_str,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    team_user_str: str,
    member_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: BodyOrgAddTeamMember,
) -> Response[Union[Any, HTTPValidationError]]:
    """Org Add Team Member

     Add a member to a team.

    Org admins can add members to teams within their organization.
    The member must either:
    - Already be part of the org or another team in the org, OR
    - Have authenticated via an SSO connection owned by the org

    Member is identified by personal user_id or nickname.
    Requires the caller to be an org admin.

    Args:
        team_user_str (str):
        member_user_str (str):
        body (BodyOrgAddTeamMember):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        team_user_str=team_user_str,
        member_user_str=member_user_str,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    team_user_str: str,
    member_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: BodyOrgAddTeamMember,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Org Add Team Member

     Add a member to a team.

    Org admins can add members to teams within their organization.
    The member must either:
    - Already be part of the org or another team in the org, OR
    - Have authenticated via an SSO connection owned by the org

    Member is identified by personal user_id or nickname.
    Requires the caller to be an org admin.

    Args:
        team_user_str (str):
        member_user_str (str):
        body (BodyOrgAddTeamMember):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            team_user_str=team_user_str,
            member_user_str=member_user_str,
            client=client,
            body=body,
        )
    ).parsed
