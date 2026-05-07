from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.body_admin_add_team import BodyAdminAddTeam
from ...models.http_validation_error import HTTPValidationError
from ...models.team_action_result import TeamActionResult
from ...types import Response


def _get_kwargs(
    org_user_str: str,
    *,
    body: BodyAdminAddTeam,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/admin/organizations/{org_user_str}/actions/add-team",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, TeamActionResult]]:
    if response.status_code == 200:
        response_200 = TeamActionResult.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, TeamActionResult]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    org_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: BodyAdminAddTeam,
) -> Response[Union[HTTPValidationError, TeamActionResult]]:
    """Add Team

     Create and add a new team to the organization.
    Supports lookup by user_id or nickname.

    The team will be created with a billing customer and locked with TOP_UP reason.
    Rate limiting is disabled for admin endpoints.

    Args:
        org_user_str: Organization user_id or nickname.
        team_user: TeamUser with the team's full_name.
        auto_control_auth_provider: Optional SSO connection ID for the team.
        admin_user_str: Optional user_id or nickname to add as team admin.
            If provided, the user's personal_auth_id will be used.

    Args:
        org_user_str (str):
        body (BodyAdminAddTeam):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, TeamActionResult]]
    """

    kwargs = _get_kwargs(
        org_user_str=org_user_str,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    org_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: BodyAdminAddTeam,
) -> Optional[Union[HTTPValidationError, TeamActionResult]]:
    """Add Team

     Create and add a new team to the organization.
    Supports lookup by user_id or nickname.

    The team will be created with a billing customer and locked with TOP_UP reason.
    Rate limiting is disabled for admin endpoints.

    Args:
        org_user_str: Organization user_id or nickname.
        team_user: TeamUser with the team's full_name.
        auto_control_auth_provider: Optional SSO connection ID for the team.
        admin_user_str: Optional user_id or nickname to add as team admin.
            If provided, the user's personal_auth_id will be used.

    Args:
        org_user_str (str):
        body (BodyAdminAddTeam):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, TeamActionResult]
    """

    return sync_detailed(
        org_user_str=org_user_str,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    org_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: BodyAdminAddTeam,
) -> Response[Union[HTTPValidationError, TeamActionResult]]:
    """Add Team

     Create and add a new team to the organization.
    Supports lookup by user_id or nickname.

    The team will be created with a billing customer and locked with TOP_UP reason.
    Rate limiting is disabled for admin endpoints.

    Args:
        org_user_str: Organization user_id or nickname.
        team_user: TeamUser with the team's full_name.
        auto_control_auth_provider: Optional SSO connection ID for the team.
        admin_user_str: Optional user_id or nickname to add as team admin.
            If provided, the user's personal_auth_id will be used.

    Args:
        org_user_str (str):
        body (BodyAdminAddTeam):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, TeamActionResult]]
    """

    kwargs = _get_kwargs(
        org_user_str=org_user_str,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    org_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: BodyAdminAddTeam,
) -> Optional[Union[HTTPValidationError, TeamActionResult]]:
    """Add Team

     Create and add a new team to the organization.
    Supports lookup by user_id or nickname.

    The team will be created with a billing customer and locked with TOP_UP reason.
    Rate limiting is disabled for admin endpoints.

    Args:
        org_user_str: Organization user_id or nickname.
        team_user: TeamUser with the team's full_name.
        auto_control_auth_provider: Optional SSO connection ID for the team.
        admin_user_str: Optional user_id or nickname to add as team admin.
            If provided, the user's personal_auth_id will be used.

    Args:
        org_user_str (str):
        body (BodyAdminAddTeam):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, TeamActionResult]
    """

    return (
        await asyncio_detailed(
            org_user_str=org_user_str,
            client=client,
            body=body,
        )
    ).parsed
