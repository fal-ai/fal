from http import HTTPStatus
from typing import Any, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.policy_role import PolicyRole
from ...models.team_role import TeamRole
from ...types import UNSET, Response, Unset


def _get_kwargs(
    member_auth_id: str,
    *,
    team_role: TeamRole,
    policy_role: Union[Unset, PolicyRole] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_team_role = team_role.value
    params["team_role"] = json_team_role

    json_policy_role: Union[Unset, str] = UNSET
    if not isinstance(policy_role, Unset):
        json_policy_role = policy_role.value

    params["policy_role"] = json_policy_role

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "patch",
        "url": f"/users/current/{member_auth_id}",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, str]]:
    if response.status_code == 200:
        response_200 = cast(str, response.json())
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
) -> Response[Union[HTTPValidationError, str]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    member_auth_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    team_role: TeamRole,
    policy_role: Union[Unset, PolicyRole] = UNSET,
) -> Response[Union[HTTPValidationError, str]]:
    """Update User Role

    Args:
        member_auth_id (str):
        team_role (TeamRole):
        policy_role (Union[Unset, PolicyRole]): Team member roles.

            Viewer and Creator serve different purposes and are not strict subsets
            of each other.  From Creator upward tiers are cumulative:
            Creator < Developer < Billing < Admin.
            Must match the DB `policy_role` enum values exactly.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, str]]
    """

    kwargs = _get_kwargs(
        member_auth_id=member_auth_id,
        team_role=team_role,
        policy_role=policy_role,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    member_auth_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    team_role: TeamRole,
    policy_role: Union[Unset, PolicyRole] = UNSET,
) -> Optional[Union[HTTPValidationError, str]]:
    """Update User Role

    Args:
        member_auth_id (str):
        team_role (TeamRole):
        policy_role (Union[Unset, PolicyRole]): Team member roles.

            Viewer and Creator serve different purposes and are not strict subsets
            of each other.  From Creator upward tiers are cumulative:
            Creator < Developer < Billing < Admin.
            Must match the DB `policy_role` enum values exactly.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, str]
    """

    return sync_detailed(
        member_auth_id=member_auth_id,
        client=client,
        team_role=team_role,
        policy_role=policy_role,
    ).parsed


async def asyncio_detailed(
    member_auth_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    team_role: TeamRole,
    policy_role: Union[Unset, PolicyRole] = UNSET,
) -> Response[Union[HTTPValidationError, str]]:
    """Update User Role

    Args:
        member_auth_id (str):
        team_role (TeamRole):
        policy_role (Union[Unset, PolicyRole]): Team member roles.

            Viewer and Creator serve different purposes and are not strict subsets
            of each other.  From Creator upward tiers are cumulative:
            Creator < Developer < Billing < Admin.
            Must match the DB `policy_role` enum values exactly.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, str]]
    """

    kwargs = _get_kwargs(
        member_auth_id=member_auth_id,
        team_role=team_role,
        policy_role=policy_role,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    member_auth_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    team_role: TeamRole,
    policy_role: Union[Unset, PolicyRole] = UNSET,
) -> Optional[Union[HTTPValidationError, str]]:
    """Update User Role

    Args:
        member_auth_id (str):
        team_role (TeamRole):
        policy_role (Union[Unset, PolicyRole]): Team member roles.

            Viewer and Creator serve different purposes and are not strict subsets
            of each other.  From Creator upward tiers are cumulative:
            Creator < Developer < Billing < Admin.
            Must match the DB `policy_role` enum values exactly.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, str]
    """

    return (
        await asyncio_detailed(
            member_auth_id=member_auth_id,
            client=client,
            team_role=team_role,
            policy_role=policy_role,
        )
    ).parsed
