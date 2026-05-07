from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.page_org_user_row import PageOrgUserRow
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    query: Union[Unset, str] = UNSET,
    team_user_str: Union[Unset, str] = UNSET,
    status: Union[Unset, Any] = UNSET,
    role: Union[Unset, Any] = UNSET,
    include_archived: Union[Unset, bool] = False,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["query"] = query

    params["team_user_str"] = team_user_str

    params["status"] = status

    params["role"] = role

    params["include_archived"] = include_archived

    params["page"] = page

    params["size"] = size

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/organizations/users",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, PageOrgUserRow]]:
    if response.status_code == 200:
        response_200 = PageOrgUserRow.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, PageOrgUserRow]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    query: Union[Unset, str] = UNSET,
    team_user_str: Union[Unset, str] = UNSET,
    status: Union[Unset, Any] = UNSET,
    role: Union[Unset, Any] = UNSET,
    include_archived: Union[Unset, bool] = False,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Response[Union[HTTPValidationError, PageOrgUserRow]]:
    """List Org Users

     List all users of the organization, grouped by email.

    Includes active members, pending invites, and unassigned SSO users
    (users who authenticated via org's owned SSO connections but have no
    team assignments or pending invites). Use the status filter to show
    only a specific category. Use the role filter to show only users with
    a specific role (not applicable to unassigned users).

    By default, excludes team assignments and invites for archived teams.
    Set include_archived=true to include them.

    Args:
        query (Union[Unset, str]): Search by email or name
        team_user_str (Union[Unset, str]): Filter by team (user_id or nickname)
        status (Union[Unset, Any]): Filter by status: active (has membership), invite_pending, or
            unassigned (SSO user without assignments)
        role (Union[Unset, Any]): Filter by role (Admin, Developer, Billing)
        include_archived (Union[Unset, bool]): Include team assignments and invites for archived
            teams Default: False.
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, PageOrgUserRow]]
    """

    kwargs = _get_kwargs(
        query=query,
        team_user_str=team_user_str,
        status=status,
        role=role,
        include_archived=include_archived,
        page=page,
        size=size,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    query: Union[Unset, str] = UNSET,
    team_user_str: Union[Unset, str] = UNSET,
    status: Union[Unset, Any] = UNSET,
    role: Union[Unset, Any] = UNSET,
    include_archived: Union[Unset, bool] = False,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Optional[Union[HTTPValidationError, PageOrgUserRow]]:
    """List Org Users

     List all users of the organization, grouped by email.

    Includes active members, pending invites, and unassigned SSO users
    (users who authenticated via org's owned SSO connections but have no
    team assignments or pending invites). Use the status filter to show
    only a specific category. Use the role filter to show only users with
    a specific role (not applicable to unassigned users).

    By default, excludes team assignments and invites for archived teams.
    Set include_archived=true to include them.

    Args:
        query (Union[Unset, str]): Search by email or name
        team_user_str (Union[Unset, str]): Filter by team (user_id or nickname)
        status (Union[Unset, Any]): Filter by status: active (has membership), invite_pending, or
            unassigned (SSO user without assignments)
        role (Union[Unset, Any]): Filter by role (Admin, Developer, Billing)
        include_archived (Union[Unset, bool]): Include team assignments and invites for archived
            teams Default: False.
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, PageOrgUserRow]
    """

    return sync_detailed(
        client=client,
        query=query,
        team_user_str=team_user_str,
        status=status,
        role=role,
        include_archived=include_archived,
        page=page,
        size=size,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    query: Union[Unset, str] = UNSET,
    team_user_str: Union[Unset, str] = UNSET,
    status: Union[Unset, Any] = UNSET,
    role: Union[Unset, Any] = UNSET,
    include_archived: Union[Unset, bool] = False,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Response[Union[HTTPValidationError, PageOrgUserRow]]:
    """List Org Users

     List all users of the organization, grouped by email.

    Includes active members, pending invites, and unassigned SSO users
    (users who authenticated via org's owned SSO connections but have no
    team assignments or pending invites). Use the status filter to show
    only a specific category. Use the role filter to show only users with
    a specific role (not applicable to unassigned users).

    By default, excludes team assignments and invites for archived teams.
    Set include_archived=true to include them.

    Args:
        query (Union[Unset, str]): Search by email or name
        team_user_str (Union[Unset, str]): Filter by team (user_id or nickname)
        status (Union[Unset, Any]): Filter by status: active (has membership), invite_pending, or
            unassigned (SSO user without assignments)
        role (Union[Unset, Any]): Filter by role (Admin, Developer, Billing)
        include_archived (Union[Unset, bool]): Include team assignments and invites for archived
            teams Default: False.
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, PageOrgUserRow]]
    """

    kwargs = _get_kwargs(
        query=query,
        team_user_str=team_user_str,
        status=status,
        role=role,
        include_archived=include_archived,
        page=page,
        size=size,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    query: Union[Unset, str] = UNSET,
    team_user_str: Union[Unset, str] = UNSET,
    status: Union[Unset, Any] = UNSET,
    role: Union[Unset, Any] = UNSET,
    include_archived: Union[Unset, bool] = False,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Optional[Union[HTTPValidationError, PageOrgUserRow]]:
    """List Org Users

     List all users of the organization, grouped by email.

    Includes active members, pending invites, and unassigned SSO users
    (users who authenticated via org's owned SSO connections but have no
    team assignments or pending invites). Use the status filter to show
    only a specific category. Use the role filter to show only users with
    a specific role (not applicable to unassigned users).

    By default, excludes team assignments and invites for archived teams.
    Set include_archived=true to include them.

    Args:
        query (Union[Unset, str]): Search by email or name
        team_user_str (Union[Unset, str]): Filter by team (user_id or nickname)
        status (Union[Unset, Any]): Filter by status: active (has membership), invite_pending, or
            unassigned (SSO user without assignments)
        role (Union[Unset, Any]): Filter by role (Admin, Developer, Billing)
        include_archived (Union[Unset, bool]): Include team assignments and invites for archived
            teams Default: False.
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, PageOrgUserRow]
    """

    return (
        await asyncio_detailed(
            client=client,
            query=query,
            team_user_str=team_user_str,
            status=status,
            role=role,
            include_archived=include_archived,
            page=page,
            size=size,
        )
    ).parsed
