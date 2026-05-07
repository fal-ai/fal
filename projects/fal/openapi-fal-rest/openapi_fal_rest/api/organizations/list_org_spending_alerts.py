from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.org_spending_alert_response import OrgSpendingAlertResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    team_user_str: Union[Unset, str] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["team_user_str"] = team_user_str

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/organizations/spending-alerts",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, list["OrgSpendingAlertResponse"]]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = OrgSpendingAlertResponse.from_dict(response_200_item_data)

            response_200.append(response_200_item)

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
) -> Response[Union[HTTPValidationError, list["OrgSpendingAlertResponse"]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    team_user_str: Union[Unset, str] = UNSET,
) -> Response[Union[HTTPValidationError, list["OrgSpendingAlertResponse"]]]:
    """List Org Spending Alerts

     List all spending alerts for the organization and its teams.

    Org admins can view spending alerts across all teams in their organization.
    Use the team_user_str filter to view alerts for a specific team.
    Requires the caller to be an org admin.
    Requires spending_alerts to be enabled at the org level.

    Args:
        team_user_str (Union[Unset, str]): Filter by team (user_id or nickname)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['OrgSpendingAlertResponse']]]
    """

    kwargs = _get_kwargs(
        team_user_str=team_user_str,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    team_user_str: Union[Unset, str] = UNSET,
) -> Optional[Union[HTTPValidationError, list["OrgSpendingAlertResponse"]]]:
    """List Org Spending Alerts

     List all spending alerts for the organization and its teams.

    Org admins can view spending alerts across all teams in their organization.
    Use the team_user_str filter to view alerts for a specific team.
    Requires the caller to be an org admin.
    Requires spending_alerts to be enabled at the org level.

    Args:
        team_user_str (Union[Unset, str]): Filter by team (user_id or nickname)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['OrgSpendingAlertResponse']]
    """

    return sync_detailed(
        client=client,
        team_user_str=team_user_str,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    team_user_str: Union[Unset, str] = UNSET,
) -> Response[Union[HTTPValidationError, list["OrgSpendingAlertResponse"]]]:
    """List Org Spending Alerts

     List all spending alerts for the organization and its teams.

    Org admins can view spending alerts across all teams in their organization.
    Use the team_user_str filter to view alerts for a specific team.
    Requires the caller to be an org admin.
    Requires spending_alerts to be enabled at the org level.

    Args:
        team_user_str (Union[Unset, str]): Filter by team (user_id or nickname)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['OrgSpendingAlertResponse']]]
    """

    kwargs = _get_kwargs(
        team_user_str=team_user_str,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    team_user_str: Union[Unset, str] = UNSET,
) -> Optional[Union[HTTPValidationError, list["OrgSpendingAlertResponse"]]]:
    """List Org Spending Alerts

     List all spending alerts for the organization and its teams.

    Org admins can view spending alerts across all teams in their organization.
    Use the team_user_str filter to view alerts for a specific team.
    Requires the caller to be an org admin.
    Requires spending_alerts to be enabled at the org level.

    Args:
        team_user_str (Union[Unset, str]): Filter by team (user_id or nickname)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['OrgSpendingAlertResponse']]
    """

    return (
        await asyncio_detailed(
            client=client,
            team_user_str=team_user_str,
        )
    ).parsed
