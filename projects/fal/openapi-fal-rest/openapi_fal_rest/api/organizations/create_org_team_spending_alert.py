from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.org_spending_alert_response import OrgSpendingAlertResponse
from ...models.spending_alert_create import SpendingAlertCreate
from ...types import Response


def _get_kwargs(
    team_user_str: str,
    *,
    body: SpendingAlertCreate,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/organizations/teams/{team_user_str}/spending-alerts",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, OrgSpendingAlertResponse]]:
    if response.status_code == 201:
        response_201 = OrgSpendingAlertResponse.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, OrgSpendingAlertResponse]]:
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
    body: SpendingAlertCreate,
) -> Response[Union[HTTPValidationError, OrgSpendingAlertResponse]]:
    """Create Org Team Spending Alert

     Create a spending alert for a team within the organization.

    Org admins can create spending alerts for any team within their organization.
    Team is identified by user_id or nickname.
    Requires the caller to be an org admin.
    Requires spending_alerts to be enabled at the org level.

    Args:
        team_user_str (str):
        body (SpendingAlertCreate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, OrgSpendingAlertResponse]]
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
    body: SpendingAlertCreate,
) -> Optional[Union[HTTPValidationError, OrgSpendingAlertResponse]]:
    """Create Org Team Spending Alert

     Create a spending alert for a team within the organization.

    Org admins can create spending alerts for any team within their organization.
    Team is identified by user_id or nickname.
    Requires the caller to be an org admin.
    Requires spending_alerts to be enabled at the org level.

    Args:
        team_user_str (str):
        body (SpendingAlertCreate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, OrgSpendingAlertResponse]
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
    body: SpendingAlertCreate,
) -> Response[Union[HTTPValidationError, OrgSpendingAlertResponse]]:
    """Create Org Team Spending Alert

     Create a spending alert for a team within the organization.

    Org admins can create spending alerts for any team within their organization.
    Team is identified by user_id or nickname.
    Requires the caller to be an org admin.
    Requires spending_alerts to be enabled at the org level.

    Args:
        team_user_str (str):
        body (SpendingAlertCreate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, OrgSpendingAlertResponse]]
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
    body: SpendingAlertCreate,
) -> Optional[Union[HTTPValidationError, OrgSpendingAlertResponse]]:
    """Create Org Team Spending Alert

     Create a spending alert for a team within the organization.

    Org admins can create spending alerts for any team within their organization.
    Team is identified by user_id or nickname.
    Requires the caller to be an org admin.
    Requires spending_alerts to be enabled at the org level.

    Args:
        team_user_str (str):
        body (SpendingAlertCreate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, OrgSpendingAlertResponse]
    """

    return (
        await asyncio_detailed(
            team_user_str=team_user_str,
            client=client,
            body=body,
        )
    ).parsed
