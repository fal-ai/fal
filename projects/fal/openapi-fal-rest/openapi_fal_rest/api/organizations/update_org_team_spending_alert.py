from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.org_spending_alert_response import OrgSpendingAlertResponse
from ...models.spending_alert_update import SpendingAlertUpdate
from ...types import Response


def _get_kwargs(
    team_user_str: str,
    alert_id: UUID,
    *,
    body: SpendingAlertUpdate,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "patch",
        "url": f"/organizations/teams/{team_user_str}/spending-alerts/{alert_id}",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, OrgSpendingAlertResponse]]:
    if response.status_code == 200:
        response_200 = OrgSpendingAlertResponse.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, OrgSpendingAlertResponse]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    team_user_str: str,
    alert_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    body: SpendingAlertUpdate,
) -> Response[Union[HTTPValidationError, OrgSpendingAlertResponse]]:
    """Update Org Team Spending Alert

     Update a spending alert for a team within the organization.

    Org admins can update spending alerts for any team within their organization.
    Team is identified by user_id or nickname.
    Requires the caller to be an org admin.

    Args:
        team_user_str (str):
        alert_id (UUID):
        body (SpendingAlertUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, OrgSpendingAlertResponse]]
    """

    kwargs = _get_kwargs(
        team_user_str=team_user_str,
        alert_id=alert_id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    team_user_str: str,
    alert_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    body: SpendingAlertUpdate,
) -> Optional[Union[HTTPValidationError, OrgSpendingAlertResponse]]:
    """Update Org Team Spending Alert

     Update a spending alert for a team within the organization.

    Org admins can update spending alerts for any team within their organization.
    Team is identified by user_id or nickname.
    Requires the caller to be an org admin.

    Args:
        team_user_str (str):
        alert_id (UUID):
        body (SpendingAlertUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, OrgSpendingAlertResponse]
    """

    return sync_detailed(
        team_user_str=team_user_str,
        alert_id=alert_id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    team_user_str: str,
    alert_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    body: SpendingAlertUpdate,
) -> Response[Union[HTTPValidationError, OrgSpendingAlertResponse]]:
    """Update Org Team Spending Alert

     Update a spending alert for a team within the organization.

    Org admins can update spending alerts for any team within their organization.
    Team is identified by user_id or nickname.
    Requires the caller to be an org admin.

    Args:
        team_user_str (str):
        alert_id (UUID):
        body (SpendingAlertUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, OrgSpendingAlertResponse]]
    """

    kwargs = _get_kwargs(
        team_user_str=team_user_str,
        alert_id=alert_id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    team_user_str: str,
    alert_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    body: SpendingAlertUpdate,
) -> Optional[Union[HTTPValidationError, OrgSpendingAlertResponse]]:
    """Update Org Team Spending Alert

     Update a spending alert for a team within the organization.

    Org admins can update spending alerts for any team within their organization.
    Team is identified by user_id or nickname.
    Requires the caller to be an org admin.

    Args:
        team_user_str (str):
        alert_id (UUID):
        body (SpendingAlertUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, OrgSpendingAlertResponse]
    """

    return (
        await asyncio_detailed(
            team_user_str=team_user_str,
            alert_id=alert_id,
            client=client,
            body=body,
        )
    ).parsed
