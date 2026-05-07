from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.org_spending_alert_v2_response import OrgSpendingAlertV2Response
from ...models.spending_alert_v2_create import SpendingAlertV2Create
from ...types import Response


def _get_kwargs(
    team_user_str: str,
    *,
    body: SpendingAlertV2Create,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/organizations/teams/{team_user_str}/spending-alerts-v2",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, OrgSpendingAlertV2Response]]:
    if response.status_code == 201:
        response_201 = OrgSpendingAlertV2Response.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, OrgSpendingAlertV2Response]]:
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
    body: SpendingAlertV2Create,
) -> Response[Union[HTTPValidationError, OrgSpendingAlertV2Response]]:
    """Create Org Team Spending Alert V2

    Args:
        team_user_str (str):
        body (SpendingAlertV2Create):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, OrgSpendingAlertV2Response]]
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
    body: SpendingAlertV2Create,
) -> Optional[Union[HTTPValidationError, OrgSpendingAlertV2Response]]:
    """Create Org Team Spending Alert V2

    Args:
        team_user_str (str):
        body (SpendingAlertV2Create):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, OrgSpendingAlertV2Response]
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
    body: SpendingAlertV2Create,
) -> Response[Union[HTTPValidationError, OrgSpendingAlertV2Response]]:
    """Create Org Team Spending Alert V2

    Args:
        team_user_str (str):
        body (SpendingAlertV2Create):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, OrgSpendingAlertV2Response]]
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
    body: SpendingAlertV2Create,
) -> Optional[Union[HTTPValidationError, OrgSpendingAlertV2Response]]:
    """Create Org Team Spending Alert V2

    Args:
        team_user_str (str):
        body (SpendingAlertV2Create):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, OrgSpendingAlertV2Response]
    """

    return (
        await asyncio_detailed(
            team_user_str=team_user_str,
            client=client,
            body=body,
        )
    ).parsed
