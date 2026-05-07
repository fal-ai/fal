import datetime
from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.org_payment_receipt import OrgPaymentReceipt
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    start_date: datetime.date,
    team_user_str: Union[Unset, str] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_start_date = start_date.isoformat()
    params["start_date"] = json_start_date

    params["team_user_str"] = team_user_str

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/organizations/receipts",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, list["OrgPaymentReceipt"]]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = OrgPaymentReceipt.from_dict(response_200_item_data)

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
) -> Response[Union[HTTPValidationError, list["OrgPaymentReceipt"]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    start_date: datetime.date,
    team_user_str: Union[Unset, str] = UNSET,
) -> Response[Union[HTTPValidationError, list["OrgPaymentReceipt"]]]:
    """Get Org Receipts Endpoint

     Get payment receipts for the organization. Requires org admin or billing role.

    Args:
        start_date (datetime.date):
        team_user_str (Union[Unset, str]): Filter by team (user_id or nickname)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['OrgPaymentReceipt']]]
    """

    kwargs = _get_kwargs(
        start_date=start_date,
        team_user_str=team_user_str,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    start_date: datetime.date,
    team_user_str: Union[Unset, str] = UNSET,
) -> Optional[Union[HTTPValidationError, list["OrgPaymentReceipt"]]]:
    """Get Org Receipts Endpoint

     Get payment receipts for the organization. Requires org admin or billing role.

    Args:
        start_date (datetime.date):
        team_user_str (Union[Unset, str]): Filter by team (user_id or nickname)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['OrgPaymentReceipt']]
    """

    return sync_detailed(
        client=client,
        start_date=start_date,
        team_user_str=team_user_str,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    start_date: datetime.date,
    team_user_str: Union[Unset, str] = UNSET,
) -> Response[Union[HTTPValidationError, list["OrgPaymentReceipt"]]]:
    """Get Org Receipts Endpoint

     Get payment receipts for the organization. Requires org admin or billing role.

    Args:
        start_date (datetime.date):
        team_user_str (Union[Unset, str]): Filter by team (user_id or nickname)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['OrgPaymentReceipt']]]
    """

    kwargs = _get_kwargs(
        start_date=start_date,
        team_user_str=team_user_str,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    start_date: datetime.date,
    team_user_str: Union[Unset, str] = UNSET,
) -> Optional[Union[HTTPValidationError, list["OrgPaymentReceipt"]]]:
    """Get Org Receipts Endpoint

     Get payment receipts for the organization. Requires org admin or billing role.

    Args:
        start_date (datetime.date):
        team_user_str (Union[Unset, str]): Filter by team (user_id or nickname)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['OrgPaymentReceipt']]
    """

    return (
        await asyncio_detailed(
            client=client,
            start_date=start_date,
            team_user_str=team_user_str,
        )
    ).parsed
