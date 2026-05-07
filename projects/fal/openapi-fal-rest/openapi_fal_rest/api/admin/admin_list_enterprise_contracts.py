import datetime
from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.contract_status import ContractStatus
from ...models.enterprise_contract import EnterpriseContract
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    user_id: Union[Unset, str] = UNSET,
    user_ids: Union[Unset, list[str]] = UNSET,
    status: Union[Unset, ContractStatus] = UNSET,
    owner: Union[Unset, str] = UNSET,
    starts_at: Union[Unset, datetime.datetime] = UNSET,
    ends_at: Union[Unset, datetime.datetime] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["user_id"] = user_id

    json_user_ids: Union[Unset, list[str]] = UNSET
    if not isinstance(user_ids, Unset):
        json_user_ids = user_ids

    params["user_ids"] = json_user_ids

    json_status: Union[Unset, str] = UNSET
    if not isinstance(status, Unset):
        json_status = status.value

    params["status"] = json_status

    params["owner"] = owner

    json_starts_at: Union[Unset, str] = UNSET
    if not isinstance(starts_at, Unset):
        json_starts_at = starts_at.isoformat()
    params["starts_at"] = json_starts_at

    json_ends_at: Union[Unset, str] = UNSET
    if not isinstance(ends_at, Unset):
        json_ends_at = ends_at.isoformat()
    params["ends_at"] = json_ends_at

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/admin/enterprise_contracts",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, list["EnterpriseContract"]]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = EnterpriseContract.from_dict(response_200_item_data)

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
) -> Response[Union[HTTPValidationError, list["EnterpriseContract"]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: Union[Unset, str] = UNSET,
    user_ids: Union[Unset, list[str]] = UNSET,
    status: Union[Unset, ContractStatus] = UNSET,
    owner: Union[Unset, str] = UNSET,
    starts_at: Union[Unset, datetime.datetime] = UNSET,
    ends_at: Union[Unset, datetime.datetime] = UNSET,
) -> Response[Union[HTTPValidationError, list["EnterpriseContract"]]]:
    """List Enterprise Contracts

     List all enterprise contracts, optionally filtered by user_id(s) or status.

    Args:
        user_id (Union[Unset, str]):
        user_ids (Union[Unset, list[str]]): Filter by multiple user_ids
        status (Union[Unset, ContractStatus]):
        owner (Union[Unset, str]):
        starts_at (Union[Unset, datetime.datetime]):
        ends_at (Union[Unset, datetime.datetime]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['EnterpriseContract']]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        user_ids=user_ids,
        status=status,
        owner=owner,
        starts_at=starts_at,
        ends_at=ends_at,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: Union[Unset, str] = UNSET,
    user_ids: Union[Unset, list[str]] = UNSET,
    status: Union[Unset, ContractStatus] = UNSET,
    owner: Union[Unset, str] = UNSET,
    starts_at: Union[Unset, datetime.datetime] = UNSET,
    ends_at: Union[Unset, datetime.datetime] = UNSET,
) -> Optional[Union[HTTPValidationError, list["EnterpriseContract"]]]:
    """List Enterprise Contracts

     List all enterprise contracts, optionally filtered by user_id(s) or status.

    Args:
        user_id (Union[Unset, str]):
        user_ids (Union[Unset, list[str]]): Filter by multiple user_ids
        status (Union[Unset, ContractStatus]):
        owner (Union[Unset, str]):
        starts_at (Union[Unset, datetime.datetime]):
        ends_at (Union[Unset, datetime.datetime]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['EnterpriseContract']]
    """

    return sync_detailed(
        client=client,
        user_id=user_id,
        user_ids=user_ids,
        status=status,
        owner=owner,
        starts_at=starts_at,
        ends_at=ends_at,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: Union[Unset, str] = UNSET,
    user_ids: Union[Unset, list[str]] = UNSET,
    status: Union[Unset, ContractStatus] = UNSET,
    owner: Union[Unset, str] = UNSET,
    starts_at: Union[Unset, datetime.datetime] = UNSET,
    ends_at: Union[Unset, datetime.datetime] = UNSET,
) -> Response[Union[HTTPValidationError, list["EnterpriseContract"]]]:
    """List Enterprise Contracts

     List all enterprise contracts, optionally filtered by user_id(s) or status.

    Args:
        user_id (Union[Unset, str]):
        user_ids (Union[Unset, list[str]]): Filter by multiple user_ids
        status (Union[Unset, ContractStatus]):
        owner (Union[Unset, str]):
        starts_at (Union[Unset, datetime.datetime]):
        ends_at (Union[Unset, datetime.datetime]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['EnterpriseContract']]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        user_ids=user_ids,
        status=status,
        owner=owner,
        starts_at=starts_at,
        ends_at=ends_at,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: Union[Unset, str] = UNSET,
    user_ids: Union[Unset, list[str]] = UNSET,
    status: Union[Unset, ContractStatus] = UNSET,
    owner: Union[Unset, str] = UNSET,
    starts_at: Union[Unset, datetime.datetime] = UNSET,
    ends_at: Union[Unset, datetime.datetime] = UNSET,
) -> Optional[Union[HTTPValidationError, list["EnterpriseContract"]]]:
    """List Enterprise Contracts

     List all enterprise contracts, optionally filtered by user_id(s) or status.

    Args:
        user_id (Union[Unset, str]):
        user_ids (Union[Unset, list[str]]): Filter by multiple user_ids
        status (Union[Unset, ContractStatus]):
        owner (Union[Unset, str]):
        starts_at (Union[Unset, datetime.datetime]):
        ends_at (Union[Unset, datetime.datetime]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['EnterpriseContract']]
    """

    return (
        await asyncio_detailed(
            client=client,
            user_id=user_id,
            user_ids=user_ids,
            status=status,
            owner=owner,
            starts_at=starts_at,
            ends_at=ends_at,
        )
    ).parsed
