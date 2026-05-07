import datetime
from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.contract_status import ContractStatus
from ...models.enterprise_contracts_sums import EnterpriseContractsSums
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    user_id: Union[Unset, str] = UNSET,
    status: Union[Unset, ContractStatus] = UNSET,
    owner: Union[Unset, str] = UNSET,
    starts_at: Union[Unset, datetime.datetime] = UNSET,
    ends_at: Union[Unset, datetime.datetime] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["user_id"] = user_id

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
        "url": "/admin/enterprise_contracts_sums",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[EnterpriseContractsSums, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = EnterpriseContractsSums.from_dict(response.json())

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
) -> Response[Union[EnterpriseContractsSums, HTTPValidationError]]:
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
    status: Union[Unset, ContractStatus] = UNSET,
    owner: Union[Unset, str] = UNSET,
    starts_at: Union[Unset, datetime.datetime] = UNSET,
    ends_at: Union[Unset, datetime.datetime] = UNSET,
) -> Response[Union[EnterpriseContractsSums, HTTPValidationError]]:
    """List Enterprise Contracts Sums

     Return sums of commitments for enterprise contracts, using the same filters
    as the paginated list endpoint.

    Args:
        user_id (Union[Unset, str]):
        status (Union[Unset, ContractStatus]):
        owner (Union[Unset, str]):
        starts_at (Union[Unset, datetime.datetime]):
        ends_at (Union[Unset, datetime.datetime]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[EnterpriseContractsSums, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
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
    status: Union[Unset, ContractStatus] = UNSET,
    owner: Union[Unset, str] = UNSET,
    starts_at: Union[Unset, datetime.datetime] = UNSET,
    ends_at: Union[Unset, datetime.datetime] = UNSET,
) -> Optional[Union[EnterpriseContractsSums, HTTPValidationError]]:
    """List Enterprise Contracts Sums

     Return sums of commitments for enterprise contracts, using the same filters
    as the paginated list endpoint.

    Args:
        user_id (Union[Unset, str]):
        status (Union[Unset, ContractStatus]):
        owner (Union[Unset, str]):
        starts_at (Union[Unset, datetime.datetime]):
        ends_at (Union[Unset, datetime.datetime]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[EnterpriseContractsSums, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        user_id=user_id,
        status=status,
        owner=owner,
        starts_at=starts_at,
        ends_at=ends_at,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: Union[Unset, str] = UNSET,
    status: Union[Unset, ContractStatus] = UNSET,
    owner: Union[Unset, str] = UNSET,
    starts_at: Union[Unset, datetime.datetime] = UNSET,
    ends_at: Union[Unset, datetime.datetime] = UNSET,
) -> Response[Union[EnterpriseContractsSums, HTTPValidationError]]:
    """List Enterprise Contracts Sums

     Return sums of commitments for enterprise contracts, using the same filters
    as the paginated list endpoint.

    Args:
        user_id (Union[Unset, str]):
        status (Union[Unset, ContractStatus]):
        owner (Union[Unset, str]):
        starts_at (Union[Unset, datetime.datetime]):
        ends_at (Union[Unset, datetime.datetime]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[EnterpriseContractsSums, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
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
    status: Union[Unset, ContractStatus] = UNSET,
    owner: Union[Unset, str] = UNSET,
    starts_at: Union[Unset, datetime.datetime] = UNSET,
    ends_at: Union[Unset, datetime.datetime] = UNSET,
) -> Optional[Union[EnterpriseContractsSums, HTTPValidationError]]:
    """List Enterprise Contracts Sums

     Return sums of commitments for enterprise contracts, using the same filters
    as the paginated list endpoint.

    Args:
        user_id (Union[Unset, str]):
        status (Union[Unset, ContractStatus]):
        owner (Union[Unset, str]):
        starts_at (Union[Unset, datetime.datetime]):
        ends_at (Union[Unset, datetime.datetime]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[EnterpriseContractsSums, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            user_id=user_id,
            status=status,
            owner=owner,
            starts_at=starts_at,
            ends_at=ends_at,
        )
    ).parsed
