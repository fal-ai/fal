from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.org_spending_lock_response import OrgSpendingLockResponse
from ...models.spending_subject_type import SpendingSubjectType
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    team_user_str: Union[Unset, str] = UNSET,
    is_active: Union[Unset, bool] = UNSET,
    subject_type: Union[Unset, SpendingSubjectType] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["team_user_str"] = team_user_str

    params["is_active"] = is_active

    json_subject_type: Union[Unset, str] = UNSET
    if not isinstance(subject_type, Unset):
        json_subject_type = subject_type.value

    params["subject_type"] = json_subject_type

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/organizations/spending-locks",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, list["OrgSpendingLockResponse"]]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = OrgSpendingLockResponse.from_dict(response_200_item_data)

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
) -> Response[Union[HTTPValidationError, list["OrgSpendingLockResponse"]]]:
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
    is_active: Union[Unset, bool] = UNSET,
    subject_type: Union[Unset, SpendingSubjectType] = UNSET,
) -> Response[Union[HTTPValidationError, list["OrgSpendingLockResponse"]]]:
    """List Org Spending Locks

    Args:
        team_user_str (Union[Unset, str]): Filter by team (user_id or nickname)
        is_active (Union[Unset, bool]):
        subject_type (Union[Unset, SpendingSubjectType]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['OrgSpendingLockResponse']]]
    """

    kwargs = _get_kwargs(
        team_user_str=team_user_str,
        is_active=is_active,
        subject_type=subject_type,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    team_user_str: Union[Unset, str] = UNSET,
    is_active: Union[Unset, bool] = UNSET,
    subject_type: Union[Unset, SpendingSubjectType] = UNSET,
) -> Optional[Union[HTTPValidationError, list["OrgSpendingLockResponse"]]]:
    """List Org Spending Locks

    Args:
        team_user_str (Union[Unset, str]): Filter by team (user_id or nickname)
        is_active (Union[Unset, bool]):
        subject_type (Union[Unset, SpendingSubjectType]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['OrgSpendingLockResponse']]
    """

    return sync_detailed(
        client=client,
        team_user_str=team_user_str,
        is_active=is_active,
        subject_type=subject_type,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    team_user_str: Union[Unset, str] = UNSET,
    is_active: Union[Unset, bool] = UNSET,
    subject_type: Union[Unset, SpendingSubjectType] = UNSET,
) -> Response[Union[HTTPValidationError, list["OrgSpendingLockResponse"]]]:
    """List Org Spending Locks

    Args:
        team_user_str (Union[Unset, str]): Filter by team (user_id or nickname)
        is_active (Union[Unset, bool]):
        subject_type (Union[Unset, SpendingSubjectType]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['OrgSpendingLockResponse']]]
    """

    kwargs = _get_kwargs(
        team_user_str=team_user_str,
        is_active=is_active,
        subject_type=subject_type,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    team_user_str: Union[Unset, str] = UNSET,
    is_active: Union[Unset, bool] = UNSET,
    subject_type: Union[Unset, SpendingSubjectType] = UNSET,
) -> Optional[Union[HTTPValidationError, list["OrgSpendingLockResponse"]]]:
    """List Org Spending Locks

    Args:
        team_user_str (Union[Unset, str]): Filter by team (user_id or nickname)
        is_active (Union[Unset, bool]):
        subject_type (Union[Unset, SpendingSubjectType]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['OrgSpendingLockResponse']]
    """

    return (
        await asyncio_detailed(
            client=client,
            team_user_str=team_user_str,
            is_active=is_active,
            subject_type=subject_type,
        )
    ).parsed
