from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.organization_detail import OrganizationDetail
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    include_user_counts: Union[Unset, bool] = False,
    include_org_admins: Union[Unset, bool] = False,
    include_billing_info: Union[Unset, bool] = False,
    include_archived: Union[Unset, bool] = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["include_user_counts"] = include_user_counts

    params["include_org_admins"] = include_org_admins

    params["include_billing_info"] = include_billing_info

    params["include_archived"] = include_archived

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/organizations/current",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, OrganizationDetail]]:
    if response.status_code == 200:
        response_200 = OrganizationDetail.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, OrganizationDetail]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    include_user_counts: Union[Unset, bool] = False,
    include_org_admins: Union[Unset, bool] = False,
    include_billing_info: Union[Unset, bool] = False,
    include_archived: Union[Unset, bool] = False,
) -> Response[Union[HTTPValidationError, OrganizationDetail]]:
    """Get Current Organization

     Get organization details for the current org.

    Returns org info and teams.
    Requires the caller to be an org admin or billing role.

    Args:
        include_user_counts: When true, includes user_count for each team and
            total_user_count aggregate. Defaults to false for faster response.
        include_org_admins: When true, includes org_admins list.
            Defaults to false for faster response.
        include_billing_info: When true, includes is_invoicing for each team.
            Defaults to false for faster response.
        include_archived: When true, includes archived teams (lock_reason=ARCHIVED).
            Defaults to false.

    Args:
        include_user_counts (Union[Unset, bool]):  Default: False.
        include_org_admins (Union[Unset, bool]):  Default: False.
        include_billing_info (Union[Unset, bool]):  Default: False.
        include_archived (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, OrganizationDetail]]
    """

    kwargs = _get_kwargs(
        include_user_counts=include_user_counts,
        include_org_admins=include_org_admins,
        include_billing_info=include_billing_info,
        include_archived=include_archived,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    include_user_counts: Union[Unset, bool] = False,
    include_org_admins: Union[Unset, bool] = False,
    include_billing_info: Union[Unset, bool] = False,
    include_archived: Union[Unset, bool] = False,
) -> Optional[Union[HTTPValidationError, OrganizationDetail]]:
    """Get Current Organization

     Get organization details for the current org.

    Returns org info and teams.
    Requires the caller to be an org admin or billing role.

    Args:
        include_user_counts: When true, includes user_count for each team and
            total_user_count aggregate. Defaults to false for faster response.
        include_org_admins: When true, includes org_admins list.
            Defaults to false for faster response.
        include_billing_info: When true, includes is_invoicing for each team.
            Defaults to false for faster response.
        include_archived: When true, includes archived teams (lock_reason=ARCHIVED).
            Defaults to false.

    Args:
        include_user_counts (Union[Unset, bool]):  Default: False.
        include_org_admins (Union[Unset, bool]):  Default: False.
        include_billing_info (Union[Unset, bool]):  Default: False.
        include_archived (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, OrganizationDetail]
    """

    return sync_detailed(
        client=client,
        include_user_counts=include_user_counts,
        include_org_admins=include_org_admins,
        include_billing_info=include_billing_info,
        include_archived=include_archived,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    include_user_counts: Union[Unset, bool] = False,
    include_org_admins: Union[Unset, bool] = False,
    include_billing_info: Union[Unset, bool] = False,
    include_archived: Union[Unset, bool] = False,
) -> Response[Union[HTTPValidationError, OrganizationDetail]]:
    """Get Current Organization

     Get organization details for the current org.

    Returns org info and teams.
    Requires the caller to be an org admin or billing role.

    Args:
        include_user_counts: When true, includes user_count for each team and
            total_user_count aggregate. Defaults to false for faster response.
        include_org_admins: When true, includes org_admins list.
            Defaults to false for faster response.
        include_billing_info: When true, includes is_invoicing for each team.
            Defaults to false for faster response.
        include_archived: When true, includes archived teams (lock_reason=ARCHIVED).
            Defaults to false.

    Args:
        include_user_counts (Union[Unset, bool]):  Default: False.
        include_org_admins (Union[Unset, bool]):  Default: False.
        include_billing_info (Union[Unset, bool]):  Default: False.
        include_archived (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, OrganizationDetail]]
    """

    kwargs = _get_kwargs(
        include_user_counts=include_user_counts,
        include_org_admins=include_org_admins,
        include_billing_info=include_billing_info,
        include_archived=include_archived,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    include_user_counts: Union[Unset, bool] = False,
    include_org_admins: Union[Unset, bool] = False,
    include_billing_info: Union[Unset, bool] = False,
    include_archived: Union[Unset, bool] = False,
) -> Optional[Union[HTTPValidationError, OrganizationDetail]]:
    """Get Current Organization

     Get organization details for the current org.

    Returns org info and teams.
    Requires the caller to be an org admin or billing role.

    Args:
        include_user_counts: When true, includes user_count for each team and
            total_user_count aggregate. Defaults to false for faster response.
        include_org_admins: When true, includes org_admins list.
            Defaults to false for faster response.
        include_billing_info: When true, includes is_invoicing for each team.
            Defaults to false for faster response.
        include_archived: When true, includes archived teams (lock_reason=ARCHIVED).
            Defaults to false.

    Args:
        include_user_counts (Union[Unset, bool]):  Default: False.
        include_org_admins (Union[Unset, bool]):  Default: False.
        include_billing_info (Union[Unset, bool]):  Default: False.
        include_archived (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, OrganizationDetail]
    """

    return (
        await asyncio_detailed(
            client=client,
            include_user_counts=include_user_counts,
            include_org_admins=include_org_admins,
            include_billing_info=include_billing_info,
            include_archived=include_archived,
        )
    ).parsed
