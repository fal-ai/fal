from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.organization_settings import OrganizationSettings
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    org_level: Union[Unset, bool] = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["org_level"] = org_level

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/organizations/organization-settings",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, OrganizationSettings]]:
    if response.status_code == 200:
        response_200 = OrganizationSettings.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, OrganizationSettings]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    org_level: Union[Unset, bool] = False,
) -> Response[Union[HTTPValidationError, OrganizationSettings]]:
    """Get Organization Settings

     Get organization settings for the current team.
    Returns default values if no settings have been configured.

    Args:
        org_level (Union[Unset, bool]): When true, read settings from the parent org level instead
            of the current team. Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, OrganizationSettings]]
    """

    kwargs = _get_kwargs(
        org_level=org_level,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    org_level: Union[Unset, bool] = False,
) -> Optional[Union[HTTPValidationError, OrganizationSettings]]:
    """Get Organization Settings

     Get organization settings for the current team.
    Returns default values if no settings have been configured.

    Args:
        org_level (Union[Unset, bool]): When true, read settings from the parent org level instead
            of the current team. Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, OrganizationSettings]
    """

    return sync_detailed(
        client=client,
        org_level=org_level,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    org_level: Union[Unset, bool] = False,
) -> Response[Union[HTTPValidationError, OrganizationSettings]]:
    """Get Organization Settings

     Get organization settings for the current team.
    Returns default values if no settings have been configured.

    Args:
        org_level (Union[Unset, bool]): When true, read settings from the parent org level instead
            of the current team. Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, OrganizationSettings]]
    """

    kwargs = _get_kwargs(
        org_level=org_level,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    org_level: Union[Unset, bool] = False,
) -> Optional[Union[HTTPValidationError, OrganizationSettings]]:
    """Get Organization Settings

     Get organization settings for the current team.
    Returns default values if no settings have been configured.

    Args:
        org_level (Union[Unset, bool]): When true, read settings from the parent org level instead
            of the current team. Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, OrganizationSettings]
    """

    return (
        await asyncio_detailed(
            client=client,
            org_level=org_level,
        )
    ).parsed
