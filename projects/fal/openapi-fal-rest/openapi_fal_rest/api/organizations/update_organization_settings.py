from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.organization_settings import OrganizationSettings
from ...models.update_organization_settings_request import UpdateOrganizationSettingsRequest
from ...types import Response


def _get_kwargs(
    *,
    body: UpdateOrganizationSettingsRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/organizations/organization-settings",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
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
    body: UpdateOrganizationSettingsRequest,
) -> Response[Union[HTTPValidationError, OrganizationSettings]]:
    """Update Organization Settings

     Update organization settings for the current team.
    Only provided fields are updated; omitted fields retain their current values.
    Requires admin role.

    Args:
        body (UpdateOrganizationSettingsRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, OrganizationSettings]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    body: UpdateOrganizationSettingsRequest,
) -> Optional[Union[HTTPValidationError, OrganizationSettings]]:
    """Update Organization Settings

     Update organization settings for the current team.
    Only provided fields are updated; omitted fields retain their current values.
    Requires admin role.

    Args:
        body (UpdateOrganizationSettingsRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, OrganizationSettings]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: UpdateOrganizationSettingsRequest,
) -> Response[Union[HTTPValidationError, OrganizationSettings]]:
    """Update Organization Settings

     Update organization settings for the current team.
    Only provided fields are updated; omitted fields retain their current values.
    Requires admin role.

    Args:
        body (UpdateOrganizationSettingsRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, OrganizationSettings]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: UpdateOrganizationSettingsRequest,
) -> Optional[Union[HTTPValidationError, OrganizationSettings]]:
    """Update Organization Settings

     Update organization settings for the current team.
    Only provided fields are updated; omitted fields retain their current values.
    Requires admin role.

    Args:
        body (UpdateOrganizationSettingsRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, OrganizationSettings]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
