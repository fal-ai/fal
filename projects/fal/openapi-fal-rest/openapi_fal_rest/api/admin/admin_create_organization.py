from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.create_organization_request import CreateOrganizationRequest
from ...models.create_organization_response import CreateOrganizationResponse
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    *,
    body: CreateOrganizationRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/admin/organizations",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[CreateOrganizationResponse, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = CreateOrganizationResponse.from_dict(response.json())

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
) -> Response[Union[CreateOrganizationResponse, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: CreateOrganizationRequest,
) -> Response[Union[CreateOrganizationResponse, HTTPValidationError]]:
    """Create Organization

     Create a new organization.

    The nickname is auto-generated from the name to ensure uniqueness.
    Optionally provide admin_user_str (nickname or id) to set an initial org admin.
    Optionally provide auto_control_auth_provider (SSO connection ID) for the organization.

    Args:
        body (CreateOrganizationRequest): Request to create a new organization.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[CreateOrganizationResponse, HTTPValidationError]]
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
    body: CreateOrganizationRequest,
) -> Optional[Union[CreateOrganizationResponse, HTTPValidationError]]:
    """Create Organization

     Create a new organization.

    The nickname is auto-generated from the name to ensure uniqueness.
    Optionally provide admin_user_str (nickname or id) to set an initial org admin.
    Optionally provide auto_control_auth_provider (SSO connection ID) for the organization.

    Args:
        body (CreateOrganizationRequest): Request to create a new organization.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[CreateOrganizationResponse, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: CreateOrganizationRequest,
) -> Response[Union[CreateOrganizationResponse, HTTPValidationError]]:
    """Create Organization

     Create a new organization.

    The nickname is auto-generated from the name to ensure uniqueness.
    Optionally provide admin_user_str (nickname or id) to set an initial org admin.
    Optionally provide auto_control_auth_provider (SSO connection ID) for the organization.

    Args:
        body (CreateOrganizationRequest): Request to create a new organization.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[CreateOrganizationResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: CreateOrganizationRequest,
) -> Optional[Union[CreateOrganizationResponse, HTTPValidationError]]:
    """Create Organization

     Create a new organization.

    The nickname is auto-generated from the name to ensure uniqueness.
    Optionally provide admin_user_str (nickname or id) to set an initial org admin.
    Optionally provide auto_control_auth_provider (SSO connection ID) for the organization.

    Args:
        body (CreateOrganizationRequest): Request to create a new organization.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[CreateOrganizationResponse, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
