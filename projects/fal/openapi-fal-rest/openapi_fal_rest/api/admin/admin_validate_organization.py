from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.organization_validation import OrganizationValidation
from ...types import Response


def _get_kwargs(
    org_user_str: str,
) -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/admin/organizations/{org_user_str}/validate",
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, OrganizationValidation]]:
    if response.status_code == 200:
        response_200 = OrganizationValidation.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, OrganizationValidation]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    org_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[HTTPValidationError, OrganizationValidation]]:
    """Validate Organization

     Validate organization structure and return validation results.
    Supports lookup by user_id or nickname.

    Args:
        org_user_str (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, OrganizationValidation]]
    """

    kwargs = _get_kwargs(
        org_user_str=org_user_str,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    org_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[HTTPValidationError, OrganizationValidation]]:
    """Validate Organization

     Validate organization structure and return validation results.
    Supports lookup by user_id or nickname.

    Args:
        org_user_str (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, OrganizationValidation]
    """

    return sync_detailed(
        org_user_str=org_user_str,
        client=client,
    ).parsed


async def asyncio_detailed(
    org_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[HTTPValidationError, OrganizationValidation]]:
    """Validate Organization

     Validate organization structure and return validation results.
    Supports lookup by user_id or nickname.

    Args:
        org_user_str (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, OrganizationValidation]]
    """

    kwargs = _get_kwargs(
        org_user_str=org_user_str,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    org_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[HTTPValidationError, OrganizationValidation]]:
    """Validate Organization

     Validate organization structure and return validation results.
    Supports lookup by user_id or nickname.

    Args:
        org_user_str (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, OrganizationValidation]
    """

    return (
        await asyncio_detailed(
            org_user_str=org_user_str,
            client=client,
        )
    ).parsed
