from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.endpoint_access_control import EndpointAccessControl
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    user_str: str,
) -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/admin/organizations/{user_str}/endpoint_access_controls",
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, list["EndpointAccessControl"]]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = EndpointAccessControl.from_dict(response_200_item_data)

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
) -> Response[Union[HTTPValidationError, list["EndpointAccessControl"]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[HTTPValidationError, list["EndpointAccessControl"]]]:
    """Admin List Endpoint Access Controls

     Admin endpoint to list all endpoint access controls for a user/organization.
    Returns all records (both ALLOWED and BLOCKED status).
    Supports lookup by user_id or nickname.

    Args:
        user_str (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['EndpointAccessControl']]]
    """

    kwargs = _get_kwargs(
        user_str=user_str,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[HTTPValidationError, list["EndpointAccessControl"]]]:
    """Admin List Endpoint Access Controls

     Admin endpoint to list all endpoint access controls for a user/organization.
    Returns all records (both ALLOWED and BLOCKED status).
    Supports lookup by user_id or nickname.

    Args:
        user_str (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['EndpointAccessControl']]
    """

    return sync_detailed(
        user_str=user_str,
        client=client,
    ).parsed


async def asyncio_detailed(
    user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[HTTPValidationError, list["EndpointAccessControl"]]]:
    """Admin List Endpoint Access Controls

     Admin endpoint to list all endpoint access controls for a user/organization.
    Returns all records (both ALLOWED and BLOCKED status).
    Supports lookup by user_id or nickname.

    Args:
        user_str (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['EndpointAccessControl']]]
    """

    kwargs = _get_kwargs(
        user_str=user_str,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[HTTPValidationError, list["EndpointAccessControl"]]]:
    """Admin List Endpoint Access Controls

     Admin endpoint to list all endpoint access controls for a user/organization.
    Returns all records (both ALLOWED and BLOCKED status).
    Supports lookup by user_id or nickname.

    Args:
        user_str (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['EndpointAccessControl']]
    """

    return (
        await asyncio_detailed(
            user_str=user_str,
            client=client,
        )
    ).parsed
