from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.endpoint_access_control_context import EndpointAccessControlContext
from ...models.endpoint_access_status import EndpointAccessStatus
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response


def _get_kwargs(
    *,
    endpoint: str,
    context: EndpointAccessControlContext,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["endpoint"] = endpoint

    json_context = context.value
    params["context"] = json_context

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/organizations/endpoint-access-controls/status",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[EndpointAccessStatus, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = EndpointAccessStatus.from_dict(response.json())

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
) -> Response[Union[EndpointAccessStatus, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    endpoint: str,
    context: EndpointAccessControlContext,
) -> Response[Union[EndpointAccessStatus, HTTPValidationError]]:
    """Get Endpoint Access Control

     Get access control status for a specific endpoint.

    Args:
        endpoint (str):
        context (EndpointAccessControlContext): API: Direct API requests using API keys
            UI: Web UI requests (Playground, Sandbox, Workflows) via proxy (X-Fal-Playground header)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[EndpointAccessStatus, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        endpoint=endpoint,
        context=context,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    endpoint: str,
    context: EndpointAccessControlContext,
) -> Optional[Union[EndpointAccessStatus, HTTPValidationError]]:
    """Get Endpoint Access Control

     Get access control status for a specific endpoint.

    Args:
        endpoint (str):
        context (EndpointAccessControlContext): API: Direct API requests using API keys
            UI: Web UI requests (Playground, Sandbox, Workflows) via proxy (X-Fal-Playground header)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[EndpointAccessStatus, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        endpoint=endpoint,
        context=context,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    endpoint: str,
    context: EndpointAccessControlContext,
) -> Response[Union[EndpointAccessStatus, HTTPValidationError]]:
    """Get Endpoint Access Control

     Get access control status for a specific endpoint.

    Args:
        endpoint (str):
        context (EndpointAccessControlContext): API: Direct API requests using API keys
            UI: Web UI requests (Playground, Sandbox, Workflows) via proxy (X-Fal-Playground header)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[EndpointAccessStatus, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        endpoint=endpoint,
        context=context,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    endpoint: str,
    context: EndpointAccessControlContext,
) -> Optional[Union[EndpointAccessStatus, HTTPValidationError]]:
    """Get Endpoint Access Control

     Get access control status for a specific endpoint.

    Args:
        endpoint (str):
        context (EndpointAccessControlContext): API: Direct API requests using API keys
            UI: Web UI requests (Playground, Sandbox, Workflows) via proxy (X-Fal-Playground header)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[EndpointAccessStatus, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            endpoint=endpoint,
            context=context,
        )
    ).parsed
