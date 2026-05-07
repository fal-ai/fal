from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.serverless_billing_override_definition import ServerlessBillingOverrideDefinition
from ...types import Response


def _get_kwargs(
    *,
    body: ServerlessBillingOverrideDefinition,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/admin/serverless/billing_overrides",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, ServerlessBillingOverrideDefinition]]:
    if response.status_code == 200:
        response_200 = ServerlessBillingOverrideDefinition.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, ServerlessBillingOverrideDefinition]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: ServerlessBillingOverrideDefinition,
) -> Response[Union[HTTPValidationError, ServerlessBillingOverrideDefinition]]:
    """Create Serverless Billing Override

    Args:
        body (ServerlessBillingOverrideDefinition):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, ServerlessBillingOverrideDefinition]]
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
    body: ServerlessBillingOverrideDefinition,
) -> Optional[Union[HTTPValidationError, ServerlessBillingOverrideDefinition]]:
    """Create Serverless Billing Override

    Args:
        body (ServerlessBillingOverrideDefinition):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, ServerlessBillingOverrideDefinition]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: ServerlessBillingOverrideDefinition,
) -> Response[Union[HTTPValidationError, ServerlessBillingOverrideDefinition]]:
    """Create Serverless Billing Override

    Args:
        body (ServerlessBillingOverrideDefinition):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, ServerlessBillingOverrideDefinition]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: ServerlessBillingOverrideDefinition,
) -> Optional[Union[HTTPValidationError, ServerlessBillingOverrideDefinition]]:
    """Create Serverless Billing Override

    Args:
        body (ServerlessBillingOverrideDefinition):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, ServerlessBillingOverrideDefinition]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
