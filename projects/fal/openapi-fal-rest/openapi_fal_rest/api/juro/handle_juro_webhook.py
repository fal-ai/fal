from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.handle_juro_webhook_response_handle_juro_webhook import HandleJuroWebhookResponseHandleJuroWebhook
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    x_juro_signature: Union[Unset, str] = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(x_juro_signature, Unset):
        headers["x-juro-signature"] = x_juro_signature

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/juro/webhook",
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, HandleJuroWebhookResponseHandleJuroWebhook]]:
    if response.status_code == 200:
        response_200 = HandleJuroWebhookResponseHandleJuroWebhook.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, HandleJuroWebhookResponseHandleJuroWebhook]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    x_juro_signature: Union[Unset, str] = UNSET,
) -> Response[Union[HTTPValidationError, HandleJuroWebhookResponseHandleJuroWebhook]]:
    """Handle Juro Webhook

    Args:
        x_juro_signature (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, HandleJuroWebhookResponseHandleJuroWebhook]]
    """

    kwargs = _get_kwargs(
        x_juro_signature=x_juro_signature,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    x_juro_signature: Union[Unset, str] = UNSET,
) -> Optional[Union[HTTPValidationError, HandleJuroWebhookResponseHandleJuroWebhook]]:
    """Handle Juro Webhook

    Args:
        x_juro_signature (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, HandleJuroWebhookResponseHandleJuroWebhook]
    """

    return sync_detailed(
        client=client,
        x_juro_signature=x_juro_signature,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    x_juro_signature: Union[Unset, str] = UNSET,
) -> Response[Union[HTTPValidationError, HandleJuroWebhookResponseHandleJuroWebhook]]:
    """Handle Juro Webhook

    Args:
        x_juro_signature (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, HandleJuroWebhookResponseHandleJuroWebhook]]
    """

    kwargs = _get_kwargs(
        x_juro_signature=x_juro_signature,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    x_juro_signature: Union[Unset, str] = UNSET,
) -> Optional[Union[HTTPValidationError, HandleJuroWebhookResponseHandleJuroWebhook]]:
    """Handle Juro Webhook

    Args:
        x_juro_signature (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, HandleJuroWebhookResponseHandleJuroWebhook]
    """

    return (
        await asyncio_detailed(
            client=client,
            x_juro_signature=x_juro_signature,
        )
    ).parsed
