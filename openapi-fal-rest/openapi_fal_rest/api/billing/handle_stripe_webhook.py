from http import HTTPStatus
from typing import Any, Dict, Optional, Union

import httpx

from ... import errors
from ...client import Client
from ...models.handle_stripe_webhook_response_handle_stripe_webhook import (
    HandleStripeWebhookResponseHandleStripeWebhook,
)
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    client: Client,
    stripe_signature: Union[Unset, str] = UNSET,
) -> Dict[str, Any]:
    url = "{}/billing/webhook".format(client.base_url)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    if not isinstance(stripe_signature, Unset):
        headers["stripe-signature"] = stripe_signature

    return {
        "method": "post",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "follow_redirects": client.follow_redirects,
    }


def _parse_response(
    *, client: Client, response: httpx.Response
) -> Optional[Union[HTTPValidationError, HandleStripeWebhookResponseHandleStripeWebhook]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = HandleStripeWebhookResponseHandleStripeWebhook.from_dict(response.json())

        return response_200
    if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Client, response: httpx.Response
) -> Response[Union[HTTPValidationError, HandleStripeWebhookResponseHandleStripeWebhook]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Client,
    stripe_signature: Union[Unset, str] = UNSET,
) -> Response[Union[HTTPValidationError, HandleStripeWebhookResponseHandleStripeWebhook]]:
    """Handle Stripe Webhook

    Args:
        stripe_signature (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, HandleStripeWebhookResponseHandleStripeWebhook]]
    """

    kwargs = _get_kwargs(
        client=client,
        stripe_signature=stripe_signature,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Client,
    stripe_signature: Union[Unset, str] = UNSET,
) -> Optional[Union[HTTPValidationError, HandleStripeWebhookResponseHandleStripeWebhook]]:
    """Handle Stripe Webhook

    Args:
        stripe_signature (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, HandleStripeWebhookResponseHandleStripeWebhook]
    """

    return sync_detailed(
        client=client,
        stripe_signature=stripe_signature,
    ).parsed


async def asyncio_detailed(
    *,
    client: Client,
    stripe_signature: Union[Unset, str] = UNSET,
) -> Response[Union[HTTPValidationError, HandleStripeWebhookResponseHandleStripeWebhook]]:
    """Handle Stripe Webhook

    Args:
        stripe_signature (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, HandleStripeWebhookResponseHandleStripeWebhook]]
    """

    kwargs = _get_kwargs(
        client=client,
        stripe_signature=stripe_signature,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Client,
    stripe_signature: Union[Unset, str] = UNSET,
) -> Optional[Union[HTTPValidationError, HandleStripeWebhookResponseHandleStripeWebhook]]:
    """Handle Stripe Webhook

    Args:
        stripe_signature (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, HandleStripeWebhookResponseHandleStripeWebhook]
    """

    return (
        await asyncio_detailed(
            client=client,
            stripe_signature=stripe_signature,
        )
    ).parsed
