from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.endpoint_billing_definition import EndpointBillingDefinition
from ...models.endpoint_provider_type import EndpointProviderType
from ...models.enterprise_status import EnterpriseStatus
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    endpoint: str,
    billing_unit: str,
    price: float,
    description: str,
    provider_type: Union[Unset, EndpointProviderType] = UNSET,
    balance_check: Union[Unset, bool] = False,
    use_compute_seconds: Union[Unset, bool] = False,
    enterprise_status: Union[Unset, EnterpriseStatus] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["endpoint"] = endpoint

    params["billing_unit"] = billing_unit

    params["price"] = price

    params["description"] = description

    json_provider_type: Union[Unset, str] = UNSET
    if not isinstance(provider_type, Unset):
        json_provider_type = provider_type.value

    params["provider_type"] = json_provider_type

    params["balance_check"] = balance_check

    params["use_compute_seconds"] = use_compute_seconds

    json_enterprise_status: Union[Unset, str] = UNSET
    if not isinstance(enterprise_status, Unset):
        json_enterprise_status = enterprise_status.value

    params["enterprise_status"] = json_enterprise_status

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/admin/endpoints/billing",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[EndpointBillingDefinition, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = EndpointBillingDefinition.from_dict(response.json())

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
) -> Response[Union[EndpointBillingDefinition, HTTPValidationError]]:
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
    billing_unit: str,
    price: float,
    description: str,
    provider_type: Union[Unset, EndpointProviderType] = UNSET,
    balance_check: Union[Unset, bool] = False,
    use_compute_seconds: Union[Unset, bool] = False,
    enterprise_status: Union[Unset, EnterpriseStatus] = UNSET,
) -> Response[Union[EndpointBillingDefinition, HTTPValidationError]]:
    """Upsert Endpoint Billing Definition

    Args:
        endpoint (str):
        billing_unit (str):
        price (float):
        description (str):
        provider_type (Union[Unset, EndpointProviderType]):
        balance_check (Union[Unset, bool]):  Default: False.
        use_compute_seconds (Union[Unset, bool]):  Default: False.
        enterprise_status (Union[Unset, EnterpriseStatus]): Enterprise readiness status for
            endpoints.

            READY: Endpoint meets enterprise compliance standards (DPA, no training on data)
            PENDING: Endpoint is under review or does not yet meet enterprise standards

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[EndpointBillingDefinition, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        endpoint=endpoint,
        billing_unit=billing_unit,
        price=price,
        description=description,
        provider_type=provider_type,
        balance_check=balance_check,
        use_compute_seconds=use_compute_seconds,
        enterprise_status=enterprise_status,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    endpoint: str,
    billing_unit: str,
    price: float,
    description: str,
    provider_type: Union[Unset, EndpointProviderType] = UNSET,
    balance_check: Union[Unset, bool] = False,
    use_compute_seconds: Union[Unset, bool] = False,
    enterprise_status: Union[Unset, EnterpriseStatus] = UNSET,
) -> Optional[Union[EndpointBillingDefinition, HTTPValidationError]]:
    """Upsert Endpoint Billing Definition

    Args:
        endpoint (str):
        billing_unit (str):
        price (float):
        description (str):
        provider_type (Union[Unset, EndpointProviderType]):
        balance_check (Union[Unset, bool]):  Default: False.
        use_compute_seconds (Union[Unset, bool]):  Default: False.
        enterprise_status (Union[Unset, EnterpriseStatus]): Enterprise readiness status for
            endpoints.

            READY: Endpoint meets enterprise compliance standards (DPA, no training on data)
            PENDING: Endpoint is under review or does not yet meet enterprise standards

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[EndpointBillingDefinition, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        endpoint=endpoint,
        billing_unit=billing_unit,
        price=price,
        description=description,
        provider_type=provider_type,
        balance_check=balance_check,
        use_compute_seconds=use_compute_seconds,
        enterprise_status=enterprise_status,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    endpoint: str,
    billing_unit: str,
    price: float,
    description: str,
    provider_type: Union[Unset, EndpointProviderType] = UNSET,
    balance_check: Union[Unset, bool] = False,
    use_compute_seconds: Union[Unset, bool] = False,
    enterprise_status: Union[Unset, EnterpriseStatus] = UNSET,
) -> Response[Union[EndpointBillingDefinition, HTTPValidationError]]:
    """Upsert Endpoint Billing Definition

    Args:
        endpoint (str):
        billing_unit (str):
        price (float):
        description (str):
        provider_type (Union[Unset, EndpointProviderType]):
        balance_check (Union[Unset, bool]):  Default: False.
        use_compute_seconds (Union[Unset, bool]):  Default: False.
        enterprise_status (Union[Unset, EnterpriseStatus]): Enterprise readiness status for
            endpoints.

            READY: Endpoint meets enterprise compliance standards (DPA, no training on data)
            PENDING: Endpoint is under review or does not yet meet enterprise standards

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[EndpointBillingDefinition, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        endpoint=endpoint,
        billing_unit=billing_unit,
        price=price,
        description=description,
        provider_type=provider_type,
        balance_check=balance_check,
        use_compute_seconds=use_compute_seconds,
        enterprise_status=enterprise_status,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    endpoint: str,
    billing_unit: str,
    price: float,
    description: str,
    provider_type: Union[Unset, EndpointProviderType] = UNSET,
    balance_check: Union[Unset, bool] = False,
    use_compute_seconds: Union[Unset, bool] = False,
    enterprise_status: Union[Unset, EnterpriseStatus] = UNSET,
) -> Optional[Union[EndpointBillingDefinition, HTTPValidationError]]:
    """Upsert Endpoint Billing Definition

    Args:
        endpoint (str):
        billing_unit (str):
        price (float):
        description (str):
        provider_type (Union[Unset, EndpointProviderType]):
        balance_check (Union[Unset, bool]):  Default: False.
        use_compute_seconds (Union[Unset, bool]):  Default: False.
        enterprise_status (Union[Unset, EnterpriseStatus]): Enterprise readiness status for
            endpoints.

            READY: Endpoint meets enterprise compliance standards (DPA, no training on data)
            PENDING: Endpoint is under review or does not yet meet enterprise standards

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[EndpointBillingDefinition, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            endpoint=endpoint,
            billing_unit=billing_unit,
            price=price,
            description=description,
            provider_type=provider_type,
            balance_check=balance_check,
            use_compute_seconds=use_compute_seconds,
            enterprise_status=enterprise_status,
        )
    ).parsed
