from http import HTTPStatus
from typing import Any, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    user_id: str,
    instance_type: str,
    *,
    is_draft: Union[Unset, bool] = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["is_draft"] = is_draft

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "delete",
        "url": f"/admin/compute/billing_overrides/{user_id}/{instance_type}",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, str]]:
    if response.status_code == 200:
        response_200 = cast(str, response.json())
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
) -> Response[Union[HTTPValidationError, str]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    user_id: str,
    instance_type: str,
    *,
    client: Union[AuthenticatedClient, Client],
    is_draft: Union[Unset, bool] = False,
) -> Response[Union[HTTPValidationError, str]]:
    """Delete Compute Billing Override

    Args:
        user_id (str):
        instance_type (str):
        is_draft (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, str]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        instance_type=instance_type,
        is_draft=is_draft,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    user_id: str,
    instance_type: str,
    *,
    client: Union[AuthenticatedClient, Client],
    is_draft: Union[Unset, bool] = False,
) -> Optional[Union[HTTPValidationError, str]]:
    """Delete Compute Billing Override

    Args:
        user_id (str):
        instance_type (str):
        is_draft (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, str]
    """

    return sync_detailed(
        user_id=user_id,
        instance_type=instance_type,
        client=client,
        is_draft=is_draft,
    ).parsed


async def asyncio_detailed(
    user_id: str,
    instance_type: str,
    *,
    client: Union[AuthenticatedClient, Client],
    is_draft: Union[Unset, bool] = False,
) -> Response[Union[HTTPValidationError, str]]:
    """Delete Compute Billing Override

    Args:
        user_id (str):
        instance_type (str):
        is_draft (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, str]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        instance_type=instance_type,
        is_draft=is_draft,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_id: str,
    instance_type: str,
    *,
    client: Union[AuthenticatedClient, Client],
    is_draft: Union[Unset, bool] = False,
) -> Optional[Union[HTTPValidationError, str]]:
    """Delete Compute Billing Override

    Args:
        user_id (str):
        instance_type (str):
        is_draft (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, str]
    """

    return (
        await asyncio_detailed(
            user_id=user_id,
            instance_type=instance_type,
            client=client,
            is_draft=is_draft,
        )
    ).parsed
