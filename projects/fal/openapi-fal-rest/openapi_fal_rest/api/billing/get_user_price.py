from http import HTTPStatus
from typing import Any, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.run_type import RunType
from ...types import UNSET, Response


def _get_kwargs(
    *,
    run_type: RunType,
    machine_type: str,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_run_type = run_type.value
    params["run_type"] = json_run_type

    params["machine_type"] = machine_type

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/billing/user_price",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, float]]:
    if response.status_code == 200:
        response_200 = cast(float, response.json())
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
) -> Response[Union[HTTPValidationError, float]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    run_type: RunType,
    machine_type: str,
) -> Response[Union[HTTPValidationError, float]]:
    """Get User Price

    Args:
        run_type (RunType):
        machine_type (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, float]]
    """

    kwargs = _get_kwargs(
        run_type=run_type,
        machine_type=machine_type,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    run_type: RunType,
    machine_type: str,
) -> Optional[Union[HTTPValidationError, float]]:
    """Get User Price

    Args:
        run_type (RunType):
        machine_type (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, float]
    """

    return sync_detailed(
        client=client,
        run_type=run_type,
        machine_type=machine_type,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    run_type: RunType,
    machine_type: str,
) -> Response[Union[HTTPValidationError, float]]:
    """Get User Price

    Args:
        run_type (RunType):
        machine_type (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, float]]
    """

    kwargs = _get_kwargs(
        run_type=run_type,
        machine_type=machine_type,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    run_type: RunType,
    machine_type: str,
) -> Optional[Union[HTTPValidationError, float]]:
    """Get User Price

    Args:
        run_type (RunType):
        machine_type (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, float]
    """

    return (
        await asyncio_detailed(
            client=client,
            run_type=run_type,
            machine_type=machine_type,
        )
    ).parsed
