from http import HTTPStatus
from typing import Any, Dict, Optional, Union, cast

import httpx

from ... import errors
from ...client import Client
from ...models.http_validation_error import HTTPValidationError
from ...models.run_type import RunType
from ...types import UNSET, Response


def _get_kwargs(
    *,
    client: Client,
    run_type: RunType,
    machine_type: str,
) -> Dict[str, Any]:
    url = "{}/billing/user_price".format(client.base_url)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    json_run_type = run_type.value

    params["run_type"] = json_run_type

    params["machine_type"] = machine_type

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "follow_redirects": client.follow_redirects,
        "params": params,
    }


def _parse_response(*, client: Client, response: httpx.Response) -> Optional[Union[HTTPValidationError, float]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = cast(float, response.json())
        return response_200
    if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(*, client: Client, response: httpx.Response) -> Response[Union[HTTPValidationError, float]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Client,
    run_type: RunType,
    machine_type: str,
) -> Response[Union[HTTPValidationError, float]]:
    """Get User Price

    Args:
        run_type (RunType): An enumeration.
        machine_type (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, float]]
    """

    kwargs = _get_kwargs(
        client=client,
        run_type=run_type,
        machine_type=machine_type,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Client,
    run_type: RunType,
    machine_type: str,
) -> Optional[Union[HTTPValidationError, float]]:
    """Get User Price

    Args:
        run_type (RunType): An enumeration.
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
    client: Client,
    run_type: RunType,
    machine_type: str,
) -> Response[Union[HTTPValidationError, float]]:
    """Get User Price

    Args:
        run_type (RunType): An enumeration.
        machine_type (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, float]]
    """

    kwargs = _get_kwargs(
        client=client,
        run_type=run_type,
        machine_type=machine_type,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Client,
    run_type: RunType,
    machine_type: str,
) -> Optional[Union[HTTPValidationError, float]]:
    """Get User Price

    Args:
        run_type (RunType): An enumeration.
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
