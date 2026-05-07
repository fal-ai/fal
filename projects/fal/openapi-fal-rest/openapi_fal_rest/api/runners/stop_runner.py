from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.runner_stop_response import RunnerStopResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    runner_id: str,
    *,
    replace_first: Union[Unset, bool] = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["replace_first"] = replace_first

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/runners/{runner_id}/stop",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, RunnerStopResponse]]:
    if response.status_code == 200:
        response_200 = RunnerStopResponse.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, RunnerStopResponse]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    runner_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    replace_first: Union[Unset, bool] = False,
) -> Response[Union[HTTPValidationError, RunnerStopResponse]]:
    """Stop Runner

    Args:
        runner_id (str): The ID of the runner to stop
        replace_first (Union[Unset, bool]): Whether or not to replace this runner before stopping
            it Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, RunnerStopResponse]]
    """

    kwargs = _get_kwargs(
        runner_id=runner_id,
        replace_first=replace_first,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    runner_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    replace_first: Union[Unset, bool] = False,
) -> Optional[Union[HTTPValidationError, RunnerStopResponse]]:
    """Stop Runner

    Args:
        runner_id (str): The ID of the runner to stop
        replace_first (Union[Unset, bool]): Whether or not to replace this runner before stopping
            it Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, RunnerStopResponse]
    """

    return sync_detailed(
        runner_id=runner_id,
        client=client,
        replace_first=replace_first,
    ).parsed


async def asyncio_detailed(
    runner_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    replace_first: Union[Unset, bool] = False,
) -> Response[Union[HTTPValidationError, RunnerStopResponse]]:
    """Stop Runner

    Args:
        runner_id (str): The ID of the runner to stop
        replace_first (Union[Unset, bool]): Whether or not to replace this runner before stopping
            it Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, RunnerStopResponse]]
    """

    kwargs = _get_kwargs(
        runner_id=runner_id,
        replace_first=replace_first,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    runner_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    replace_first: Union[Unset, bool] = False,
) -> Optional[Union[HTTPValidationError, RunnerStopResponse]]:
    """Stop Runner

    Args:
        runner_id (str): The ID of the runner to stop
        replace_first (Union[Unset, bool]): Whether or not to replace this runner before stopping
            it Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, RunnerStopResponse]
    """

    return (
        await asyncio_detailed(
            runner_id=runner_id,
            client=client,
            replace_first=replace_first,
        )
    ).parsed
