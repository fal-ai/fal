import datetime
from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.job_stats import JobStats
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    app_alias: str,
    start_time: datetime.datetime,
    end_time: Union[Unset, datetime.datetime] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["app_alias"] = app_alias

    json_start_time = start_time.isoformat()
    params["start_time"] = json_start_time

    json_end_time: Union[Unset, str] = UNSET
    if not isinstance(end_time, Unset):
        json_end_time = end_time.isoformat()
    params["end_time"] = json_end_time

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/usage/stats/idle_private_app",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, list["JobStats"]]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = JobStats.from_dict(response_200_item_data)

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
) -> Response[Union[HTTPValidationError, list["JobStats"]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    app_alias: str,
    start_time: datetime.datetime,
    end_time: Union[Unset, datetime.datetime] = UNSET,
) -> Response[Union[HTTPValidationError, list["JobStats"]]]:
    """Get Private App Idle Stats

    Args:
        app_alias (str):
        start_time (datetime.datetime):
        end_time (Union[Unset, datetime.datetime]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['JobStats']]]
    """

    kwargs = _get_kwargs(
        app_alias=app_alias,
        start_time=start_time,
        end_time=end_time,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    app_alias: str,
    start_time: datetime.datetime,
    end_time: Union[Unset, datetime.datetime] = UNSET,
) -> Optional[Union[HTTPValidationError, list["JobStats"]]]:
    """Get Private App Idle Stats

    Args:
        app_alias (str):
        start_time (datetime.datetime):
        end_time (Union[Unset, datetime.datetime]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['JobStats']]
    """

    return sync_detailed(
        client=client,
        app_alias=app_alias,
        start_time=start_time,
        end_time=end_time,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    app_alias: str,
    start_time: datetime.datetime,
    end_time: Union[Unset, datetime.datetime] = UNSET,
) -> Response[Union[HTTPValidationError, list["JobStats"]]]:
    """Get Private App Idle Stats

    Args:
        app_alias (str):
        start_time (datetime.datetime):
        end_time (Union[Unset, datetime.datetime]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['JobStats']]]
    """

    kwargs = _get_kwargs(
        app_alias=app_alias,
        start_time=start_time,
        end_time=end_time,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    app_alias: str,
    start_time: datetime.datetime,
    end_time: Union[Unset, datetime.datetime] = UNSET,
) -> Optional[Union[HTTPValidationError, list["JobStats"]]]:
    """Get Private App Idle Stats

    Args:
        app_alias (str):
        start_time (datetime.datetime):
        end_time (Union[Unset, datetime.datetime]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['JobStats']]
    """

    return (
        await asyncio_detailed(
            client=client,
            app_alias=app_alias,
            start_time=start_time,
            end_time=end_time,
        )
    ).parsed
